"""
Phase 3: YOLO 감지 이벤트 → BiLSTM 예측 파이프라인
실행: python src/pipeline.py --pitcher-id 605483
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PITCH_LABEL_MAP = {
    0: "FF",    # 포심 패스트볼
    1: "SI",    # 싱커
    2: "FC",    # 커터
    3: "SL",    # 슬라이더
    4: "CU",    # 커브
    5: "CH",    # 체인지업
    6: "FS",    # 스플리터
    7: "OTHER",
}
PITCH_NAME_KO = {
    "FF": "포심", "SI": "싱커", "FC": "커터",
    "SL": "슬라이더", "CU": "커브", "CH": "체인지업",
    "FS": "스플리터", "OTHER": "기타",
}

SEQ_FEAT_COLS = ["release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z", "pitch_label"]
CTX_FEAT_COLS = [
    "balls", "strikes",
    "out_0", "out_1", "out_2",
    "inning", "inning_top",
    "on_1b_flag", "on_2b_flag", "on_3b_flag",
    "score_diff",
    "stand_enc", "p_throws_enc",
    "p_ff_pct", "p_si_pct", "p_fc_pct", "p_sl_pct",
    "p_cu_pct", "p_ch_pct", "p_fs_pct",
    "pitcher_count_top_enc",
    "matchup_count",
]
SEQ_LEN = 5


def find_pitcher_in_text(text: str, data_path: str = None) -> list[dict]:
    """
    영상 제목 등 자유 텍스트에서 투수 이름 자동 탐색.
    한국어 이름도 영어 로마자로 변환 후 검색.
    """
    import re

    KO_EN = {
        "오타니": "Ohtani", "야마모토": "Yamamoto", "다르빗슈": "Darvish",
        "류현진": "Ryu", "최지만": "Choi", "사사키": "Sasaki",
        "마에다": "Maeda", "다나카": "Tanaka", "코울": "Cole",
        "버랜더": "Verlander", "슈어저": "Scherzer", "게릿": "Gerrit",
        "이정후": "Lee", "김하성": "Kim Ha",
    }
    for kor, eng in KO_EN.items():
        text = text.replace(kor, eng)

    tokens = re.split(r"[\s|,\-_\.·]+", text)
    seen: set[int] = set()
    results: list[dict] = []

    for i, tok in enumerate(tokens):
        candidates = [tok]
        if i + 1 < len(tokens):
            candidates.append(f"{tok} {tokens[i + 1]}")
        for c in candidates:
            c = c.strip()
            if len(c) < 3:
                continue
            for r in find_pitcher(c, data_path):
                if r["pitcher"] not in seen:
                    seen.add(r["pitcher"])
                    results.append(r)

    return sorted(results, key=lambda x: x["pitch_count"], reverse=True)[:5]


def find_pitcher(name: str, data_path: str = None) -> list[dict]:
    """
    투수 이름으로 ID 검색
    Returns: [{"pitcher": int, "player_name": str, "pitch_count": int}, ...]
    """
    data_path = data_path or os.path.join(ROOT, "data", "raw", "statcast_2025_full.csv")
    df = pd.read_csv(data_path, usecols=["pitcher", "player_name"])
    mask = df["player_name"].str.contains(name, case=False, na=False)
    matches = df[mask].groupby(["pitcher", "player_name"]).size().reset_index(name="pitch_count")
    return matches.to_dict("records")


class PitchPipeline:
    """
    투구 이벤트(YOLO)를 받아 다음 구종을 예측하는 파이프라인.

    사용법:
        pipeline = PitchPipeline(pitcher_id=605483)
        for yolo_event in events:
            result = pipeline.on_pitch_detected(yolo_event)
            print(result["next_pitch"], result["confidence"])
    """

    def __init__(
        self,
        pitcher_id: int,
        data_path: str = None,
        model_path: str = None,
        scaler_path: str = None,
    ):
        from tensorflow import keras
        import sys
        sys.path.insert(0, os.path.join(ROOT, "src"))
        from feature_engineering import build_features

        data_path   = data_path   or os.path.join(ROOT, "data", "raw", "statcast_2025_full.csv")
        model_path  = model_path  or os.path.join(ROOT, "models", "pitch_predictor.h5")
        scaler_path = scaler_path or os.path.join(ROOT, "models", "scaler.pkl")

        df_raw = pd.read_csv(data_path)
        df_pitcher = df_raw[df_raw["pitcher"] == pitcher_id]
        if len(df_pitcher) == 0:
            raise ValueError(f"투수 ID {pitcher_id} 데이터 없음. find_pitcher()로 ID 확인")

        player_name = df_pitcher["player_name"].iloc[0] if "player_name" in df_pitcher.columns else str(pitcher_id)
        print(f"[파이프라인] 투수: {player_name} (ID={pitcher_id})  투구수: {len(df_pitcher)}")

        df = build_features(df_pitcher)

        # 학습 때 사용한 scaler로 수치 피처 정규화
        scaler = joblib.load(scaler_path)
        scale_cols = ["release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z"]
        available  = [c for c in scale_cols if c in df.columns]
        df[available] = scaler.transform(df[available])

        self._df = df.sort_values(
            ["game_date", "at_bat_number", "pitch_number"]
        ).reset_index(drop=True)

        self._model       = keras.models.load_model(model_path)
        self._pitcher_id  = pitcher_id
        self._pitch_idx   = 0

    # ── 상태 조회 ─────────────────────────────────────────────────

    @property
    def total_pitches(self) -> int:
        return len(self._df)

    @property
    def pitch_idx(self) -> int:
        return self._pitch_idx

    @property
    def exhausted(self) -> bool:
        return self._pitch_idx >= len(self._df)

    def reset(self):
        self._pitch_idx = 0

    # ── 경기 상황 ─────────────────────────────────────────────────

    def get_game_context(self, idx: int) -> dict:
        """특정 투구 인덱스의 경기 상황 반환 (feature_engineering 인코딩 기준)"""
        if idx >= len(self._df):
            return {}
        row = self._df.iloc[idx]

        def _int(key, d=0):
            return int(row[key]) if key in row.index else d

        def _bool(key):
            return bool(row[key]) if key in row.index else False

        # out_0/1/2 원-핫 → 아웃카운트 복원
        outs = next(
            (i for i, col in enumerate(["out_0", "out_1", "out_2"])
             if col in row.index and int(row[col]) == 1),
            0,
        )
        # stand_enc: 1=R, 0=L  /  inning_top: 1=Top, 0=Bottom
        return {
            "balls":      _int("balls"),
            "strikes":    _int("strikes"),
            "outs":       outs,
            "inning":     _int("inning", 1),
            "inning_top": _bool("inning_top"),
            "on_1b":      _bool("on_1b_flag"),
            "on_2b":      _bool("on_2b_flag"),
            "on_3b":      _bool("on_3b_flag"),
            "score_diff": _int("score_diff"),
            "batter_id":  _int("batter"),
            "stand":      "R" if _int("stand_enc") == 1 else "L",
        }

    def run_all(self, yolo_events: list[dict] = None) -> list[dict]:
        """전체 투구에 대해 예측 실행. 각 결과에 game_context 포함."""
        self.reset()
        yolo_events = yolo_events or []
        results = []

        while not self.exhausted:
            pre_idx = self._pitch_idx
            ye = yolo_events[pre_idx] if pre_idx < len(yolo_events) else None
            r = self.on_pitch_detected(ye)
            if r is not None:
                r["game_context"] = self.get_game_context(pre_idx)
                results.append(r)

        self.reset()
        return results

    # ── 핵심 메서드 ───────────────────────────────────────────────

    def on_pitch_detected(self, yolo_event: dict = None) -> dict | None:
        """
        YOLO가 투구 이벤트를 감지할 때마다 호출.
        Statcast 시퀀스를 한 칸 advance하고 다음 투구 예측 반환.

        Args:
            yolo_event: yolo_detector.process_video()의 pitch_events 원소 (선택)
                        {"start_frame", "end_frame", "trajectory", "estimated_speed_mph"}

        Returns:
            {
              "pitch_idx":        int,
              "current_pitch":    str,          # 방금 던진 구종 코드 (FF, SL …)
              "current_pitch_ko": str,
              "next_pitch":       str | None,   # 예측된 다음 구종
              "next_pitch_ko":    str | None,
              "confidence":       float | None,
              "probabilities":    dict[str, float],
              "yolo_speed_mph":   float | None,
              "note":             str | None,
            }
            or None — 시퀀스 소진
        """
        if self.exhausted:
            return None

        row = self._df.iloc[self._pitch_idx]
        current_code = PITCH_LABEL_MAP.get(int(row["pitch_label"]), "OTHER")

        result: dict = {
            "pitch_idx":        self._pitch_idx,
            "current_pitch":    current_code,
            "current_pitch_ko": PITCH_NAME_KO.get(current_code, current_code),
            "next_pitch":       None,
            "next_pitch_ko":    None,
            "confidence":       None,
            "probabilities":    {},
            "yolo_speed_mph":   yolo_event.get("estimated_speed_mph") if yolo_event else None,
            "note":             None,
        }

        # SEQ_LEN 미만이면 예측 불가 (시퀀스 누적 단계)
        if self._pitch_idx < SEQ_LEN:
            result["note"] = f"시퀀스 누적 중 ({self._pitch_idx + 1}/{SEQ_LEN})"
            self._pitch_idx += 1
            return result

        # 직전 SEQ_LEN 개 투구로 시퀀스 구성
        window   = self._df.iloc[self._pitch_idx - SEQ_LEN : self._pitch_idx]
        seq_cols = [c for c in SEQ_FEAT_COLS if c in window.columns]
        X_seq    = window[seq_cols].values.astype(np.float32)[np.newaxis]  # (1, 5, 6)

        ctx_cols = [c for c in CTX_FEAT_COLS if c in row.index]
        X_ctx    = row[ctx_cols].values.astype(np.float32)[np.newaxis]     # (1, 13)

        # 모델 학습 시 _encode_ids와 동일하게 % vocab_size 처리
        pitcher_enc = np.array([[int(row["pitcher"]) % 2000]])
        batter_enc  = np.array([[int(row.get("batter", 0)) % 3000]])

        probs = self._model.predict(
            {
                "seq_input":     X_seq,
                "ctx_input":     X_ctx,
                "pitcher_input": pitcher_enc,
                "batter_input":  batter_enc,
            },
            verbose=0,
        )[0]

        next_idx  = int(np.argmax(probs))
        next_code = PITCH_LABEL_MAP[next_idx]

        result.update({
            "next_pitch":    next_code,
            "next_pitch_ko": PITCH_NAME_KO.get(next_code, next_code),
            "confidence":    round(float(probs[next_idx]), 3),
            "probabilities": {
                PITCH_LABEL_MAP[i]: round(float(p), 3)
                for i, p in enumerate(probs)
            },
        })

        self._pitch_idx += 1
        return result


# ── CLI 데모 ──────────────────────────────────────────────────────

def _demo(pitcher_id: int, n_pitches: int, data_path: str, model_path: str, scaler_path: str):
    pipeline = PitchPipeline(
        pitcher_id=pitcher_id,
        data_path=data_path,
        model_path=model_path,
        scaler_path=scaler_path,
    )

    print(f"\n{'─'*60}")
    print(f"  투구#   현재 구종        예측 다음 구종    신뢰도")
    print(f"{'─'*60}")

    for _ in range(min(n_pitches, pipeline.total_pitches)):
        r = pipeline.on_pitch_detected()
        if r is None:
            break

        note = f"  ← {r['note']}" if r["note"] else ""
        if r["next_pitch"]:
            print(
                f"  [{r['pitch_idx']:>3}]  {r['current_pitch']:<6} ({r['current_pitch_ko']:<5})"
                f"  →  {r['next_pitch']:<6} ({r['next_pitch_ko']:<5})"
                f"  {r['confidence']:.1%}{note}"
            )
        else:
            print(
                f"  [{r['pitch_idx']:>3}]  {r['current_pitch']:<6} ({r['current_pitch_ko']:<5}){note}"
            )

    print(f"{'─'*60}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ROOT)

    parser = argparse.ArgumentParser(description="투구 예측 파이프라인 데모")
    parser.add_argument("--pitcher-id", type=int,  default=None, help="MLB 투수 ID")
    parser.add_argument("--search",     type=str,  default=None, help="투수 이름 검색")
    parser.add_argument("--n-pitches",  type=int,  default=20,   help="데모 투구 수")
    parser.add_argument("--data",       default=None)
    parser.add_argument("--model",      default=None)
    parser.add_argument("--scaler",     default=None)
    args = parser.parse_args()

    data_path = args.data or os.path.join(ROOT, "data", "raw", "statcast_2025_full.csv")

    if args.search:
        results = find_pitcher(args.search, data_path)
        if not results:
            print(f"'{args.search}' 검색 결과 없음")
        else:
            print(f"\n투수 검색 결과 '{args.search}':")
            for r in results:
                print(f"  ID={r['pitcher']:>7}  {r['player_name']:<30}  투구수={r['pitch_count']}")
        sys.exit(0)

    if args.pitcher_id is None:
        parser.error("--pitcher-id 또는 --search 중 하나 필요")

    _demo(
        pitcher_id=args.pitcher_id,
        n_pitches=args.n_pitches,
        data_path=data_path,
        model_path=args.model,
        scaler_path=args.scaler,
    )
