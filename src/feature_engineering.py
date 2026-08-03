"""
Feature Engineering 모듈
LSTM 입력을 위한 시퀀스 데이터 생성
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 예측 대상 구종 (빈도 낮은 건 OTHER로 묶음)
PITCH_CLASSES = {
    "FF": 0,  # 포심 패스트볼
    "SI": 1,  # 싱커
    "FC": 2,  # 커터
    "SL": 3,  # 슬라이더
    "CU": 4,  # 커브
    "CH": 5,  # 체인지업
    "FS": 6,  # 스플리터
    "OTHER": 7,
}

# 이전 몇 개 투구를 시퀀스로 볼 것인가.
# Day 2 실험: seq_len 3/5/8을 동일 시즌 데이터로 비교한 결과
#   seq_len=3 → 227,235 시퀀스, 전체 정확도 48.4%, macro-F1 43.4%
#   seq_len=5 → 61,318 시퀀스,  전체 정확도 46.6%, macro-F1 39.2% (기존 기본값)
#   seq_len=8 → 3,494 시퀀스,   전체 정확도 38.5%, macro-F1 16.3% (샘플 부족으로 불안정)
# seq_len이 커질수록 "직전 N구 + 1구가 필요"라는 조건 때문에 학습 가능한 타석 수가
# 급격히 줄어들어, 문맥 정보가 늘어나는 이득보다 데이터가 줄어드는 손해가 더 컸다.
# 그래서 기본값을 5 → 3으로 낮췄다. 자세한 내용은 docs/blog/day2.md 참고.
SEQUENCE_LENGTH = 3


def map_pitch_type(pitch: str) -> int:
    """Statcast 구종 코드를 학습용 정수 라벨로 변환. 미등록 구종은 OTHER로 묶는다."""
    return PITCH_CLASSES.get(pitch, PITCH_CLASSES["OTHER"])


_PITCH_CODES = ["FF", "SI", "FC", "SL", "CU", "CH", "FS"]


def _add_pitcher_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """
    투수 성향 피처 3종 추가:
    1. 투수 전체 구종 비율 (p_ff_pct … p_fs_pct) — 7개
    2. 투수×카운트 조합에서 가장 자주 던지는 구종 — 1개
    3. 투수×타자 역대 상대 횟수 — 1개

    투수마다 구종 레퍼토리와 카운트별 패턴이 크게 다르기 때문에, 시퀀스/컨텍스트
    피처만으로는 "이 투수는 원래 슬라이더를 잘 안 던진다" 같은 개인 성향을 모델이
    학습하기 어렵다. 이 함수는 그 성향을 명시적인 피처로 미리 계산해서 넣어준다.
    """
    df = df.copy()

    # 1. 투수별 구종 비율
    for code in _PITCH_CODES:
        col = f"p_{code.lower()}_pct"
        df[col] = df.groupby("pitcher")["pitch_type"].transform(
            lambda x, c=code: (x == c).mean()
        )

    # 2. 투수×카운트 최빈 구종
    df["_count_key"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    top_by_count = (
        df.groupby(["pitcher", "_count_key"])["pitch_type"]
        .agg(lambda x: x.mode().iloc[0] if len(x) > 0 else "FF")
        .reset_index()
        .rename(columns={"pitch_type": "_top_pitch_at_count"})
    )
    df = df.merge(top_by_count, on=["pitcher", "_count_key"], how="left")
    df["pitcher_count_top_enc"] = df["_top_pitch_at_count"].apply(map_pitch_type)
    df.drop(columns=["_count_key", "_top_pitch_at_count"], inplace=True)

    # 3. 투수×타자 누적 상대 횟수 (현재 투구 이전)
    df["matchup_count"] = df.groupby(["pitcher", "batter"]).cumcount()

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Statcast 원본 투구 단위 데이터에 모델 입력용 파생 피처를 추가한다."""
    df = df.copy()

    df["pitch_label"] = df["pitch_type"].apply(map_pitch_type)
    df["score_diff"] = df["fld_score"] - df["bat_score"]
    df["stand_enc"] = (df["stand"] == "R").astype(int)
    df["p_throws_enc"] = (df["p_throws"] == "R").astype(int)
    df["inning_top"] = (df["inning_topbot"] == "Top").astype(int)

    # 주자 상황: 각 루를 독립 바이너리 플래그로 표현
    df["on_1b_flag"] = df["on_1b"].notna().astype(int)
    df["on_2b_flag"] = df["on_2b"].notna().astype(int)
    df["on_3b_flag"] = df["on_3b"].notna().astype(int)

    # 아웃카운트: 원-핫 인코딩
    df["out_0"] = (df["outs_when_up"] == 0).astype(int)
    df["out_1"] = (df["outs_when_up"] == 1).astype(int)
    df["out_2"] = (df["outs_when_up"] == 2).astype(int)

    # 수치형 결측치 중앙값으로 대체
    num_cols = ["release_speed", "release_spin_rate", "pfx_x", "pfx_z", "plate_x", "plate_z"]
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # 투수 성향 피처 추가
    df = _add_pitcher_tendency(df)

    return df


def build_sequences(
    df: pd.DataFrame,
    seq_len: int = SEQUENCE_LENGTH,
    scaler: Optional[StandardScaler] = None,
    scaler_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    타석(at_bat_number) 단위로 LSTM 시퀀스 생성.

    각 타석 내에서 직전 seq_len개의 투구를 시퀀스 입력으로, 그다음 투구의 구종을
    라벨로 사용한다. 타석 길이가 seq_len + 1보다 짧으면(즉 예측할 다음 구가 없으면)
    건너뛴다.

    Args:
        df: Statcast 원본 투구 단위 데이터프레임.
        seq_len: 시퀀스로 사용할 직전 투구 개수.
        scaler: 기학습된 scaler 전달 시 fit 없이 transform만 적용 (평가/추론용).
            None이면 새로 fit한다 (학습용).
        scaler_path: scaler 저장 경로. None이면 저장하지 않는다.

    Returns:
        (X_seq, X_ctx, y, pitcher_ids, batter_ids, scaler) 튜플.
        X_seq: (N, seq_len, seq_feat_dim) 시퀀스 피처
        X_ctx: (N, ctx_feat_dim) 상황 컨텍스트 피처
        y: (N,) 다음 구종 라벨
        pitcher_ids / batter_ids: (N,) 원본 MLB 선수 ID
    """
    df = build_features(df)

    # pitch_label은 범주형이므로 스케일링 제외
    scale_cols = ["release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z"]
    seq_features = scale_cols + ["pitch_label"]
    ctx_features = [
        "balls", "strikes",
        "out_0", "out_1", "out_2",
        "inning", "inning_top",
        "on_1b_flag", "on_2b_flag", "on_3b_flag",
        "score_diff",
        "stand_enc", "p_throws_enc",
        # 투수 성향 피처
        "p_ff_pct", "p_si_pct", "p_fc_pct", "p_sl_pct",
        "p_cu_pct", "p_ch_pct", "p_fs_pct",
        "pitcher_count_top_enc",
        "matchup_count",
    ]

    # 스케일러 fit (학습용) 또는 재사용 (평가용)
    available_scale_cols = [c for c in scale_cols if c in df.columns]
    if scaler is None:
        scaler = StandardScaler()
        df[available_scale_cols] = scaler.fit_transform(df[available_scale_cols])
    else:
        df[available_scale_cols] = scaler.transform(df[available_scale_cols])

    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"[저장] scaler → {scaler_path}")

    X_seq_list, X_ctx_list, y_list = [], [], []
    pitcher_list, batter_list = [], []

    groups = df.groupby(["game_date", "pitcher", "batter", "at_bat_number"])

    for _, ab in groups:
        ab = ab.sort_values("pitch_number").reset_index(drop=True)

        if len(ab) < seq_len + 1:
            continue

        for i in range(seq_len, len(ab)):
            window = ab.iloc[i - seq_len:i]
            target = ab.iloc[i]

            seq_cols = [f for f in seq_features if f in window.columns]
            seq = window[seq_cols].values.astype(np.float32)
            if seq.shape[0] != seq_len:
                continue

            ctx_cols = [f for f in ctx_features if f in target.index]
            ctx = target[ctx_cols].values.astype(np.float32)

            X_seq_list.append(seq)
            X_ctx_list.append(ctx)
            y_list.append(int(target["pitch_label"]))
            pitcher_list.append(int(target["pitcher"]))
            batter_list.append(int(target["batter"]))

    return (
        np.array(X_seq_list),
        np.array(X_ctx_list),
        np.array(y_list),
        np.array(pitcher_list),
        np.array(batter_list),
        scaler,
    )


if __name__ == "__main__":
    # 간단한 동작 확인용. 실제 학습은 src/model.py를 통해 전체 시즌 데이터로 실행한다.
    sample_path = os.path.join("data", "raw", "statcast_2025-04.csv")
    df = pd.read_csv(sample_path)
    X_seq, X_ctx, y, pitchers, batters, _ = build_sequences(df)
    print(f"X_seq  : {X_seq.shape}")
    print(f"X_ctx  : {X_ctx.shape}")
    print(f"y      : {y.shape}")
    print(f"구종 분포: {np.bincount(y)}")
