"""
Feature Engineering 모듈
LSTM 입력을 위한 시퀀스 데이터 생성
"""

import pandas as pd
import numpy as np
import joblib
import os
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

SEQUENCE_LENGTH = 5  # 이전 몇 개 투구를 시퀀스로 볼 것인가


def map_pitch_type(pitch: str) -> int:
    return PITCH_CLASSES.get(pitch, PITCH_CLASSES["OTHER"])


_PITCH_CODES = ["FF", "SI", "FC", "SL", "CU", "CH", "FS"]


def _add_pitcher_tendency(df: pd.DataFrame) -> pd.DataFrame:
    """
    투수 성향 피처 3종 추가:
    1. 투수 전체 구종 비율 (p_ff_pct … p_fs_pct) — 7개
    2. 투수×카운트 조합에서 가장 자주 던지는 구종 — 1개
    3. 투수×타자 역대 상대 횟수 — 1개
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
    df = df.copy()

    df["pitch_label"]  = df["pitch_type"].apply(map_pitch_type)
    df["score_diff"]   = df["fld_score"] - df["bat_score"]
    df["stand_enc"]    = (df["stand"]    == "R").astype(int)
    df["p_throws_enc"] = (df["p_throws"] == "R").astype(int)
    df["inning_top"]   = (df["inning_topbot"] == "Top").astype(int)

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


def build_sequences(df: pd.DataFrame, seq_len: int = SEQUENCE_LENGTH,
                    scaler: StandardScaler = None, scaler_path: str = None):
    """
    타석(at_bat_number) 단위로 LSTM 시퀀스 생성

    Args:
        scaler:      기학습된 scaler 전달 시 fit 없이 transform만 적용 (평가/추론용)
        scaler_path: scaler 저장 경로 (None이면 저장 안 함)

    Returns:
        X_seq, X_ctx, y, pitcher_ids, batter_ids, scaler
    """
    df = build_features(df)

    # pitch_label은 범주형이므로 스케일링 제외
    scale_cols  = ["release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z"]
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
    df = pd.read_csv("data/raw/statcast_2023_04_sample.csv")
    X_seq, X_ctx, y, pitchers, batters = build_sequences(df)
    print(f"X_seq  : {X_seq.shape}")
    print(f"X_ctx  : {X_ctx.shape}")
    print(f"y      : {y.shape}")
    print(f"구종 분포: {np.bincount(y)}")
