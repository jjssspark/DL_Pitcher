"""
EDA (탐색적 데이터 분석)
실행: python notebooks/eda.py
결과 이미지: notebooks/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "raw", "statcast_2023_04_sample.csv")
FIG_DIR   = os.path.join(ROOT, "notebooks", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["figure.dpi"] = 120
sns.set_theme(style="whitegrid")


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print(f"[오류] 데이터 없음: {DATA_PATH}")
        print("먼저 실행하세요: python src/data_collector.py")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    print(f"[로드] {len(df):,}개 투구, {df.shape[1]}개 컬럼")
    return df


def analyze_pitch_distribution(df: pd.DataFrame):
    """1. 구종 분포"""
    counts = df["pitch_type"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(counts.index, counts.values, color=sns.color_palette("Set2", len(counts)))
    axes[0].set_title("구종 분포 (빈도)")
    axes[0].set_xlabel("구종")
    axes[0].set_ylabel("투구 수")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 50, str(v), ha="center", fontsize=9)

    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=sns.color_palette("Set2", len(counts)))
    axes[1].set_title("구종 비율")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "01_pitch_distribution.png"))
    print("[저장] 01_pitch_distribution.png")
    plt.show()
    print(counts.to_string())


def analyze_speed_by_pitch(df: pd.DataFrame):
    """2. 구종별 구속 분포"""
    df_clean    = df.dropna(subset=["release_speed"])
    top_pitches = df_clean["pitch_type"].value_counts().head(6).index
    data        = [df_clean[df_clean["pitch_type"] == p]["release_speed"].values
                   for p in top_pitches]

    fig, ax = plt.subplots(figsize=(12, 6))
    bp      = ax.boxplot(data, labels=top_pitches, patch_artist=True)
    for patch, color in zip(bp["boxes"], sns.color_palette("Set2", len(top_pitches))):
        patch.set_facecolor(color)
    ax.set_title("구종별 구속 분포 (mph)")
    ax.set_xlabel("구종")
    ax.set_ylabel("구속 (mph)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "02_speed_by_pitch.png"))
    print("[저장] 02_speed_by_pitch.png")
    plt.show()


def analyze_count_pattern(df: pd.DataFrame):
    """3. 볼카운트별 구종 선택 패턴"""
    df      = df.copy()
    df["count"]  = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    top_pitches  = df["pitch_type"].value_counts().head(5).index
    pivot        = (df[df["pitch_type"].isin(top_pitches)]
                    .groupby(["count", "pitch_type"]).size()
                    .unstack(fill_value=0))
    pivot_pct    = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot_pct.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("볼카운트별 구종 선택 비율 (%)")
    ax.set_xlabel("볼카운트 (볼-스트라이크)")
    ax.set_ylabel("비율 (%)")
    ax.legend(title="구종", bbox_to_anchor=(1.01, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "03_count_pattern.png"))
    print("[저장] 03_count_pattern.png")
    plt.show()


def analyze_runner_pattern(df: pd.DataFrame):
    """4. 주자 상황별 구종 변화 (1,3루 / 2,3루 포함)"""
    df = df.copy()

    def runner_label(row):
        parts = []
        if pd.notna(row.get("on_1b")): parts.append("1루")
        if pd.notna(row.get("on_2b")): parts.append("2루")
        if pd.notna(row.get("on_3b")): parts.append("3루")
        return ",".join(parts) if parts else "주자없음"

    df["runner_label"] = df.apply(runner_label, axis=1)
    top_pitches = df["pitch_type"].value_counts().head(4).index
    pivot       = (df[df["pitch_type"].isin(top_pitches)]
                   .groupby(["runner_label", "pitch_type"]).size()
                   .unstack(fill_value=0))
    pivot_pct   = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_pct.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("주자 상황별 구종 비율 (%)")
    ax.set_xlabel("주자 상황")
    ax.set_ylabel("비율 (%)")
    ax.legend(title="구종", bbox_to_anchor=(1.01, 1))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "04_runner_pattern.png"))
    print("[저장] 04_runner_pattern.png")
    plt.show()


def analyze_out_pattern(df: pd.DataFrame):
    """5. 아웃카운트별 구종 변화"""
    top_pitches = df["pitch_type"].value_counts().head(5).index
    pivot       = (df[df["pitch_type"].isin(top_pitches)]
                   .groupby(["outs_when_up", "pitch_type"]).size()
                   .unstack(fill_value=0))
    pivot_pct   = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_pct.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("아웃카운트별 구종 비율 (%)")
    ax.set_xlabel("아웃카운트")
    ax.set_xticklabels(["0아웃", "1아웃", "2아웃"], rotation=0)
    ax.set_ylabel("비율 (%)")
    ax.legend(title="구종", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "05_out_pattern.png"))
    print("[저장] 05_out_pattern.png")
    plt.show()


def analyze_missing_values(df: pd.DataFrame):
    """6. 결측치 현황"""
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    print("\n[결측치 비율 (%)]\n", missing.round(2).to_string())

    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.4)))
        ax.barh(missing.index, missing.values, color="salmon")
        ax.set_title("컬럼별 결측 비율 (%)")
        ax.set_xlabel("결측 비율 (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "06_missing_values.png"))
        print("[저장] 06_missing_values.png")
        plt.show()


def print_summary(df: pd.DataFrame):
    print("\n" + "="*50)
    print(f"총 투구 수   : {len(df):,}")
    print(f"고유 투수 수 : {df['pitcher'].nunique()}")
    print(f"고유 타자 수 : {df['batter'].nunique()}")
    print(f"기간         : {df['game_date'].min()} ~ {df['game_date'].max()}")
    print(f"구종 종류    : {df['pitch_type'].nunique()}종")
    print(f"평균 구속    : {df['release_speed'].mean():.1f} mph")
    print("="*50)


if __name__ == "__main__":
    df = load_data()
    print_summary(df)
    analyze_pitch_distribution(df)
    analyze_speed_by_pitch(df)
    analyze_count_pattern(df)
    analyze_runner_pattern(df)
    analyze_out_pattern(df)
    analyze_missing_values(df)
    print(f"\n[완료] 그래프 저장: {FIG_DIR}")
