"""
특징 집합을 바꿔가며 홀드아웃 성능을 비교한다 (절제 실험).

기존 7개 특징은 전부 궤적의 '모양'이라 OFFSPEED f1이 0.14에 묶여 있었다. 체인지업·
스플리터는 패스트볼과 모양이 비슷하고 속도만 다르기 때문이다. 여기서는 시간 계열과
기하 계열을 따로 붙여, **어느 계열이 OFFSPEED를 움직이는지** 를 분리해서 본다.

정확도만 보지 않는다. 셋을 함께 본다 (판정 규칙, docs/NEXT_SESSION.md):
  1. 홀드아웃 정확도 — 기준선은 최빈값(약 0.483)이지 33% 랜덤이 아니다
  2. 특징 중요도 최대/최소 비 — 1에 가까우면 정확도와 무관하게 무신호다 (TS-014)
  3. 순열검정 — --permutations N (기본 0, 생략)
그리고 전체 정확도는 OFFSPEED를 전부 틀려도 별로 안 떨어지므로 OFFSPEED의
precision/recall을 따로 찍는다.

실행:
  venv/bin/python3 scripts/eval_feature_sets.py
  venv/bin/python3 scripts/eval_feature_sets.py --permutations 1000
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, precision_recall_fscore_support  # noqa: E402

from pitch_type_cv.group_classifier import train_classifier  # noqa: E402
from pitch_type_cv.trajectory_features import GEOMETRY_ONLY_COLUMNS  # noqa: E402

OUT_DIR = os.path.join(ROOT, "output", "pitch_type_cv")
DATASET_PATH = os.path.join(OUT_DIR, "dataset_clips.csv")

# 홀드아웃 경기. 813027은 OFFSPEED 비율이 12.2%로 4경기 중 중간이라 홀드아웃이 특정
# 구종에 치우치지 않는다 (노트북 셀 3과 같은 값을 쓴다 — 다르면 비교가 무의미하다).
HOLDOUT_GAME_PK = 813027

# 시간 계열: 같은 구간을 지나는 데 걸린 프레임 = 속도의 역수. 좌표만으로 남는 속도 단서다.
TIME_COLUMNS = ["frame_span", "end_frame", "speed_ratio_late_early"]
# 기하 계열: 릴리스 위치와 낙차의 분포. 모양이지만 기존 7개가 재지 않던 축이다.
GEOMETRY_EXTRA_COLUMNS = ["release_x", "release_y", "late_drop_ratio", "vertical_accel_px"]

FEATURE_SETS = {
    "기준(7)": GEOMETRY_ONLY_COLUMNS,
    "+시간(10)": GEOMETRY_ONLY_COLUMNS + TIME_COLUMNS,
    "+기하(11)": GEOMETRY_ONLY_COLUMNS + GEOMETRY_EXTRA_COLUMNS,
    "전체(14)": GEOMETRY_ONLY_COLUMNS + TIME_COLUMNS + GEOMETRY_EXTRA_COLUMNS,
}

LABELS = ["FASTBALL", "BREAKING", "OFFSPEED"]


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATASET_PATH)
    df = df[df["has_trajectory"]].reset_index(drop=True)
    train_df = df[df["game_pk"] != HOLDOUT_GAME_PK].reset_index(drop=True)
    holdout_df = df[df["game_pk"] == HOLDOUT_GAME_PK].reset_index(drop=True)
    if train_df.empty or holdout_df.empty:
        raise SystemExit(f"[중단] game_pk={HOLDOUT_GAME_PK} 홀드아웃 분리 실패")
    return train_df, holdout_df


def evaluate(
    columns: list[str], train_df: pd.DataFrame, holdout_df: pd.DataFrame
) -> tuple[dict, pd.Series]:
    model = train_classifier(train_df, train_df["group"].tolist(), feature_columns=columns)
    y_true = holdout_df["group"].tolist()
    y_pred = model.predict(holdout_df[columns]).tolist()

    accuracy = accuracy_score(y_true, y_pred)
    majority = train_df["group"].value_counts().idxmax()
    baseline = accuracy_score(y_true, [majority] * len(y_true))

    precision, recall, f1, _support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    importances = pd.Series(model.feature_importances_, index=columns)
    # 중요도가 정확히 0인 특징은 '한 번도 분기에 쓰이지 않았다'는 뜻이다. 0으로 나누는
    # 대신 무한대로 남겨 눈에 띄게 둔다 — 평평함(무신호)과는 반대 방향의 신호다.
    spread = float("inf") if importances.min() == 0 else importances.max() / importances.min()

    result = {
        "특징수": len(columns),
        "정확도": accuracy,
        "기준선": baseline,
        "기준선대비": accuracy - baseline,
        "중요도비": spread,
    }
    for i, label in enumerate(LABELS):
        result[f"{label}_precision"] = precision[i]
        result[f"{label}_recall"] = recall[i]
        result[f"{label}_f1"] = f1[i]
    return result, importances


def permutation_test(
    columns: list[str], train_df: pd.DataFrame, holdout_df: pd.DataFrame,
    accuracy: float, n: int,
) -> tuple[np.ndarray, float]:
    """라벨을 섞어 학습했을 때의 정확도 분포와 비교한다."""
    rng = np.random.default_rng(0)
    y_true = holdout_df["group"].tolist()
    scores = []
    for i in range(n):
        shuffled = rng.permutation(train_df["group"].values)
        null_model = train_classifier(train_df, list(shuffled), feature_columns=columns)
        scores.append(accuracy_score(y_true, null_model.predict(holdout_df[columns])))
        if (i + 1) % 100 == 0:
            print(f"  순열검정 {i + 1}/{n}", flush=True)
    scores = np.array(scores)
    return scores, float((scores >= accuracy).mean())


def plot_comparison(table: pd.DataFrame, path: str) -> None:
    plt.rcParams["font.family"] = "AppleGothic"   # 한글 tofu 방지 (TS-004)
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax_acc, ax_off) = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(table))

    ax_acc.bar(x, table["정확도"], color="#3b82f6")
    ax_acc.axhline(table["기준선"].iloc[0], color="red", linestyle="--",
                   label=f"최빈값 기준선 {table['기준선'].iloc[0]:.3f}")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(table.index, rotation=15)
    ax_acc.set_ylim(0.4, 0.75)
    ax_acc.set_ylabel("홀드아웃 정확도")
    ax_acc.set_title("전체 정확도 — OFFSPEED를 다 틀려도 별로 안 떨어진다")
    for i, value in enumerate(table["정확도"]):
        ax_acc.text(i, value + 0.005, f"{value:.3f}", ha="center", fontsize=9)
    ax_acc.legend()

    width = 0.27
    for offset, metric, color in [
        (-width, "OFFSPEED_precision", "#94a3b8"),
        (0.0, "OFFSPEED_recall", "#f59e0b"),
        (width, "OFFSPEED_f1", "#ef4444"),
    ]:
        ax_off.bar(x + offset, table[metric], width, label=metric.split("_")[1], color=color)
    ax_off.set_xticks(x)
    ax_off.set_xticklabels(table.index, rotation=15)
    ax_off.set_ylim(0, 0.6)
    ax_off.set_ylabel("점수")
    ax_off.set_title("OFFSPEED — 이번 실험의 판정 대상")
    ax_off.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=0,
                        help="순열검정 횟수. 0이면 생략 (1000회는 수 분 걸린다)")
    args = parser.parse_args()

    train_df, holdout_df = load_split()
    print(f"학습 {len(train_df)}개 / 홀드아웃 {len(holdout_df)}개 (game_pk={HOLDOUT_GAME_PK})")
    print(holdout_df["group"].value_counts().to_string())
    print()

    results = {}
    importance_by_set = {}
    for name, columns in FEATURE_SETS.items():
        result, importances = evaluate(columns, train_df, holdout_df)
        results[name] = result
        importance_by_set[name] = importances
        print(f"[{name}] 정확도 {result['정확도']:.3f} "
              f"(기준선 대비 {result['기준선대비']:+.3f})  중요도비 {result['중요도비']:.2f}")
        print(f"         OFFSPEED  p={result['OFFSPEED_precision']:.2f} "
              f"r={result['OFFSPEED_recall']:.2f} f1={result['OFFSPEED_f1']:.2f}")

    table = pd.DataFrame(results).T
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "feature_ablation.csv")
    table.to_csv(csv_path)
    png_path = os.path.join(OUT_DIR, "feature_ablation.png")
    plot_comparison(table, png_path)
    print(f"\n저장: {csv_path}\n저장: {png_path}")

    print("\n특징 중요도 (전체 집합):")
    print(importance_by_set["전체(14)"].sort_values(ascending=False).to_string())

    if args.permutations:
        best = table["OFFSPEED_f1"].idxmax()
        print(f"\n순열검정 {args.permutations}회 — 대상: {best}", flush=True)
        scores, p_value = permutation_test(
            FEATURE_SETS[best], train_df, holdout_df,
            float(table.loc[best, "정확도"]), args.permutations,
        )
        print(f"실제 정확도 : {table.loc[best, '정확도']:.3f}")
        print(f"셔플 평균   : {scores.mean():.3f}  (95%tile {np.percentile(scores, 95):.3f})")
        print(f"p-value     : {p_value:.4f}")
        np.save(os.path.join(OUT_DIR, "feature_ablation_null_scores.npy"), scores)


if __name__ == "__main__":
    main()
