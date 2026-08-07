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
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

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

# 박스 계열: 공의 겉보기 크기 변화. v1 캐시로 만든 데이터셋에서는 두 값이 0이라
# 이 집합이 '좌표전체'와 같아진다 — 판정 전에 캐시가 v2인지 확인할 것.
BOX_COLUMNS = ["box_growth_per_frame", "release_box_size"]

COORD_ALL = GEOMETRY_ONLY_COLUMNS + TIME_COLUMNS + GEOMETRY_EXTRA_COLUMNS

FEATURE_SETS = {
    "기준(7)": GEOMETRY_ONLY_COLUMNS,
    "+시간(10)": GEOMETRY_ONLY_COLUMNS + TIME_COLUMNS,
    "+기하(11)": GEOMETRY_ONLY_COLUMNS + GEOMETRY_EXTRA_COLUMNS,
    "좌표전체(14)": COORD_ALL,
    "+박스(16)": COORD_ALL + BOX_COLUMNS,
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
    # 임계값과 무관한 판별력. 정확도·f1은 클래스 가중치로도 움직여서 '정보가 늘었나'와
    # '동작점을 옮겼나'가 섞인다 — 소수 클래스 판정은 이 값으로 한다 (TS-023).
    proba = model.predict_proba(holdout_df[columns])
    auc = {
        label: roc_auc_score(
            (np.asarray(y_true) == label).astype(int),
            proba[:, list(model.classes_).index(label)],
        )
        for label in LABELS
    }
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
        result[f"{label}_auc"] = auc[label]
    return result, importances


def evaluate_logo(columns: list[str], df: pd.DataFrame) -> dict:
    """
    경기 단위 leave-one-game-out. 홀드아웃 1경기짜리 수치는 흔들린다 — 실측에서
    같은 특징 집합의 OFFSPEED f1이 단일 홀드아웃 0.10, LOGO 4폴드 0.233이었다.
    """
    accuracies, baselines, offspeed_f1, offspeed_auc = [], [], [], []
    for game_pk in sorted(df["game_pk"].unique()):
        train_df = df[df["game_pk"] != game_pk]
        fold_df = df[df["game_pk"] == game_pk]
        model = train_classifier(
            train_df, train_df["group"].tolist(), feature_columns=columns
        )
        y_true = fold_df["group"].tolist()
        accuracies.append(accuracy_score(y_true, model.predict(fold_df[columns])))
        baselines.append(accuracy_score(
            y_true, [train_df["group"].value_counts().idxmax()] * len(y_true)
        ))
        _p, _r, f1, _s = precision_recall_fscore_support(
            y_true, model.predict(fold_df[columns]), labels=["OFFSPEED"], zero_division=0
        )
        offspeed_f1.append(f1[0])
        proba = model.predict_proba(fold_df[columns])
        offspeed_auc.append(roc_auc_score(
            (np.asarray(y_true) == "OFFSPEED").astype(int),
            proba[:, list(model.classes_).index("OFFSPEED")],
        ))
    return {
        "LOGO_정확도": float(np.mean(accuracies)),
        "LOGO_기준선": float(np.mean(baselines)),
        "LOGO_기준선대비": float(np.mean(accuracies) - np.mean(baselines)),
        "LOGO_OFFSPEED_f1": float(np.mean(offspeed_f1)),
        "LOGO_OFFSPEED_auc": float(np.mean(offspeed_auc)),
        "LOGO_폴드별": " / ".join(f"{a:.2f}" for a in accuracies),
    }


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

    all_df = pd.concat([train_df, holdout_df], ignore_index=True)
    results = {}
    importance_by_set = {}
    for name, columns in FEATURE_SETS.items():
        missing = [c for c in columns if c not in train_df.columns]
        if missing:
            # v1 캐시로 만든 데이터셋에는 박스 컬럼이 없다. 죽지 말고 건너뛴다 —
            # 나머지 집합의 비교는 그대로 유효하다.
            print(f"[{name}] 건너뜀 — 데이터셋에 없는 컬럼: {missing}")
            continue
        result, importances = evaluate(columns, train_df, holdout_df)
        result.update(evaluate_logo(columns, all_df))
        results[name] = result
        importance_by_set[name] = importances
        print(f"[{name}] 홀드아웃 {result['정확도']:.3f} "
              f"({result['기준선대비']:+.3f})  LOGO {result['LOGO_정확도']:.3f} "
              f"({result['LOGO_기준선대비']:+.3f}, 폴드별 {result['LOGO_폴드별']})")
        print(f"         중요도비 {result['중요도비']:.2f}   "
              f"OFFSPEED p={result['OFFSPEED_precision']:.2f} "
              f"r={result['OFFSPEED_recall']:.2f} f1={result['OFFSPEED_f1']:.2f}")
        print(f"         AUC  FASTBALL {result['FASTBALL_auc']:.3f} / "
              f"BREAKING {result['BREAKING_auc']:.3f} / "
              f"OFFSPEED {result['OFFSPEED_auc']:.3f}  "
              f"(LOGO OFFSPEED {result['LOGO_OFFSPEED_auc']:.3f})", flush=True)

    table = pd.DataFrame(results).T
    # .T가 문자열 컬럼(LOGO_폴드별) 때문에 전체를 object로 만든다 — 수치 컬럼만 되돌린다.
    numeric = table.columns.drop("LOGO_폴드별")
    table[numeric] = table[numeric].astype(float)

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "feature_ablation.csv")
    table.to_csv(csv_path)
    png_path = os.path.join(OUT_DIR, "feature_ablation.png")
    plot_comparison(table, png_path)
    print(f"\n저장: {csv_path}\n저장: {png_path}")

    last_name = list(importance_by_set)[-1]   # 건너뛴 집합은 여기 없다
    print(f"\n특징 중요도 ({last_name}):")
    print(importance_by_set[last_name].sort_values(ascending=False).to_string())

    # 판정. 소수 클래스는 정확도가 아니라 AUC로 본다 (TS-023).
    base_auc = table.loc["기준(7)", "LOGO_OFFSPEED_auc"]
    best_auc = table["LOGO_OFFSPEED_auc"].max()
    best_name = table["LOGO_OFFSPEED_auc"].idxmax()
    print(f"\n[판정] OFFSPEED LOGO AUC  기준(7) {base_auc:.3f} -> "
          f"최고 {best_auc:.3f} ({best_name}), 차 {best_auc - base_auc:+.3f}")
    if best_auc - base_auc < 0.03:
        print("       0.03 미만 = 특징이 OFFSPEED에 정보를 넣지 못했다. "
              "정확도가 올랐다면 다른 클래스에서 온 것이다.")

    if args.permutations:
        # 순열검정은 "채택할 모델의 성능이 우연인가"를 묻는다. OFFSPEED f1이 가장 높은
        # 집합이 아니라 실제로 쓸 집합(정확도 최고)을 대상으로 해야 답이 맞는다.
        best = table["정확도"].idxmax()
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
