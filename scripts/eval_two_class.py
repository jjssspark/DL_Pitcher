"""
FASTBALL vs BREAKING 2분류 — 3분류가 OFFSPEED에서 막혔을 때의 대비 경로.

3분류를 고집할 이유가 없다는 것이 실측으로 나왔다. 체인지업·스플리터를 패스트볼과
가르는 것은 속도인데 픽셀 좌표로는 절대 속도를 복원할 수 없고, 좌표 기반 특징을 7개
더 넣어도 OFFSPEED의 one-vs-rest AUC가 0.641 -> 0.648로 제자리였다 (TS-023).

반면 FASTBALL과 BREAKING은 궤적 모양만으로 갈린다 (AUC 0.851 / 0.819). 그 둘만 남기면
"중계 영상만으로 되는 것과 안 되는 것"의 경계를 명시하는 결론이 된다 — OFFSPEED를
억지로 끼워 넣어 셋 다 흐리게 만드는 것보다 정직하다.

실행:
  venv/bin/python3 scripts/eval_two_class.py
"""
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
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from pitch_type_cv.group_classifier import train_classifier  # noqa: E402
from pitch_type_cv.trajectory_features import FEATURE_COLUMNS  # noqa: E402

OUT_DIR = os.path.join(ROOT, "output", "pitch_type_cv")
DATASET_PATH = os.path.join(OUT_DIR, "dataset_clips.csv")

TWO_CLASS = ["FASTBALL", "BREAKING"]
POSITIVE = "FASTBALL"
HOLDOUT_GAME_PK = 813027


def load_two_class() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATASET_PATH)
    df = df[df["has_trajectory"] & df["group"].isin(TWO_CLASS)].reset_index(drop=True)
    # 박스 컬럼이 없는 v1 데이터셋에서도 돌아가게 한다.
    columns = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(columns)
    if missing:
        print(f"[주의] 데이터셋에 없는 특징 {sorted(missing)} — 나머지 {len(columns)}개로 돈다")
    return df, columns


def leave_one_game_out(df: pd.DataFrame, columns: list[str]) -> dict:
    """
    경기 단위 교차검증. 홀드아웃 1경기짜리 수치는 경기별 중계 조건에 흔들린다.
    각 폴드에서 학습에 안 쓰인 경기를 통째로 평가한다 — 투수도 겹치지 않는다.
    """
    rows, curves = [], []
    for game_pk in sorted(df["game_pk"].unique()):
        train_df = df[df["game_pk"] != game_pk]
        fold_df = df[df["game_pk"] == game_pk]
        model = train_classifier(train_df, train_df["group"].tolist(), feature_columns=columns)

        y_true = fold_df["group"].tolist()
        y_pred = model.predict(fold_df[columns])
        score = model.predict_proba(fold_df[columns])[:, list(model.classes_).index(POSITIVE)]
        y_binary = (np.asarray(y_true) == POSITIVE).astype(int)

        majority = train_df["group"].value_counts().idxmax()
        rows.append({
            "game_pk": game_pk,
            "n": len(fold_df),
            "정확도": accuracy_score(y_true, y_pred),
            "기준선": accuracy_score(y_true, [majority] * len(y_true)),
            "AUC": roc_auc_score(y_binary, score),
        })
        fpr, tpr, _ = roc_curve(y_binary, score)
        curves.append((game_pk, fpr, tpr, rows[-1]["AUC"]))

    return {"folds": pd.DataFrame(rows), "curves": curves}


def plot_results(folds: pd.DataFrame, curves: list, cm: np.ndarray, path: str) -> None:
    plt.rcParams["font.family"] = "AppleGothic"   # 한글 tofu 방지 (TS-004)
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax_fold, ax_roc, ax_cm) = plt.subplots(1, 3, figsize=(16, 4.5))

    x = np.arange(len(folds))
    ax_fold.bar(x - 0.2, folds["정확도"], 0.4, label="정확도", color="#3b82f6")
    ax_fold.bar(x + 0.2, folds["기준선"], 0.4, label="최빈값 기준선", color="#cbd5e1")
    ax_fold.set_xticks(x)
    ax_fold.set_xticklabels([f"{g}\n(n={n})" for g, n in zip(folds["game_pk"], folds["n"])],
                            fontsize=8)
    ax_fold.set_ylim(0, 1)
    ax_fold.set_ylabel("정확도")
    ax_fold.set_title(f"경기별 (LOGO) — 평균 {folds['정확도'].mean():.3f} "
                      f"vs 기준선 {folds['기준선'].mean():.3f}")
    ax_fold.legend(fontsize=8)

    for game_pk, fpr, tpr, auc in curves:
        ax_roc.plot(fpr, tpr, linewidth=1.5, label=f"{game_pk} (AUC {auc:.3f})")
    ax_roc.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1)
    ax_roc.set_xlabel("거짓 양성률")
    ax_roc.set_ylabel("참 양성률")
    ax_roc.set_title(f"ROC — 평균 AUC {folds['AUC'].mean():.3f}")
    ax_roc.legend(fontsize=8)

    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks([0, 1], TWO_CLASS)
    ax_cm.set_yticks([0, 1], TWO_CLASS)
    ax_cm.set_xlabel("예측")
    ax_cm.set_ylabel("실제")
    ax_cm.set_title(f"홀드아웃 혼동행렬 (game_pk={HOLDOUT_GAME_PK})")
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax_cm, fraction=0.046)

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    df, columns = load_two_class()
    counts = df["group"].value_counts()
    print(f"2분류 대상 {len(df)}개 "
          f"(FASTBALL {counts.get('FASTBALL', 0)} / BREAKING {counts.get('BREAKING', 0)})")
    print(f"특징 {len(columns)}개\n")

    result = leave_one_game_out(df, columns)
    folds = result["folds"]
    print("경기 단위 leave-one-game-out")
    print(folds.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n평균 정확도 {folds['정확도'].mean():.3f} "
          f"vs 기준선 {folds['기준선'].mean():.3f} "
          f"({folds['정확도'].mean() - folds['기준선'].mean():+.3f})")
    print(f"평균 AUC   {folds['AUC'].mean():.3f}")

    # 고정 홀드아웃으로 분류 리포트를 따로 낸다 — 3분류 결과와 같은 경기로 비교하기 위해서다.
    train_df = df[df["game_pk"] != HOLDOUT_GAME_PK]
    holdout_df = df[df["game_pk"] == HOLDOUT_GAME_PK]
    model = train_classifier(train_df, train_df["group"].tolist(), feature_columns=columns)
    y_pred = model.predict(holdout_df[columns])
    print(f"\n홀드아웃 game_pk={HOLDOUT_GAME_PK} ({len(holdout_df)}개)")
    print(classification_report(holdout_df["group"], y_pred, zero_division=0))

    cm = confusion_matrix(holdout_df["group"], y_pred, labels=TWO_CLASS)
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "two_class_folds.csv")
    png_path = os.path.join(OUT_DIR, "two_class_result.png")
    folds.to_csv(csv_path, index=False)
    plot_results(folds, result["curves"], cm, png_path)
    print(f"저장: {csv_path}\n저장: {png_path}")


if __name__ == "__main__":
    main()
