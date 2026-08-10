"""
척도 불변 특징만으로 2분류를 다시 학습해 앱에서 되는지 본다.

근거: 앱 사슬의 특징을 학습 분포와 비교했더니 규칙성이 하나 나왔다 (TS-025 후속).

  비율 특징      curvature_ratio -0.27σ, speed_ratio_late_early +0.13σ   -> 옮겨감
  절대 픽셀 특징  vertical_accel 4분의 1, apparent_speed 절반,
                horizontal_deviation 3분의 1, path_length 3분의 2      -> 무너짐

그 결과 모델이 앱에서 39개 중 37개를 FASTBALL로 찍고 BREAKING을 하나도 못 맞힌다
(AUC 0.460). 휘어짐을 재는 vertical_accel_px가 눌려서 전부 직구로 보이는 것이다.

그래서 절대 픽셀 크기에 의존하는 특징을 빼거나 현(chord) 길이로 나눠 정규화한다.
새 데이터 수집이 필요 없다 — 기존 1029개로 다시 학습하고, 저장해둔 앱 사슬 39개로
바로 평가한다.

성공 기준 (돌리기 전에 정한다):
  앱 AUC >= 0.65  &  앱 정확도 >= 0.60  &  실험실 LOGO 정확도 >= 0.70

실행:
  venv/bin/python3 scripts/eval_scale_free_features.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score  # noqa: E402

from pitch_type_cv.group_classifier import train_classifier  # noqa: E402
from pitch_type_cv.trajectory_features import FEATURE_COLUMNS  # noqa: E402

OUT_DIR = os.path.join(ROOT, "output", "pitch_type_cv")
TRAIN_PATH = os.path.join(OUT_DIR, "dataset_clips.csv")
APP_PATH = os.path.join(OUT_DIR, "app_chain_features.csv")
TWO_CLASSES = ["FASTBALL", "BREAKING"]
POSITIVE = "FASTBALL"

# 이미 비율이거나 픽셀 크기와 무관한 것들
NATIVE_SCALE_FREE = [
    "curvature_ratio",
    "speed_ratio_late_early",
    "late_drop_ratio",
    "frame_span",
    "duration_frames",
]
# 현(chord=straight_line_px) 길이로 나눠 만든 정규화 특징
DERIVED = [
    "drop_over_chord",
    "dev_over_chord",
    "accel_over_chord",
    "speed_over_chord",
    "box_growth_rel",
]
SCALE_FREE = NATIVE_SCALE_FREE + DERIVED


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """절대 픽셀 특징을 궤적 자체의 크기로 나눠 척도를 없앤다."""
    df = df.copy()
    chord = df["straight_line_px"].replace(0, np.nan)
    box = df["release_box_size"].replace(0, np.nan)
    df["drop_over_chord"] = df["vertical_drop_px"] / chord
    df["dev_over_chord"] = df["horizontal_deviation_px"] / chord
    df["accel_over_chord"] = df["vertical_accel_px"] / chord
    df["speed_over_chord"] = df["apparent_speed_px_per_frame"] / chord
    df["box_growth_rel"] = df["box_growth_per_frame"] / box
    for col in DERIVED:
        df[col] = df[col].fillna(0.0)
    return df


def leave_one_game_out(df: pd.DataFrame, columns: list[str]) -> tuple[float, float, list]:
    accs, aucs, per_game = [], [], []
    for game_pk in sorted(df["game_pk"].unique()):
        tr = df[df["game_pk"] != game_pk]
        te = df[df["game_pk"] == game_pk]
        model = train_classifier(tr, tr["group"].tolist(), feature_columns=columns)
        pred = model.predict(te[columns])
        score = model.predict_proba(te[columns])[:, list(model.classes_).index(POSITIVE)]
        acc = accuracy_score(te["group"], pred)
        auc = roc_auc_score((te["group"] == POSITIVE).astype(int), score)
        accs.append(acc); aucs.append(auc); per_game.append((game_pk, len(te), acc, auc))
    return float(np.mean(accs)), float(np.mean(aucs)), per_game


def evaluate_on_app(model, app: pd.DataFrame, columns: list[str], name: str) -> dict:
    pred = model.predict(app[columns])
    score = model.predict_proba(app[columns])[:, list(model.classes_).index(POSITIVE)]
    y = (app["truth"] == POSITIVE).astype(int)
    acc = accuracy_score(app["truth"], pred)
    auc = roc_auc_score(y, score)
    counts = pd.Series(pred).value_counts().to_dict()
    print(f"\n[{name}] 앱 사슬 {len(app)}개")
    print(f"  정확도 {acc:.3f}   AUC {auc:.3f}   예측 분포 {counts}")
    print(classification_report(app["truth"], pred, zero_division=0, digits=2))
    return {"acc": acc, "auc": auc}


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    train = train[train["has_trajectory"] & train["group"].isin(TWO_CLASSES)].reset_index(drop=True)
    app = pd.read_csv(APP_PATH)
    if app.empty:
        raise SystemExit("앱 사슬 특징이 없다. diagnose_domain_gap.py를 먼저 돌려라.")

    train_d = add_derived(train)
    app_d = add_derived(app)

    print(f"학습 {len(train_d)}개 / 앱 {len(app_d)}개")
    print(f"척도 불변 특징 {len(SCALE_FREE)}개: {', '.join(SCALE_FREE)}\n")

    # 참고: 정규화가 실제로 격차를 줄였는지 확인
    print(f"{'특징':<26}{'학습 평균':>11}{'앱 평균':>11}{'차이(σ)':>10}")
    print("-" * 58)
    for col in SCALE_FREE:
        tm, ts, am = train_d[col].mean(), train_d[col].std(), app_d[col].mean()
        z = (am - tm) / ts if ts else 0.0
        print(f"{col:<26}{tm:>11.3f}{am:>11.3f}{z:>10.2f}{' <<' if abs(z) >= 1 else ''}")

    # 기존 16개 특징 (대조군)
    base_acc, base_auc, _ = leave_one_game_out(train, FEATURE_COLUMNS)
    base_model = train_classifier(train, train["group"].tolist(), feature_columns=FEATURE_COLUMNS)
    print(f"\n[기존 16개] 실험실 LOGO 정확도 {base_acc:.3f}  AUC {base_auc:.3f}")
    base_app = evaluate_on_app(base_model, app, FEATURE_COLUMNS, "기존 16개")

    # 척도 불변 특징
    sf_acc, sf_auc, per_game = leave_one_game_out(train_d, SCALE_FREE)
    sf_model = train_classifier(train_d, train_d["group"].tolist(), feature_columns=SCALE_FREE)
    print(f"\n[척도 불변 {len(SCALE_FREE)}개] 실험실 LOGO 정확도 {sf_acc:.3f}  AUC {sf_auc:.3f}")
    for g, n, a, u in per_game:
        print(f"    {g}  n={n:<4} 정확도 {a:.3f}  AUC {u:.3f}")
    sf_app = evaluate_on_app(sf_model, app_d, SCALE_FREE, f"척도 불변 {len(SCALE_FREE)}개")

    print("\n" + "=" * 58)
    print(f"{'':<14}{'실험실 정확도':>14}{'앱 정확도':>12}{'앱 AUC':>10}")
    print(f"{'기존 16개':<14}{base_acc:>14.3f}{base_app['acc']:>12.3f}{base_app['auc']:>10.3f}")
    print(f"{'척도 불변':<14}{sf_acc:>14.3f}{sf_app['acc']:>12.3f}{sf_app['auc']:>10.3f}")

    ok = sf_app["auc"] >= 0.65 and sf_app["acc"] >= 0.60 and sf_acc >= 0.70
    print("\n[판정] 기준: 앱 AUC>=0.65 & 앱 정확도>=0.60 & 실험실 정확도>=0.70")
    if ok:
        print("  통과 — 척도 정규화가 도메인 격차를 메웠다. 이 특징 집합으로 간다.")
    else:
        missed = []
        if sf_app["auc"] < 0.65: missed.append(f"앱 AUC {sf_app['auc']:.3f}")
        if sf_app["acc"] < 0.60: missed.append(f"앱 정확도 {sf_app['acc']:.3f}")
        if sf_acc < 0.70: missed.append(f"실험실 정확도 {sf_acc:.3f}")
        print(f"  미달 — {', '.join(missed)}")
        print("  -> 척도만의 문제가 아니다. 전체 중계 영상으로 재수집·재학습 (갈래 C)이 남는다.")


if __name__ == "__main__":
    main()
