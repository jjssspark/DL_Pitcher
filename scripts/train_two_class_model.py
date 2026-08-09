"""
FASTBALL vs BREAKING 2분류 모델을 학습해 저장한다.

`scripts/eval_two_class.py`는 폴드마다 학습하고 버린다 — 성능을 재는 스크립트라
그게 맞다. 앱에 붙이려면 전체 데이터로 한 번 학습한 모델이 파일로 있어야 한다.

성능 수치는 여기서 내지 않는다. 학습셋으로 잰 정확도는 홀드아웃이 아니므로
의미가 없고, 진짜 수치는 eval_two_class.py의 LOGO 0.783 / AUC 0.881이다.

실행:
  venv/bin/python3 scripts/train_two_class_model.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd  # noqa: E402

from pitch_type_cv.group_classifier import (  # noqa: E402
    load_classifier,
    save_classifier,
    train_classifier,
)
from pitch_type_cv.live_classifier import TWO_CLASS_MODEL_PATH, TWO_CLASSES  # noqa: E402
from pitch_type_cv.trajectory_features import FEATURE_COLUMNS  # noqa: E402

DATASET_PATH = os.path.join(ROOT, "output", "pitch_type_cv", "dataset_clips.csv")


def main() -> None:
    df = pd.read_csv(DATASET_PATH)
    df = df[df["has_trajectory"] & df["group"].isin(TWO_CLASSES)].reset_index(drop=True)

    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise SystemExit(f"데이터셋에 특징 {sorted(missing)}이 없다. "
                         f"build_pitch_group_clips_dataset.py를 먼저 돌려라.")

    counts = df["group"].value_counts()
    print(f"학습 {len(df)}개 "
          f"(FASTBALL {counts.get('FASTBALL', 0)} / BREAKING {counts.get('BREAKING', 0)})")
    print(f"특징 {len(FEATURE_COLUMNS)}개, 경기 {sorted(df['game_pk'].unique())}")

    model = train_classifier(df, df["group"].tolist())
    os.makedirs(os.path.dirname(TWO_CLASS_MODEL_PATH), exist_ok=True)
    save_classifier(model, TWO_CLASS_MODEL_PATH)
    print(f"\n저장: {TWO_CLASS_MODEL_PATH}")

    # 저장된 파일이 같은 예측을 내는지 확인한다. joblib 왕복에서 깨지면
    # 앱에서만 조용히 다른 값이 나오고, 그건 디버깅하기 가장 나쁜 형태다.
    reloaded = load_classifier(TWO_CLASS_MODEL_PATH)
    before = model.predict(df[FEATURE_COLUMNS])
    after = reloaded.predict(df[FEATURE_COLUMNS])
    if not (before == after).all():
        raise SystemExit("저장 전후 예측이 다르다 — 직렬화가 깨졌다")
    print(f"검증: 재로드 후 {len(df)}개 예측 전부 일치")
    print(f"클래스: {list(reloaded.classes_)}")


if __name__ == "__main__":
    main()
