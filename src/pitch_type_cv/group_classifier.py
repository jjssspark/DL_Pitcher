"""궤적 특징 → 구종 그룹(FASTBALL/BREAKING/OFFSPEED) 분류기 학습·추론."""
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from pitch_type_cv.trajectory_features import FEATURE_COLUMNS


def train_classifier(
    X: pd.DataFrame, y: list[str], random_state: int = 42
) -> GradientBoostingClassifier:
    """FEATURE_COLUMNS 특징으로 3그룹 분류기를 학습한다."""
    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X[FEATURE_COLUMNS], y)
    return model


def predict_group(model: GradientBoostingClassifier, features: dict) -> tuple[str, dict[str, float]]:
    """단일 궤적 특징 벡터로 구종 그룹과 그룹별 확률을 예측한다."""
    X = pd.DataFrame([features])[FEATURE_COLUMNS]
    predicted = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    probabilities = dict(zip(model.classes_, proba.tolist()))
    return predicted, probabilities


def save_classifier(model: GradientBoostingClassifier, path: str) -> None:
    joblib.dump(model, path)


def load_classifier(path: str) -> GradientBoostingClassifier:
    return joblib.load(path)
