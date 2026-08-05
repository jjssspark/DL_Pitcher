import numpy as np
import pandas as pd
import pytest

from pitch_type_cv.group_classifier import (
    load_classifier,
    predict_group,
    save_classifier,
    train_classifier,
)
from pitch_type_cv.trajectory_features import FEATURE_COLUMNS


def _synthetic_dataset(seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """3그룹이 뚜렷이 분리되는 합성 특징 데이터를 생성한다."""
    rng = np.random.default_rng(seed)
    centers = {
        "FASTBALL": np.array([90.0, 5.0]),
        "BREAKING": np.array([70.0, 40.0]),
        "OFFSPEED": np.array([75.0, 15.0]),
    }
    rows = []
    labels = []
    for group, center in centers.items():
        for _ in range(30):
            speed, curvature_pct = center + rng.normal(0, 1.5, size=2)
            row = {col: 0.0 for col in FEATURE_COLUMNS}
            row["apparent_speed_px_per_frame"] = speed
            row["curvature_ratio"] = 1.0 + curvature_pct / 100.0
            row["duration_frames"] = 20
            rows.append(row)
            labels.append(group)
    return pd.DataFrame(rows), labels


def test_train_and_predict_recovers_correct_group():
    X, y = _synthetic_dataset()
    model = train_classifier(X, y)

    fastball_features = {col: 0.0 for col in FEATURE_COLUMNS}
    fastball_features["apparent_speed_px_per_frame"] = 90.0
    fastball_features["curvature_ratio"] = 1.05

    predicted_group, probabilities = predict_group(model, fastball_features)

    assert predicted_group == "FASTBALL"
    assert set(probabilities.keys()) == {"FASTBALL", "BREAKING", "OFFSPEED"}
    assert probabilities[predicted_group] == pytest.approx(max(probabilities.values()))
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_save_and_load_classifier_preserves_predictions(tmp_path):
    X, y = _synthetic_dataset()
    model = train_classifier(X, y)
    model_path = str(tmp_path / "group_classifier.pkl")

    save_classifier(model, model_path)
    loaded = load_classifier(model_path)

    features = {col: 0.0 for col in FEATURE_COLUMNS}
    features["apparent_speed_px_per_frame"] = 70.0
    features["curvature_ratio"] = 1.4

    original_pred, _ = predict_group(model, features)
    loaded_pred, _ = predict_group(loaded, features)

    assert original_pred == loaded_pred
