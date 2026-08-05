import math

import pytest

from pitch_type_cv.trajectory_features import compute_trajectory_features, FEATURE_COLUMNS


def test_returns_none_when_fewer_than_min_points():
    assert compute_trajectory_features([(0.0, 0.0), (1.0, 1.0)]) is None


def test_straight_line_trajectory_has_curvature_ratio_near_one():
    trajectory = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    features = compute_trajectory_features(trajectory)

    assert features["curvature_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert features["horizontal_deviation_px"] == pytest.approx(0.0, abs=1e-6)
    assert features["duration_frames"] == 4


def test_curved_trajectory_has_higher_curvature_and_deviation():
    trajectory = [(0.0, 0.0), (1.0, 3.0), (2.0, 4.0), (3.0, 3.0), (4.0, 0.0)]
    features = compute_trajectory_features(trajectory)

    assert features["straight_line_px"] == pytest.approx(4.0, abs=1e-6)
    assert features["curvature_ratio"] > 1.5
    assert features["horizontal_deviation_px"] == pytest.approx(4.0, abs=1e-6)


def test_vertical_drop_uses_first_and_last_point():
    trajectory = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    features = compute_trajectory_features(trajectory)

    assert features["vertical_drop_px"] == pytest.approx(2.0, abs=1e-6)


def test_apparent_speed_is_straight_line_over_frame_gaps():
    trajectory = [(0.0, 0.0), (4.0, 0.0), (8.0, 0.0)]
    features = compute_trajectory_features(trajectory)

    assert features["apparent_speed_px_per_frame"] == pytest.approx(4.0, abs=1e-6)


def test_feature_columns_match_returned_dict_keys():
    trajectory = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    features = compute_trajectory_features(trajectory)

    assert set(FEATURE_COLUMNS) == set(features.keys())
