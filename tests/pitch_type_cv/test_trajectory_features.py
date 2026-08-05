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


import cv2
import numpy as np


@pytest.fixture
def synthetic_video(tmp_path):
    """5초, 30fps, 32x32 합성 비디오 파일 생성."""
    path = str(tmp_path / "synthetic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (32, 32))
    for _ in range(150):  # 5초 분량
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    return path


def test_frames_in_window_returns_expected_frame_count(synthetic_video):
    from pitch_type_cv.trajectory_features import frames_in_window

    frames = frames_in_window(
        synthetic_video, timestamp_sec=3.0, lookback_start_sec=2.0, lookback_end_sec=0.5
    )

    # start_frame=int((3.0-2.0)*30)=30, end_frame=int((3.0-0.5)*30)=75 → 46개(30~75 포함)
    assert len(frames) == 46
    assert frames[0].shape == (32, 32, 3)


@pytest.fixture
def synthetic_video_60fps(tmp_path):
    """60fps, 32x32 합성 비디오 파일 생성 (약 200프레임, ~3.3초 분량)."""
    path = str(tmp_path / "synthetic_60fps.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 60.0, (32, 32))
    for _ in range(200):
        writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    writer.release()
    return path


def test_frames_in_window_does_not_truncate_on_high_fps_video(synthetic_video_60fps):
    from pitch_type_cv.trajectory_features import frames_in_window

    # timestamp_sec=3.0, 기본 lookback(3.0~0.3초 전) → start_frame=0, end_frame=162 → 163프레임 필요
    frames = frames_in_window(
        synthetic_video_60fps, timestamp_sec=3.0,
        lookback_start_sec=3.0, lookback_end_sec=0.3,
    )

    # 하드코딩된 max_frames=90 캡에 걸리면 90개에서 잘린다 — 그보다 많이 반환돼야 함
    assert len(frames) >= 160
    assert len(frames) > 90


def test_sampling_step_rounds_ntsc_framerate_to_nearest_integer():
    from pitch_type_cv.trajectory_features import _sampling_step

    # NTSC 59.94fps를 30fps로 → 59.94/30 = 1.998. 내림하면 1이 되어 샘플링이 통째로
    # 무효화되므로 반올림해야 한다 (실제 720p 다운로드가 정확히 이 케이스)
    assert _sampling_step(59.94, 30.0) == 2
    assert _sampling_step(29.97, 30.0) == 1
    assert _sampling_step(30.0, 60.0) == 1   # 업샘플링은 불가
    assert _sampling_step(59.94, None) == 1  # 샘플링 비활성


def test_frames_in_window_subsamples_to_target_fps(synthetic_video_60fps):
    from pitch_type_cv.trajectory_features import frames_in_window

    kwargs = dict(
        timestamp_sec=3.0, lookback_start_sec=3.0, lookback_end_sec=0.3,
    )
    full = frames_in_window(synthetic_video_60fps, **kwargs)
    halved = frames_in_window(synthetic_video_60fps, target_fps=30.0, **kwargs)

    # 60fps 영상을 30fps로 샘플링 → 매 2번째 프레임만 (163 → 82)
    assert len(halved) == math.ceil(len(full) / 2)


def test_frames_in_window_does_not_upsample_when_target_fps_exceeds_video_fps(
    synthetic_video,
):
    from pitch_type_cv.trajectory_features import frames_in_window

    kwargs = dict(
        timestamp_sec=3.0, lookback_start_sec=2.0, lookback_end_sec=0.5,
    )
    baseline = frames_in_window(synthetic_video, **kwargs)
    requested = frames_in_window(synthetic_video, target_fps=60.0, **kwargs)

    # 30fps 영상에 60fps를 요구해도 프레임을 늘릴 수는 없다 — 그대로여야 한다
    assert len(requested) == len(baseline)


def test_frames_in_window_returns_empty_list_for_invalid_path():
    from pitch_type_cv.trajectory_features import frames_in_window

    frames = frames_in_window("not_a_real_file.mp4", timestamp_sec=1.0)
    assert frames == []


def test_trajectory_from_frames_uses_highest_confidence_detection():
    from pitch_type_cv.trajectory_features import trajectory_from_frames

    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]

    def fake_detect(frame):
        return [
            {"bbox": [0, 0, 2, 2], "conf": 0.4, "cx": 1, "cy": 1},
            {"bbox": [5, 5, 7, 7], "conf": 0.9, "cx": 6, "cy": 6},
        ]

    trajectory = trajectory_from_frames(frames, fake_detect)

    assert trajectory == [(6.0, 6.0), (6.0, 6.0), (6.0, 6.0)]


def test_trajectory_from_frames_skips_frames_with_no_detection():
    from pitch_type_cv.trajectory_features import trajectory_from_frames

    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    call_count = {"n": 0}

    def fake_detect(frame):
        call_count["n"] += 1
        if call_count["n"] == 2:
            return []
        return [{"bbox": [0, 0, 2, 2], "conf": 0.5, "cx": 1, "cy": 1}]

    trajectory = trajectory_from_frames(frames, fake_detect)

    assert trajectory == [(1.0, 1.0), (1.0, 1.0)]
