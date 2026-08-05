"""투구 궤적(YOLO 픽셀 좌표 시퀀스) → 구종 그룹 분류용 특징 추출."""
import math
from typing import Callable

import cv2
import numpy as np

FEATURE_COLUMNS = [
    "duration_frames",
    "path_length_px",
    "straight_line_px",
    "curvature_ratio",
    "vertical_drop_px",
    "horizontal_deviation_px",
    "apparent_speed_px_per_frame",
]


def _max_perpendicular_deviation(trajectory: list[tuple[float, float]]) -> float:
    """궤적의 각 점이 시작점-끝점 직선에서 얼마나 벗어나는지의 최댓값."""
    x0, y0 = trajectory[0]
    x1, y1 = trajectory[-1]
    line_len = math.hypot(x1 - x0, y1 - y0)
    if line_len == 0:
        return 0.0

    max_dev = 0.0
    for px, py in trajectory:
        dev = abs((y1 - y0) * px - (x1 - x0) * py + x1 * y0 - y1 * x0) / line_len
        max_dev = max(max_dev, dev)
    return max_dev


def compute_trajectory_features(
    trajectory: list[tuple[float, float]], min_points: int = 3
) -> dict | None:
    """
    궤적(픽셀 좌표 시퀀스)에서 구종 그룹 분류용 특징을 계산한다.
    포인트 수가 min_points 미만이면 None (궤적 감지 실패로 간주, 해당 샘플 제외).
    """
    if len(trajectory) < min_points:
        return None

    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]

    straight_line_px = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    path_length_px = sum(
        math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        for i in range(1, len(trajectory))
    )
    curvature_ratio = path_length_px / straight_line_px if straight_line_px > 0 else 1.0
    duration_frames = len(trajectory)
    apparent_speed_px_per_frame = (
        straight_line_px / (duration_frames - 1) if duration_frames > 1 else 0.0
    )

    return {
        "duration_frames": duration_frames,
        "path_length_px": path_length_px,
        "straight_line_px": straight_line_px,
        "curvature_ratio": curvature_ratio,
        "vertical_drop_px": ys[-1] - ys[0],
        "horizontal_deviation_px": _max_perpendicular_deviation(trajectory),
        "apparent_speed_px_per_frame": apparent_speed_px_per_frame,
    }


def frames_in_window(
    video_path: str,
    timestamp_sec: float,
    lookback_start_sec: float = 3.0,
    lookback_end_sec: float = 0.3,
    max_frames: int = 90,
) -> list[np.ndarray]:
    """
    timestamp_sec 기준 (timestamp_sec - lookback_start_sec) ~ (timestamp_sec - lookback_end_sec)
    구간의 프레임을 모두 추출한다. 영상을 열 수 없으면 빈 리스트.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = max(0, int((timestamp_sec - lookback_start_sec) * fps))
    end_frame = max(start_frame, int((timestamp_sec - lookback_end_sec) * fps))
    min_required_frames = end_frame - start_frame + 1
    effective_max_frames = max(max_frames, min_required_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[np.ndarray] = []
    frame_idx = start_frame
    while frame_idx <= end_frame and len(frames) < effective_max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames


def trajectory_from_frames(
    frames: list[np.ndarray],
    detect_fn: Callable[[np.ndarray], list[dict]],
) -> list[tuple[float, float]]:
    """
    프레임 시퀀스에서 프레임별 최고-신뢰도 공 감지 결과로 궤적을 구성한다.
    detect_fn은 yolo_detector.detect_ball_in_frame과 동일한 반환 형식
    ([{"bbox", "conf", "cx", "cy"}, ...])을 따라야 한다. 감지 없는 프레임은 건너뛴다.
    """
    trajectory: list[tuple[float, float]] = []
    for frame in frames:
        detections = detect_fn(frame)
        if not detections:
            continue
        best = max(detections, key=lambda d: d["conf"])
        trajectory.append((float(best["cx"]), float(best["cy"])))
    return trajectory


def extract_trajectory_window(
    video_path: str,
    timestamp_sec: float,
    model,
    lookback_start_sec: float = 3.0,
    lookback_end_sec: float = 0.3,
) -> list[tuple[float, float]]:
    """
    OCR 타임스탬프 구간의 공 궤적을 추출한다.
    model: yolo_detector.load_model()로 로드한 YOLO 모델.
    """
    from yolo_detector import detect_ball_in_frame  # 지연 import: 순수 함수는 ultralytics에 비의존

    frames = frames_in_window(video_path, timestamp_sec, lookback_start_sec, lookback_end_sec)
    return trajectory_from_frames(frames, lambda f: detect_ball_in_frame(model, f))
