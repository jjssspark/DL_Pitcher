"""투구 궤적(YOLO 픽셀 좌표 시퀀스) → 구종 그룹 분류용 특징 추출."""
import math

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
