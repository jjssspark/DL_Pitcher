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


def longest_smooth_run(
    trajectory: list[tuple[float, float]], max_jump_px: float
) -> list[tuple[float, float]]:
    """
    연속 프레임 간 이동거리가 max_jump_px 이하인 최장 구간만 남긴다.

    COCO 사전학습 YOLO의 'sports ball'은 중계 화면에서 공 대신 다른 둥근 물체를
    자주 잡는다. 그 결과 궤적이 프레임마다 화면을 가로지르는 랜덤워크가 되고,
    곡률비가 실제 투구(1~1.5)의 수십 배로 튄다. 실제 공은 프레임 사이를 순간이동하지
    않으므로, 공간적으로 이어지는 최장 구간이 실제 비행 구간일 가능성이 가장 높다.
    """
    if len(trajectory) < 2:
        return list(trajectory)

    best_start = best_end = 0      # [best_start, best_end) 반열린 구간
    run_start = 0
    for i in range(1, len(trajectory) + 1):
        broken = (
            i == len(trajectory)
            or math.dist(trajectory[i - 1], trajectory[i]) > max_jump_px
        )
        if not broken:
            continue
        if i - run_start > best_end - best_start:
            best_start, best_end = run_start, i
        run_start = i

    return trajectory[best_start:best_end]


def _sampling_step(fps: float, target_fps: float | None) -> int:
    """
    target_fps에 가장 가까운 정수 프레임 간격. NTSC 계열 영상은 59.94/29.97처럼 정수
    배수에서 미세하게 어긋나므로 내림이 아니라 반올림해야 한다 — 59.94/30 = 1.998을
    내림하면 1이 되어 샘플링이 통째로 무효화된다. 1 미만(업샘플링 요구)은 1로 묶는다.
    """
    if not target_fps:
        return 1
    return max(1, round(fps / target_fps))


def frames_in_window(
    video_path: str,
    timestamp_sec: float,
    lookback_start_sec: float = 3.0,
    lookback_end_sec: float = 0.3,
    max_frames: int = 90,
    target_fps: float | None = None,
) -> list[np.ndarray]:
    """
    timestamp_sec 기준 (timestamp_sec - lookback_start_sec) ~ (timestamp_sec - lookback_end_sec)
    구간의 프레임을 모두 추출한다. 영상을 열 수 없으면 빈 리스트.

    target_fps를 주면 그 프레임레이트에 맞춰 균등 샘플링한다 (60fps 영상 + target 30fps
    → 매 2번째 프레임). 같은 시간 구간을 소스 fps와 무관하게 일정한 프레임 수로 처리해,
    고프레임레이트 영상에서 감지 비용이 배로 늘어나는 것을 막는다. 프레임 수를 늘리지는
    않으므로 영상 fps보다 높은 값을 줘도 원본 그대로 반환한다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = _sampling_step(fps, target_fps)
    start_frame = max(0, int((timestamp_sec - lookback_start_sec) * fps))
    end_frame = max(start_frame, int((timestamp_sec - lookback_end_sec) * fps))
    min_required_frames = (end_frame - start_frame) // step + 1
    effective_max_frames = max(max_frames, min_required_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[np.ndarray] = []
    frame_idx = start_frame
    while frame_idx <= end_frame and len(frames) < effective_max_frames:
        # 건너뛸 프레임은 grab()만 호출해 BGR 변환·복사 비용을 생략한다
        if not cap.grab():
            break
        if (frame_idx - start_frame) % step == 0:
            ret, frame = cap.retrieve()
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


def extract_trajectory_points(
    video_path: str,
    timestamp_sec: float,
    model,
    lookback_start_sec: float = 3.0,
    lookback_end_sec: float = 0.3,
    target_fps: float | None = None,
) -> list[tuple[float, float, float]]:
    """
    OCR 타임스탬프 구간의 원시 공 감지 결과를 (x, y, conf)로 돌려준다.

    extract_trajectory_window와 달리 신뢰도를 버리지 않는다. 이 결과를 캐시해두면
    신뢰도·연속성 임계값을 바꿔가며 재실험할 때 영상을 다시 디코딩·추론하지 않아도
    된다 — 경기당 실측 39분이 걸리는 작업이다.
    """
    from yolo_detector import detect_ball_in_frame  # 지연 import: 순수 함수는 ultralytics에 비의존

    frames = frames_in_window(
        video_path, timestamp_sec, lookback_start_sec, lookback_end_sec,
        target_fps=target_fps,
    )
    points: list[tuple[float, float, float]] = []
    for frame in frames:
        detections = detect_ball_in_frame(model, frame)
        if not detections:
            continue
        best = max(detections, key=lambda d: d["conf"])
        points.append((float(best["cx"]), float(best["cy"]), float(best["conf"])))
    return points


def extract_trajectory_window(
    video_path: str,
    timestamp_sec: float,
    model,
    lookback_start_sec: float = 3.0,
    lookback_end_sec: float = 0.3,
    target_fps: float | None = None,
) -> list[tuple[float, float]]:
    """
    OCR 타임스탬프 구간의 공 궤적을 추출한다.
    model: yolo_detector.load_model()로 로드한 YOLO 모델.
    target_fps: 감지 대상 프레임 샘플링 레이트 (frames_in_window 참고).
    """
    from yolo_detector import detect_ball_in_frame  # 지연 import: 순수 함수는 ultralytics에 비의존

    frames = frames_in_window(
        video_path, timestamp_sec, lookback_start_sec, lookback_end_sec,
        target_fps=target_fps,
    )
    return trajectory_from_frames(frames, lambda f: detect_ball_in_frame(model, f))
