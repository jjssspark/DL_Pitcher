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


def longest_moving_chain(
    candidates_by_frame: list[tuple[int, list[tuple[float, float, float]]]],
    max_jump_px: float,
    min_total_move_px: float,
    max_gap_frames: int = 2,
    min_step_px: float = 3.0,
) -> list[tuple[float, float]]:
    """
    프레임별 감지 후보들 중 **실제로 이동하는** 최장 사슬을 고른다.

    `longest_smooth_run`으로는 부족하다는 것이 실측으로 드러났다(TS-017). 파인튜닝한
    감지기는 화면의 고정된 지점을 30프레임 넘게, 실제 공보다 높은 신뢰도(0.71 vs 0.45)로
    잡는다. 정지 물체는 "연속 프레임 간 이동이 작다"는 조건을 완벽하게 만족하므로
    최장 매끄러운 구간 규칙에서 항상 이긴다. TS-013에서 글러브가 이긴 것과 같은 구조다.

    그래서 두 조건을 함께 건다.
      - 연속성: 프레임 간 이동이 max_jump_px 이하 (순간이동 = 다른 물체)
      - 운동성: 사슬 전체의 시작-끝 변위가 min_total_move_px 이상 (정지 = 공이 아님)

    프레임 단위 최고 신뢰도를 쓰지 않고 후보 전체를 받는 이유는, 정지 오탐과 공이
    같은 프레임에 동시에 잡힐 때 신뢰도로 고르면 오탐이 선택되기 때문이다.

    max_gap_frames는 감지가 한두 프레임 빠져도 궤적을 잇는다 — 실측에서 흔하다.
    다만 그 관용 때문에 정지 오탐이 큰 점프로 공에 이어붙는 일이 생기고(실측 투구 #135:
    정지 3프레임 → 110px 점프 → 공 3프레임), 전체 변위가 크므로 운동성 조건도 통과해
    버린다. 그래서 사슬을 고른 뒤 앞뒤의 정지 구간(프레임당 이동 min_step_px 미만)을
    잘라낸다. 미트에 들어간 뒤의 정지 꼬리도 같은 처리로 없어진다 — 남겨두면 곡률이
    평평해져 구종 차이를 지운다.
    """
    nodes = [
        (frame_idx, x, y)
        for frame_idx, detections in candidates_by_frame
        for x, y, _conf in detections
    ]
    if not nodes:
        return []

    # 프레임 순 DP: best_len[i] = i에서 끝나는 최장 사슬 길이, prev[i] = 직전 노드
    best_len = [1] * len(nodes)
    prev = [-1] * len(nodes)
    for i, (frame_i, xi, yi) in enumerate(nodes):
        for j, (frame_j, xj, yj) in enumerate(nodes[:i]):
            gap = frame_i - frame_j
            if not 1 <= gap <= max_gap_frames:
                continue
            # 간격이 벌어진 만큼 허용 이동량도 비례해서 늘린다.
            if math.dist((xj, yj), (xi, yi)) > max_jump_px * gap:
                continue
            if best_len[j] + 1 > best_len[i]:
                best_len[i] = best_len[j] + 1
                prev[i] = j

    # 운동성 조건을 만족하는 사슬 중 가장 긴 것. 정지 사슬은 아무리 길어도 탈락한다.
    best_chain: list[tuple[float, float]] = []
    for end in range(len(nodes)):
        chain_idx = []
        cursor = end
        while cursor != -1:
            chain_idx.append(cursor)
            cursor = prev[cursor]
        chain_idx.reverse()

        points = _trim_static_ends([(nodes[k][1], nodes[k][2]) for k in chain_idx], min_step_px)
        if len(points) < 2 or math.dist(points[0], points[-1]) < min_total_move_px:
            continue
        if len(points) > len(best_chain):
            best_chain = points

    return best_chain


def _trim_static_ends(
    points: list[tuple[float, float]], min_step_px: float
) -> list[tuple[float, float]]:
    """
    앞뒤에서 프레임당 이동이 min_step_px 미만인 구간을 잘라낸다.

    정지 구간의 **마지막 점도 함께 버린다.** 그 점은 정지 물체의 좌표이고, 거기서
    다음 점으로의 "큰 이동"은 공의 운동이 아니라 다른 물체로 건너뛴 것이기 때문이다.
    """
    start, end = 0, len(points) - 1
    while start < end and math.dist(points[start], points[start + 1]) < min_step_px:
        start += 1
    if start > 0:
        start += 1

    while end > start and math.dist(points[end - 1], points[end]) < min_step_px:
        end -= 1
    if end < len(points) - 1:
        end -= 1

    return points[start:end + 1] if start <= end else []


def extract_trajectory_candidates(
    video_path: str,
    timestamp_sec: float,
    model,
    lookback_start_sec: float = 3.0,
    lookback_end_sec: float = 0.3,
    target_fps: float | None = None,
    imgsz: int = 640,
) -> list[tuple[int, list[tuple[float, float, float]]]]:
    """
    윈도우 안 프레임별로 **모든** 감지 후보를 (frame_idx, [(x, y, conf), ...])로 돌려준다.

    `extract_trajectory_points`는 프레임마다 최고 신뢰도 1건만 남기는데, 정지 오탐이
    실제 공보다 신뢰도가 높은 경우가 있어 그 시점에 공이 통째로 버려진다(TS-017).
    후보를 전부 남겨야 longest_moving_chain이 운동성으로 고를 수 있다.
    """
    from yolo_detector import detect_ball_in_frame  # 지연 import: 순수 함수는 ultralytics에 비의존

    frames = frames_in_window(
        video_path, timestamp_sec, lookback_start_sec, lookback_end_sec,
        target_fps=target_fps,
    )
    result = []
    for frame_idx, frame in enumerate(frames):
        detections = detect_ball_in_frame(model, frame, imgsz=imgsz)
        if not detections:
            continue
        result.append((
            frame_idx,
            [(float(d["cx"]), float(d["cy"]), float(d["conf"])) for d in detections],
        ))
    return result


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
