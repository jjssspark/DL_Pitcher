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
    # 아래는 2026-08-07 추가. OFFSPEED(체인지업·스플리터)는 패스트볼과 궤적 '모양'이
    # 비슷하고 속도만 다른데 위 7개는 전부 모양이라, f1이 0.14에 묶여 있었다.
    "frame_span",
    "end_frame",
    "speed_ratio_late_early",
    "release_x",
    "release_y",
    "late_drop_ratio",
    "vertical_accel_px",
    # 좌표 기반 확장으로는 OFFSPEED가 안 갈린다는 것이 실측으로 확인돼(TS-023) 추가한
    # 속도 대리 지표. 박스가 커지는 속도 = 공이 카메라에 가까워지는 속도다.
    # v1 캐시(박스 크기 없음)로 계산하면 둘 다 0이 된다.
    "box_growth_per_frame",
    "release_box_size",
]

# 특징 확장 전의 집합. 절제 실험에서 기준으로 쓴다.
GEOMETRY_ONLY_COLUMNS = FEATURE_COLUMNS[:7]


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
    trajectory: list[tuple[float, float]],
    min_points: int = 3,
    frame_indices: list[int] | None = None,
    box_sizes: list[float] | None = None,
) -> dict | None:
    """
    궤적(픽셀 좌표 시퀀스)에서 구종 그룹 분류용 특징을 계산한다.
    포인트 수가 min_points 미만이면 None (궤적 감지 실패로 간주, 해당 샘플 제외).

    frame_indices는 각 좌표가 나온 프레임 번호다. 주면 시간 계열 특징이 실제 경과
    프레임으로 계산되고, 없으면 연속 프레임으로 가정한다(구 OCR 경로 호출 호환).
    """
    if len(trajectory) < min_points:
        return None

    xs = [p[0] for p in trajectory]
    ys = [p[1] for p in trajectory]
    frames = list(frame_indices) if frame_indices else list(range(len(trajectory)))

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
        # 시간 계열: 같은 구간을 지나는 데 걸린 프레임 수가 곧 속도의 역수다.
        # duration_frames와 다르다 — 저쪽은 감지된 '점의 개수'라 누락 프레임을 못 센다.
        "frame_span": frames[-1] - frames[0],
        # 윈도우가 클립 시작 기준으로 고정돼 있어(2.8~4.2초, ADR-0010) 프레임 번호가
        # 클립 간 비교 가능하다. 느린 공은 미트에 늦게 도달한다.
        "end_frame": frames[-1],
        "speed_ratio_late_early": _late_over_early_speed(xs, ys, frames),
        "release_x": xs[0],
        "release_y": ys[0],
        "late_drop_ratio": _late_drop_ratio(ys),
        "vertical_accel_px": _vertical_acceleration(ys, frames),
        "box_growth_per_frame": _box_growth_per_frame(box_sizes, frames),
        "release_box_size": box_sizes[0] if box_sizes else 0.0,
    }


def _box_growth_per_frame(box_sizes: list[float] | None, frames: list[int]) -> float:
    """
    박스 크기의 프레임당 증가율 (최소자승 기울기).

    끝점 두 개의 차로 재지 않는다 — 감지 박스는 정수 픽셀이고 공이 10~20px라
    1~2px 노이즈가 그대로 기울기가 된다. 전체 점에 직선을 맞추는 편이 안정적이다.
    박스 크기가 없는 v1 캐시에서는 0을 준다.
    """
    if not box_sizes or len(set(frames)) < 2:
        return 0.0
    t = np.asarray(frames, dtype=float) - frames[0]
    slope, _intercept = np.polyfit(t, np.asarray(box_sizes, dtype=float), 1)
    return float(slope)


def box_sizes_for_chain(
    chain: list[tuple[int, float, float]],
    candidates_by_frame: list[tuple[int, list[tuple]]],
) -> list[float]:
    """
    사슬의 각 점에 대응하는 감지 박스 크기를 후보에서 되찾는다.

    사슬은 좌표만 돌려주는데, 한 프레임에 후보가 여럿이면(정지 오탐 + 실제 공)
    사슬이 고른 그 좌표의 박스를 집어야 한다. 그래서 좌표로 대조한다.
    박스 크기가 없는 구 스키마 후보는 0으로 둔다.
    """
    by_frame = dict(candidates_by_frame)
    sizes = []
    for frame_idx, x, y in chain:
        size = 0.0
        for detection in by_frame.get(frame_idx, []):
            if len(detection) >= 5 and detection[0] == x and detection[1] == y:
                size = math.sqrt(float(detection[3]) * float(detection[4]))
                break
        sizes.append(size)
    return sizes


def _late_over_early_speed(
    xs: list[float], ys: list[float], frames: list[int]
) -> float:
    """
    궤적 후반부 / 전반부의 프레임당 속도 비.

    원근 때문에 공이 카메라에 가까워질수록 픽셀 속도가 커진다. 그 증가율은 절대
    속도와 달리 화면 배율에 덜 민감해, 픽셀 좌표만 있어도 남는 몇 안 되는 속도 단서다.
    반드시 프레임당으로 나눈다 — 감지가 빠진 구간의 스텝을 그대로 쓰면 가속으로 오인한다.
    """
    speeds = [
        math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) / max(1, frames[i] - frames[i - 1])
        for i in range(1, len(xs))
    ]
    if not speeds:
        return 1.0

    split = len(speeds) // 2 or 1          # 스텝이 1개뿐이면 전후 구분이 없다 -> 1.0
    early = _mean(speeds[:split])
    late = _mean(speeds[split:])
    if early <= 0 or not late:
        return 1.0
    return late / early


def _late_drop_ratio(ys: list[float]) -> float:
    """
    전체 낙차 중 후반 절반에서 생긴 비율.

    체인지업·스플리터는 낙차가 후반에 몰린다. 전체 낙차(vertical_drop_px)가 같아도
    분포가 다르므로 기존 특징으로는 구분되지 않는다. 낙차가 0이면 0.5(중립)를 준다.
    """
    total = ys[-1] - ys[0]
    if abs(total) < 1e-9:
        return 0.5
    mid = len(ys) // 2
    return (ys[-1] - ys[mid]) / total


def _vertical_acceleration(ys: list[float], frames: list[int]) -> float:
    """
    y를 프레임의 2차식으로 적합했을 때의 d²y/dt². 느린 공일수록 중력에 오래 노출된다.

    프레임 번호가 3개 미만으로 서로 다르면 2차 적합이 불정이므로 0을 준다.
    """
    t = np.asarray(frames, dtype=float) - frames[0]
    if len(set(frames)) < 3:
        return 0.0
    quadratic, _linear, _const = np.polyfit(t, np.asarray(ys, dtype=float), 2)
    return float(2.0 * quadratic)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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
    """좌표만 필요한 호출부용. 실제 계산은 longest_moving_chain_frames가 한다."""
    return [
        (x, y)
        for _frame, x, y in longest_moving_chain_frames(
            candidates_by_frame, max_jump_px, min_total_move_px,
            max_gap_frames=max_gap_frames, min_step_px=min_step_px,
        )
    ]


def longest_moving_chain_frames(
    candidates_by_frame: list[tuple[int, list[tuple[float, float, float]]]],
    max_jump_px: float,
    min_total_move_px: float,
    max_gap_frames: int = 2,
    min_step_px: float = 3.0,
) -> list[tuple[int, float, float]]:
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
    # 감지 후보는 (x, y, conf) 또는 박스 크기가 붙은 (x, y, conf, w, h)로 들어온다.
    # 사슬은 좌표만 쓰므로 앞 둘만 떼어 구·신 캐시 스키마를 모두 받는다.
    nodes = _drop_static_nodes([
        (frame_idx, float(detection[0]), float(detection[1]))
        for frame_idx, detections in candidates_by_frame
        for detection in detections
    ])
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
            step = math.dist((xj, yj), (xi, yi))
            # 간격이 벌어진 만큼 허용 이동량도 비례해서 늘린다.
            if step > max_jump_px * gap:
                continue
            # 움직이지 않는 두 감지 사이에는 간선을 만들지 않는다. 이 조건이 없으면
            # 정지 오탐끼리 긴 사슬을 만들고, gap 관용을 타고 진짜 공에 이어붙어
            # 사슬 한가운데 정지 구간이 박힌다(TS-019, 실측 투구 #53).
            if step < min_step_px:
                continue
            if best_len[j] + 1 > best_len[i]:
                best_len[i] = best_len[j] + 1
                prev[i] = j

    # 운동성 조건을 만족하는 사슬 중 가장 긴 것. 정지 사슬은 아무리 길어도 탈락한다.
    best_chain: list[tuple[int, float, float]] = []
    for end in range(len(nodes)):
        chain_idx = []
        cursor = end
        while cursor != -1:
            chain_idx.append(cursor)
            cursor = prev[cursor]
        chain_idx.reverse()

        points = _trim_chain_ends([nodes[k] for k in chain_idx], min_step_px)
        if len(points) < 2:
            continue
        displacement = math.dist(points[0][1:], points[-1][1:])
        if displacement < min_total_move_px:
            continue
        if len(points) > len(best_chain):
            best_chain = points

    return best_chain


def _trim_chain_ends(
    nodes: list[tuple[int, float, float]], min_step_px: float, outlier_ratio: float = 3.0
) -> list[tuple[int, float, float]]:
    """
    사슬 양끝에서 (1) 정지 구간과 (2) 속도 이상치를 잘라낸다.

    간선 조건으로 정지 물체끼리는 이어지지 않게 됐지만, 정지 감지 **하나**가 큰 점프로
    사슬 끝에 붙는 건 여전히 가능하다(TS-019). 그 점프는 이동량이 커서 정지 트림에
    걸리지 않는다. 대신 주변 대비 명백한 속도 이상치라는 점으로 판별한다 — 실측에서
    프레임당 55px 점프가 10px대 구간 옆에 붙어 있었다.

    비교는 반드시 **프레임당 속도**로 한다. 원 이동거리로 비교하면 감지가 한 프레임
    빠진 구간(gap 2)의 스텝이 2배로 나와, 정상적으로 가속하는 공의 끝부분이 이상치로
    잘려나간다.
    """
    trimmed = _trim_static_ends(nodes, min_step_px)

    for _ in range(2):   # 양끝에 최대 1개씩만 붙을 수 있으므로 두 번이면 충분
        if len(trimmed) < 3:   # 스텝이 2개는 있어야 이상치를 판별할 수 있다
            break
        speeds = _per_frame_speeds(trimmed)
        if speeds[0] > outlier_ratio * _median(speeds[1:]):
            trimmed = trimmed[1:]
            continue
        if speeds[-1] > outlier_ratio * _median(speeds[:-1]):
            trimmed = trimmed[:-1]
            continue
        break

    return trimmed


def _drop_static_nodes(
    nodes: list[tuple[int, float, float]],
    radius_px: float = 3.0,
    min_repeats: int = 3,
) -> list[tuple[int, float, float]]:
    """
    여러 프레임에서 사실상 같은 좌표에 나타나는 감지를 버린다.

    이게 정지 오탐의 정의다. 사슬을 만든 **뒤에** 걸러내려는 시도는 네 번 실패했다 —
    정지 구간이 사슬의 앞, 중간, 뒤, 전체 어디에든 붙을 수 있고 그때마다 다른 처방이
    필요했기 때문이다(TS-013 / TS-014 / TS-017 / TS-019). 사슬을 만들기 전에 후보에서
    빼면 네 경우가 한 번에 사라진다.

    90mph 공은 30fps에서 포구 직전까지도 프레임당 3px 이상 움직인다. 3픽셀 반경 안에
    서로 다른 프레임의 감지가 3개 이상 모여 있으면 공이 아니다.
    """
    kept = []
    for frame_idx, x, y in nodes:
        overlaps = sum(
            1 for other_frame, ox, oy in nodes
            if other_frame != frame_idx and math.dist((x, y), (ox, oy)) <= radius_px
        )
        if overlaps + 1 < min_repeats:
            kept.append((frame_idx, x, y))
    return kept


def _per_frame_speeds(nodes: list[tuple[int, float, float]]) -> list[float]:
    """이웃한 노드 사이의 프레임당 이동량."""
    speeds = []
    for (f0, x0, y0), (f1, x1, y1) in zip(nodes, nodes[1:]):
        speeds.append(math.dist((x0, y0), (x1, y1)) / max(1, f1 - f0))
    return speeds


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _trim_static_ends(
    nodes: list[tuple[int, float, float]], min_step_px: float
) -> list[tuple[int, float, float]]:
    """
    앞뒤에서 프레임당 이동이 min_step_px 미만인 구간을 잘라낸다.

    정지 구간의 **마지막 점도 함께 버린다.** 그 점은 정지 물체의 좌표이고, 거기서
    다음 점으로의 "큰 이동"은 공의 운동이 아니라 다른 물체로 건너뛴 것이기 때문이다.
    """
    if len(nodes) < 2:
        return list(nodes)

    speeds = _per_frame_speeds(nodes)
    start, end = 0, len(nodes) - 1

    while start < end and speeds[start] < min_step_px:
        start += 1
    if start > 0:
        start += 1

    while end > start and speeds[end - 1] < min_step_px:
        end -= 1
    if end < len(nodes) - 1:
        end -= 1

    return nodes[start:end + 1] if start <= end else []


def extract_trajectory_candidates(
    video_path: str,
    timestamp_sec: float,
    model,
    lookback_start_sec: float = 3.0,
    lookback_end_sec: float = 0.3,
    target_fps: float | None = None,
    imgsz: int = 640,
    conf: float | None = None,
) -> list[tuple[int, list[tuple[float, float, float, float, float]]]]:
    """
    윈도우 안 프레임별로 **모든** 감지 후보를
    (frame_idx, [(x, y, conf, w, h), ...])로 돌려준다.

    `extract_trajectory_points`는 프레임마다 최고 신뢰도 1건만 남기는데, 정지 오탐이
    실제 공보다 신뢰도가 높은 경우가 있어 그 시점에 공이 통째로 버려진다(TS-017).
    후보를 전부 남겨야 longest_moving_chain이 운동성으로 고를 수 있다.

    박스 크기(w, h)를 함께 남기는 이유: 픽셀 좌표만으로는 절대 속도를 복원할 수 없어
    OFFSPEED(체인지업·스플리터)가 패스트볼과 갈리지 않는다는 것이 실측으로 확인됐다
    (좌표 특징 7개를 추가해도 OFFSPEED AUC 0.641 -> 0.648). 공이 카메라에 가까워지는
    속도, 즉 박스가 커지는 속도가 남은 유일한 속도 대리 지표다. mp4는 처리 후 지우므로
    여기서 안 남기면 재수집이 한 번 더 필요해진다.
    """
    from yolo_detector import detect_ball_in_frame  # 지연 import: 순수 함수는 ultralytics에 비의존

    frames = frames_in_window(
        video_path, timestamp_sec, lookback_start_sec, lookback_end_sec,
        target_fps=target_fps,
    )
    result = []
    for frame_idx, frame in enumerate(frames):
        detections = detect_ball_in_frame(model, frame, imgsz=imgsz, conf=conf)
        if not detections:
            continue
        result.append((frame_idx, [
            (
                float(d["cx"]), float(d["cy"]), float(d["conf"]),
                float(d["bbox"][2] - d["bbox"][0]), float(d["bbox"][3] - d["bbox"][1]),
            )
            for d in detections
        ]))
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
