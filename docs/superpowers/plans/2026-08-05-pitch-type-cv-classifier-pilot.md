# 영상 기반 구종 그룹 분류기 (CV 파일럿) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 방송 영상 궤적(YOLO 픽셀 좌표)만으로 투구를 3그룹(FASTBALL/BREAKING/OFFSPEED)으로 분류하는
파일럿 파이프라인을 만들고, Statcast 정답 대비 정확도를 검증한다.

**Architecture:** OCR 타임스탬프(`scan_pitch_overlays`)를 단일 기준점 삼아 Statcast 정답과 YOLO
궤적 구간을 동기화한다. 궤적 → 수작업 특징(속도·곡률·낙하폭) → GradientBoosting 분류기. 모든 I/O
의존성(비디오 다운로드, OCR, 궤적 추출)은 의존성 주입으로 분리해 순수 로직을 유닛 테스트하고, 실제
연결은 `scripts/build_pitch_group_dataset.py`에서 한 번에 수행한다.

**Tech Stack:** Python, scikit-learn(GradientBoostingClassifier), opencv-python-headless, pandas,
joblib, pytest(신규 추가), 기존 `yolo_detector.py`/`pose_detector.py`/`pybaseball` 재사용.

## Global Constraints

- 기존 `yolo_detector.py`, `pose_detector.py`, `feature_engineering.py`는 수정하지 않는다 — import만 한다
- 파일럿 데이터 수집 범위는 Fox 중계로 한정한다 (OCR 오버레이 포맷 의존)
- 구종 분류는 3그룹(FASTBALL/BREAKING/OFFSPEED)까지만 — 7개 세분류(FF/SI/FC/SL/CU/CH/FS) 아님
- 다음 구종 예측(BiLSTM) 통합은 이번 스펙 범위 밖 — 하지 않는다
- Streamlit 앱(`streamlit_app/`) 통합은 이번 스펙 범위 밖 — 하지 않는다
- 딥러닝(3D-CNN, end-to-end 시퀀스 모델)은 쓰지 않는다 — 수작업 특징 + 경량 ML만
- 성공 기준은 완벽한 정확도가 아니라 3그룹 랜덤 베이스라인(33%)을 유의미하게 상회하는지
- 산출물은 `output/pitch_type_cv/`에 저장한다 (없으면 생성)
- 학습·검증은 `notebooks/01_pitch_group_classifier.ipynb`에서 진행한다 (스크립트로 블랙박스 학습 금지)
- `pitch_group_map.py`의 매핑: FF·SI·FC→FASTBALL, SL·CU→BREAKING, CH·FS→OFFSPEED, 그 외(KN 등)→제외

**참고 스펙:** `docs/superpowers/specs/2026-08-05-pitch-type-cv-classifier-design.md`

---

## File Structure

```
conftest.py                                    # 신규 — tests/에서 src/ import 가능하게 sys.path 설정
requirements-dev.txt                           # 수정 — pytest 추가
src/pitch_type_cv/
  __init__.py                                  # 신규 — 패키지 마커
  pitch_group_map.py                           # 신규 — Statcast pitch_type → 3그룹 매핑
  trajectory_features.py                       # 신규 — 궤적 윈도우 추출 + 특징 계산
  group_classifier.py                          # 신규 — 분류기 학습/추론/저장
  dataset.py                                   # 신규 — OCR/Statcast 페어링 + 데이터셋 조립 (의존성 주입)
scripts/
  build_pitch_group_dataset.py                 # 신규 — 실제 의존성 연결 + GAME_LIST 실행
notebooks/
  01_pitch_group_classifier.ipynb              # 신규 — 학습·검증·시각화
tests/pitch_type_cv/
  test_pitch_group_map.py                      # 신규
  test_trajectory_features.py                  # 신규
  test_group_classifier.py                     # 신규
  test_dataset.py                               # 신규
```

---

### Task 1: 테스트 인프라 설정 + 구종 그룹 매핑 (`pitch_group_map.py`)

**Files:**
- Modify: `requirements-dev.txt` (pytest 추가)
- Create: `conftest.py`
- Create: `src/pitch_type_cv/__init__.py`
- Create: `src/pitch_type_cv/pitch_group_map.py`
- Test: `tests/pitch_type_cv/test_pitch_group_map.py`

**Interfaces:**
- Produces: `pitch_type_to_group(pitch_type: str) -> str | None` — 이후 모든 태스크가 구종 그룹 판정에 사용

- [ ] **Step 1: pytest 의존성 추가 및 설치**

`requirements-dev.txt`의 jupyter 블록 위에 추가:

```
pytest==8.4.2
```

설치:

```bash
venv/bin/pip install pytest==8.4.2
```

- [ ] **Step 2: conftest.py 작성 (src/ import 경로 설정)**

`conftest.py` (레포 루트):

```python
"""pytest가 src/ 하위 모듈을 import할 수 있도록 경로를 추가한다."""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
```

- [ ] **Step 3: 실패하는 테스트 작성**

`tests/pitch_type_cv/test_pitch_group_map.py`:

```python
from pitch_type_cv.pitch_group_map import pitch_type_to_group


def test_maps_fastball_family():
    assert pitch_type_to_group("FF") == "FASTBALL"
    assert pitch_type_to_group("SI") == "FASTBALL"
    assert pitch_type_to_group("FC") == "FASTBALL"


def test_maps_breaking_family():
    assert pitch_type_to_group("SL") == "BREAKING"
    assert pitch_type_to_group("CU") == "BREAKING"


def test_maps_offspeed_family():
    assert pitch_type_to_group("CH") == "OFFSPEED"
    assert pitch_type_to_group("FS") == "OFFSPEED"


def test_returns_none_for_unmapped_pitch_type():
    assert pitch_type_to_group("KN") is None
    assert pitch_type_to_group("") is None
```

- [ ] **Step 4: 테스트 실행 → 실패 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_pitch_group_map.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'pitch_type_cv'`

- [ ] **Step 5: 구현**

`src/pitch_type_cv/__init__.py`:

```python
"""영상 궤적 기반 구종 그룹(3-class) 분류 파이프라인."""
```

`src/pitch_type_cv/pitch_group_map.py`:

```python
"""Statcast pitch_type 코드를 3그룹(FASTBALL/BREAKING/OFFSPEED)으로 매핑한다."""

PITCH_GROUP_MAP: dict[str, str] = {
    "FF": "FASTBALL", "SI": "FASTBALL", "FC": "FASTBALL",
    "SL": "BREAKING", "CU": "BREAKING",
    "CH": "OFFSPEED", "FS": "OFFSPEED",
}


def pitch_type_to_group(pitch_type: str) -> str | None:
    """Statcast pitch_type을 3그룹으로 매핑. 매핑되지 않는 코드(KN 등)는 None."""
    return PITCH_GROUP_MAP.get(pitch_type)
```

- [ ] **Step 6: 테스트 실행 → 통과 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_pitch_group_map.py -v
```
Expected: PASS (4 passed)

- [ ] **Step 7: 커밋**

```bash
git add conftest.py requirements-dev.txt src/pitch_type_cv/__init__.py \
        src/pitch_type_cv/pitch_group_map.py tests/pitch_type_cv/test_pitch_group_map.py
git commit -m "feat: 구종 3그룹 매핑 + pytest 인프라 추가"
```

---

### Task 2: 궤적 → 특징 계산 (`trajectory_features.py` 순수 함수)

**Files:**
- Modify: `src/pitch_type_cv/trajectory_features.py` (신규 생성, Task 3에서 이어서 확장)
- Test: `tests/pitch_type_cv/test_trajectory_features.py`

**Interfaces:**
- Consumes: 없음 (순수 함수)
- Produces: `FEATURE_COLUMNS: list[str]`, `compute_trajectory_features(trajectory: list[tuple[float, float]], min_points: int = 3) -> dict | None` — Task 4(분류기), Task 6(데이터셋 조립)이 사용

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/pitch_type_cv/test_trajectory_features.py`:

```python
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_trajectory_features.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'pitch_type_cv.trajectory_features'`

- [ ] **Step 3: 구현**

`src/pitch_type_cv/trajectory_features.py` (이 태스크에서는 아래 부분만 작성 — `frames_in_window`,
`trajectory_from_frames`, `extract_trajectory_window`은 Task 3에서 같은 파일에 추가):

```python
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
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_trajectory_features.py -v
```
Expected: PASS (6 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/pitch_type_cv/trajectory_features.py tests/pitch_type_cv/test_trajectory_features.py
git commit -m "feat: 궤적 기반 구종 분류 특징 계산 함수 추가"
```

---

### Task 3: 궤적 윈도우 추출 (`frames_in_window`, `trajectory_from_frames`, `extract_trajectory_window`)

**Files:**
- Modify: `src/pitch_type_cv/trajectory_features.py` (Task 2 파일에 이어서 추가)
- Test: `tests/pitch_type_cv/test_trajectory_features.py` (이어서 추가)

**Interfaces:**
- Consumes: `yolo_detector.detect_ball_in_frame(model, frame, imgsz=640) -> list[dict]` (반환 형식
  `[{"bbox":[x1,y1,x2,y2], "conf":float, "cx":int, "cy":int}, ...]`, 기존 코드, 수정하지 않음)
- Produces: `frames_in_window(video_path, timestamp_sec, lookback_start_sec=3.0, lookback_end_sec=0.3, max_frames=90) -> list[np.ndarray]`,
  `trajectory_from_frames(frames, detect_fn) -> list[tuple[float, float]]`,
  `extract_trajectory_window(video_path, timestamp_sec, model, lookback_start_sec=3.0, lookback_end_sec=0.3) -> list[tuple[float, float]]`
  — Task 6(데이터셋 조립)이 `extract_trajectory_window`를 사용

- [ ] **Step 1: 실패하는 테스트 작성 (합성 비디오 fixture 포함)**

`tests/pitch_type_cv/test_trajectory_features.py` 파일 끝에 추가:

```python
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_trajectory_features.py -v
```
Expected: FAIL with `ImportError: cannot import name 'frames_in_window'`

- [ ] **Step 3: 구현**

`src/pitch_type_cv/trajectory_features.py` 파일 상단 import에 추가:

```python
from typing import Callable

import cv2
import numpy as np
```

파일 끝에 추가:

```python
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

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[np.ndarray] = []
    frame_idx = start_frame
    while frame_idx <= end_frame and len(frames) < max_frames:
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
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_trajectory_features.py -v
```
Expected: PASS (10 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/pitch_type_cv/trajectory_features.py tests/pitch_type_cv/test_trajectory_features.py
git commit -m "feat: OCR 타임스탬프 기반 궤적 윈도우 추출 함수 추가"
```

---

### Task 4: 구종 그룹 분류기 (`group_classifier.py`)

**Files:**
- Create: `src/pitch_type_cv/group_classifier.py`
- Test: `tests/pitch_type_cv/test_group_classifier.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS` from `pitch_type_cv.trajectory_features` (Task 2)
- Produces: `train_classifier(X: pd.DataFrame, y: list[str], random_state: int = 42) -> GradientBoostingClassifier`,
  `predict_group(model, features: dict) -> tuple[str, dict[str, float]]`,
  `save_classifier(model, path: str) -> None`, `load_classifier(path: str) -> GradientBoostingClassifier`
  — Task 9(노트북)가 이 네 함수를 사용

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/pitch_type_cv/test_group_classifier.py`:

```python
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
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_group_classifier.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'pitch_type_cv.group_classifier'`

- [ ] **Step 3: 구현**

`src/pitch_type_cv/group_classifier.py`:

```python
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
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_group_classifier.py -v
```
Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/pitch_type_cv/group_classifier.py tests/pitch_type_cv/test_group_classifier.py
git commit -m "feat: 궤적 특징 기반 구종 그룹 분류기 추가"
```

---

### Task 5: OCR 타임스탬프 ↔ Statcast 페어링 (`dataset.py` 순수 로직)

**Files:**
- Create: `src/pitch_type_cv/dataset.py` (이 태스크에서는 페어링 함수만, Task 6·7에서 이어서 확장)
- Test: `tests/pitch_type_cv/test_dataset.py`

**Interfaces:**
- Produces: `pair_timestamps_with_statcast(timestamps: list[float], pitch_types: list[str]) -> list[tuple[float, str]]`
  — Task 6이 사용

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/pitch_type_cv/test_dataset.py`:

```python
import logging

from pitch_type_cv.dataset import pair_timestamps_with_statcast


def test_pairs_equal_length_lists_fully():
    result = pair_timestamps_with_statcast([1.0, 2.0, 3.0], ["FF", "SL", "CH"])
    assert result == [(1.0, "FF"), (2.0, "SL"), (3.0, "CH")]


def test_truncates_to_shorter_list_when_counts_mismatch():
    result = pair_timestamps_with_statcast([1.0, 2.0, 3.0, 4.0], ["FF", "SL"])
    assert result == [(1.0, "FF"), (2.0, "SL")]


def test_logs_warning_on_count_mismatch(caplog):
    with caplog.at_level(logging.WARNING):
        pair_timestamps_with_statcast([1.0, 2.0, 3.0], ["FF"])

    assert any("3" in r.message and "1" in r.message for r in caplog.records)
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_dataset.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'pitch_type_cv.dataset'`

- [ ] **Step 3: 구현**

`src/pitch_type_cv/dataset.py`:

```python
"""OCR 타임스탬프와 Statcast 정답을 페어링해 학습용 데이터셋을 조립한다."""
import logging

logger = logging.getLogger(__name__)


def pair_timestamps_with_statcast(
    timestamps: list[float], pitch_types: list[str]
) -> list[tuple[float, str]]:
    """
    i번째 OCR 타임스탬프를 i번째 Statcast pitch_type과 순서대로 페어링한다.
    개수가 다르면 짧은 쪽 길이까지만 앞에서부터 매칭하고 경고를 남긴다.
    """
    n = min(len(timestamps), len(pitch_types))
    if len(timestamps) != len(pitch_types):
        logger.warning(
            "OCR 타임스탬프(%d개)와 Statcast 투구 수(%d개)가 달라 앞에서부터 %d개만 매칭합니다.",
            len(timestamps), len(pitch_types), n,
        )
    return list(zip(timestamps[:n], pitch_types[:n]))
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_dataset.py -v
```
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/pitch_type_cv/dataset.py tests/pitch_type_cv/test_dataset.py
git commit -m "feat: OCR 타임스탬프-Statcast 페어링 로직 추가"
```

---

### Task 6: 단일 경기 데이터셋 조립 (`build_dataset_for_game`)

**Files:**
- Modify: `src/pitch_type_cv/dataset.py` (Task 5 파일에 이어서 추가)
- Test: `tests/pitch_type_cv/test_dataset.py` (이어서 추가)

**Interfaces:**
- Consumes: `pitch_type_to_group` (Task 1), `compute_trajectory_features` (Task 2),
  `pair_timestamps_with_statcast` (Task 5, 이 파일 내부)
- Produces: `build_dataset_for_game(game_pk, youtube_url, fetch_statcast, resolve_video, scan_overlays, extract_trajectory) -> pd.DataFrame`
  — Task 7이 사용

  주입되는 콜러블 계약:
  - `fetch_statcast(game_pk: int) -> pd.DataFrame` — **시간순 정렬된** `pitch_type` 컬럼을 가진
    DataFrame을 반환해야 함 (정렬 책임은 호출자에게 있음)
  - `resolve_video(youtube_url: str) -> str` — 로컬 비디오 파일 경로 반환
  - `scan_overlays(video_path: str) -> tuple[list[float], list[dict]]` — 첫 번째 반환값(타임스탬프)만 사용
  - `extract_trajectory(video_path: str, timestamp_sec: float) -> list[tuple[float, float]]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/pitch_type_cv/test_dataset.py` 파일 끝에 추가:

```python
import pandas as pd

from pitch_type_cv.dataset import build_dataset_for_game


def test_build_dataset_for_game_produces_labeled_rows():
    statcast_df = pd.DataFrame({"pitch_type": ["FF", "SL", "CH"]})
    trajectories = {
        1.0: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        2.0: [(0.0, 0.0), (1.0, 3.0), (2.0, 0.0)],
        3.0: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
    }

    df = build_dataset_for_game(
        game_pk=775300,
        youtube_url="https://youtu.be/fake",
        fetch_statcast=lambda game_pk: statcast_df,
        resolve_video=lambda url: "fake_video.mp4",
        scan_overlays=lambda video_path: ([1.0, 2.0, 3.0], []),
        extract_trajectory=lambda video_path, ts: trajectories[ts],
    )

    assert len(df) == 3
    assert set(df["group"]) == {"FASTBALL", "BREAKING", "OFFSPEED"}
    assert (df["game_pk"] == 775300).all()


def test_build_dataset_for_game_excludes_unmapped_pitch_types():
    statcast_df = pd.DataFrame({"pitch_type": ["FF", "KN"]})  # KN은 매핑 안 됨
    trajectory = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]

    df = build_dataset_for_game(
        game_pk=775300,
        youtube_url="https://youtu.be/fake",
        fetch_statcast=lambda game_pk: statcast_df,
        resolve_video=lambda url: "fake_video.mp4",
        scan_overlays=lambda video_path: ([1.0, 2.0], []),
        extract_trajectory=lambda video_path, ts: trajectory,
    )

    assert len(df) == 1
    assert df.iloc[0]["group"] == "FASTBALL"


def test_build_dataset_for_game_excludes_short_trajectories():
    statcast_df = pd.DataFrame({"pitch_type": ["FF", "SL"]})
    trajectories = {
        1.0: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        2.0: [(0.0, 0.0)],  # 포인트 부족 (min_points=3 미만)
    }

    df = build_dataset_for_game(
        game_pk=775300,
        youtube_url="https://youtu.be/fake",
        fetch_statcast=lambda game_pk: statcast_df,
        resolve_video=lambda url: "fake_video.mp4",
        scan_overlays=lambda video_path: ([1.0, 2.0], []),
        extract_trajectory=lambda video_path, ts: trajectories[ts],
    )

    assert len(df) == 1
    assert df.iloc[0]["group"] == "FASTBALL"
```

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_dataset.py -v
```
Expected: FAIL with `ImportError: cannot import name 'build_dataset_for_game'`

- [ ] **Step 3: 구현**

`src/pitch_type_cv/dataset.py` 상단 import에 추가:

```python
from typing import Callable

import pandas as pd

from pitch_type_cv.pitch_group_map import pitch_type_to_group
from pitch_type_cv.trajectory_features import compute_trajectory_features
```

파일 끝에 추가:

```python
def build_dataset_for_game(
    game_pk: int,
    youtube_url: str,
    fetch_statcast: Callable[[int], pd.DataFrame],
    resolve_video: Callable[[str], str],
    scan_overlays: Callable[[str], tuple[list[float], list[dict]]],
    extract_trajectory: Callable[[str, float], list[tuple[float, float]]],
) -> pd.DataFrame:
    """
    한 경기의 영상+Statcast에서 (궤적 특징, 구종 그룹) 라벨링된 데이터셋 행을 조립한다.
    매핑 안 되는 구종(OTHER)과 궤적 포인트 부족 샘플은 제외한다.
    """
    video_path = resolve_video(youtube_url)
    statcast_df = fetch_statcast(game_pk)
    pitch_types = statcast_df["pitch_type"].tolist()

    timestamps, _ = scan_overlays(video_path)
    pairs = pair_timestamps_with_statcast(timestamps, pitch_types)

    rows = []
    for timestamp_sec, pitch_type in pairs:
        group = pitch_type_to_group(pitch_type)
        if group is None:
            continue

        trajectory = extract_trajectory(video_path, timestamp_sec)
        features = compute_trajectory_features(trajectory)
        if features is None:
            continue

        rows.append({
            **features,
            "group": group,
            "game_pk": game_pk,
            "timestamp_sec": timestamp_sec,
        })

    return pd.DataFrame(rows)
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_dataset.py -v
```
Expected: PASS (6 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/pitch_type_cv/dataset.py tests/pitch_type_cv/test_dataset.py
git commit -m "feat: 단일 경기 구종 그룹 데이터셋 조립 함수 추가"
```

---

### Task 7: 다중 경기 데이터셋 조립 + 경기별 에러 처리 (`build_dataset`)

**Files:**
- Modify: `src/pitch_type_cv/dataset.py` (Task 6 파일에 이어서 추가)
- Test: `tests/pitch_type_cv/test_dataset.py` (이어서 추가)

**Interfaces:**
- Consumes: `build_dataset_for_game` (Task 6, 이 파일 내부)
- Produces: `GameSpec` (dataclass: `game_pk: int`, `youtube_url: str`), `build_dataset(games: list[GameSpec], fetch_statcast, resolve_video, scan_overlays, extract_trajectory) -> pd.DataFrame`
  — Task 8(스크립트)이 사용

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/pitch_type_cv/test_dataset.py` 파일 끝에 추가:

```python
from pitch_type_cv.dataset import GameSpec, build_dataset


def test_build_dataset_skips_failing_game_and_keeps_others(caplog):
    good_statcast = pd.DataFrame({"pitch_type": ["FF"]})
    trajectory = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]

    def fetch_statcast(game_pk: int) -> pd.DataFrame:
        if game_pk == 111:
            raise RuntimeError("Statcast 조회 실패")
        return good_statcast

    with caplog.at_level(logging.WARNING):
        df = build_dataset(
            games=[GameSpec(game_pk=111, youtube_url="https://youtu.be/bad"),
                   GameSpec(game_pk=222, youtube_url="https://youtu.be/good")],
            fetch_statcast=fetch_statcast,
            resolve_video=lambda url: "fake_video.mp4",
            scan_overlays=lambda video_path: ([1.0], []),
            extract_trajectory=lambda video_path, ts: trajectory,
        )

    assert len(df) == 1
    assert df.iloc[0]["game_pk"] == 222
    assert any("111" in r.message for r in caplog.records)


def test_build_dataset_returns_empty_dataframe_when_all_games_fail():
    df = build_dataset(
        games=[GameSpec(game_pk=111, youtube_url="https://youtu.be/bad")],
        fetch_statcast=lambda game_pk: (_ for _ in ()).throw(RuntimeError("fail")),
        resolve_video=lambda url: "fake_video.mp4",
        scan_overlays=lambda video_path: ([], []),
        extract_trajectory=lambda video_path, ts: [],
    )

    assert df.empty
```

`logging` import는 Task 5 테스트에서 이미 파일 상단에 있으므로 재사용한다.

- [ ] **Step 2: 테스트 실행 → 실패 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_dataset.py -v
```
Expected: FAIL with `ImportError: cannot import name 'GameSpec'`

- [ ] **Step 3: 구현**

`src/pitch_type_cv/dataset.py` 상단 import에 추가:

```python
from dataclasses import dataclass
```

파일 끝에 추가:

```python
@dataclass
class GameSpec:
    game_pk: int
    youtube_url: str


def build_dataset(
    games: list[GameSpec],
    fetch_statcast: Callable[[int], pd.DataFrame],
    resolve_video: Callable[[str], str],
    scan_overlays: Callable[[str], tuple[list[float], list[dict]]],
    extract_trajectory: Callable[[str, float], list[tuple[float, float]]],
) -> pd.DataFrame:
    """
    여러 경기의 데이터셋을 조립한다. 개별 경기 처리 중 예외가 나면 그 경기만 건너뛰고
    나머지는 계속 처리한다 (네트워크 실패, OCR 실패 등에 대비).
    """
    frames = []
    for game in games:
        try:
            frames.append(
                build_dataset_for_game(
                    game.game_pk, game.youtube_url,
                    fetch_statcast, resolve_video, scan_overlays, extract_trajectory,
                )
            )
        except Exception as exc:
            logger.warning("게임 %s 처리 실패, 건너뜁니다: %s", game.game_pk, exc)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: 테스트 실행 → 통과 확인**

```bash
venv/bin/python3 -m pytest tests/pitch_type_cv/test_dataset.py -v
```
Expected: PASS (8 passed)

- [ ] **Step 5: 커밋**

```bash
git add src/pitch_type_cv/dataset.py tests/pitch_type_cv/test_dataset.py
git commit -m "feat: 다중 경기 데이터셋 조립 + 경기별 에러 격리 추가"
```

---

### Task 8: 실행 스크립트 (`scripts/build_pitch_group_dataset.py`)

**Files:**
- Create: `scripts/build_pitch_group_dataset.py`

**Interfaces:**
- Consumes: `pitch_type_cv.dataset.{GameSpec, build_dataset}` (Task 7),
  `pitch_type_cv.trajectory_features.extract_trajectory_window` (Task 3),
  기존 `yolo_detector.{load_model, resolve_video_path}`, `pose_detector.scan_pitch_overlays`,
  `pybaseball.statcast_single_game` (모두 기존 코드, 수정하지 않음)
- Produces: `output/pitch_type_cv/dataset.csv`

이 태스크는 실제 네트워크·비디오·YOLO 의존성을 연결하는 배선(wiring) 코드라 유닛 테스트 대상이
아니다 (Task 6·7에서 이미 순수 로직을 페이크로 검증했음). `GAME_LIST`는 실제 5-10경기의
`(game_pk, YouTube URL)`로 채워야 동작한다 — 지금은 빈 리스트로 두고, 실행 시 안내 메시지를 출력한다.

- [ ] **Step 1: 스크립트 작성**

`scripts/build_pitch_group_dataset.py`:

```python
"""
CV 구종 그룹 분류기 파일럿 — 학습용 데이터셋 생성.
실행: venv/bin/python3 scripts/build_pitch_group_dataset.py

GAME_LIST에 Fox 중계 + YouTube에 영상이 있는 game_pk 5-10개를 직접 채운 뒤 실행한다.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pitch_type_cv.dataset import GameSpec, build_dataset  # noqa: E402
from pitch_type_cv.trajectory_features import extract_trajectory_window  # noqa: E402

# 실제 파일럿 경기 목록 (game_pk, YouTube URL) — 5-10개를 채운 뒤 실행
GAME_LIST: list[GameSpec] = [
    # GameSpec(game_pk=775300, youtube_url="https://youtu.be/gMm3EODDb6w"),
]

CACHE_DIR = os.path.join(ROOT, "streamlit_app", ".yolo_cache")
OUT_DIR = os.path.join(ROOT, "output", "pitch_type_cv")
OUT_PATH = os.path.join(OUT_DIR, "dataset.csv")


def main() -> None:
    if not GAME_LIST:
        print(
            "[중단] GAME_LIST가 비어 있습니다. "
            "5-10개의 (game_pk, YouTube URL)을 직접 채운 뒤 다시 실행하세요."
        )
        return

    from pybaseball import statcast_single_game
    from pose_detector import scan_pitch_overlays
    from yolo_detector import load_model, resolve_video_path

    print("[1/3] YOLO 모델 로드...")
    yolo_model = load_model()

    def fetch_statcast(game_pk: int):
        df = statcast_single_game(game_pk)
        return df.sort_values(
            ["game_date", "at_bat_number", "pitch_number"]
        ).reset_index(drop=True)

    def resolve_video(url: str) -> str:
        os.makedirs(CACHE_DIR, exist_ok=True)
        return resolve_video_path(url, download_dir=CACHE_DIR)

    def extract_trajectory(video_path: str, timestamp_sec: float):
        return extract_trajectory_window(video_path, timestamp_sec, yolo_model)

    print(f"[2/3] {len(GAME_LIST)}개 경기 데이터셋 조립 중...")
    dataset_df = build_dataset(
        games=GAME_LIST,
        fetch_statcast=fetch_statcast,
        resolve_video=resolve_video,
        scan_overlays=scan_pitch_overlays,
        extract_trajectory=extract_trajectory,
    )

    if dataset_df.empty:
        print("[중단] 생성된 데이터셋이 비어 있습니다. 위 경고 로그를 확인하세요.")
        return

    print(f"[3/3] 저장 중... ({len(dataset_df)}개 샘플)")
    os.makedirs(OUT_DIR, exist_ok=True)
    dataset_df.to_csv(OUT_PATH, index=False)
    print(f"저장 완료: {OUT_PATH}")
    print(dataset_df["group"].value_counts())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 구문 오류 없는지 확인 (실제 실행은 GAME_LIST 채운 뒤 별도로)**

```bash
venv/bin/python3 -c "import ast; ast.parse(open('scripts/build_pitch_group_dataset.py').read())"
```
Expected: 에러 없이 종료

```bash
venv/bin/python3 scripts/build_pitch_group_dataset.py
```
Expected: `[중단] GAME_LIST가 비어 있습니다...` 출력 (GAME_LIST가 아직 비어 있으므로 정상)

- [ ] **Step 3: 커밋**

```bash
git add scripts/build_pitch_group_dataset.py
git commit -m "feat: CV 구종 그룹 데이터셋 생성 스크립트 추가"
```

---

### Task 9: 학습·검증 노트북 (`notebooks/01_pitch_group_classifier.ipynb`)

**Files:**
- Create: `notebooks/01_pitch_group_classifier.ipynb`

**Interfaces:**
- Consumes: `output/pitch_type_cv/dataset.csv` (Task 8 스크립트 산출물),
  `pitch_type_cv.group_classifier.{train_classifier, predict_group, save_classifier}` (Task 4),
  `pitch_type_cv.trajectory_features.FEATURE_COLUMNS` (Task 2)
- Produces: `output/pitch_type_cv/confusion_matrix.png`, `output/pitch_type_cv/group_accuracy.png`,
  `output/pitch_type_cv/group_classifier.pkl`

이 노트북은 실제 파일럿 데이터(`GAME_LIST`에 실제 URL을 채워 Task 8 스크립트를 돌린 결과)가 있어야
끝까지 실행된다. NotebookEdit 도구로 아래 셀들을 순서대로 만든다.

- [ ] **Step 1: 노트북 생성 — 셀 구성**

NotebookEdit으로 `notebooks/01_pitch_group_classifier.ipynb`에 아래 순서로 셀을 추가한다
(kernel: Python 3, venv의 ipykernel 사용):

**셀 1 (markdown):**
```markdown
# 01. CV 구종 그룹 분류기 — 학습·검증

`scripts/build_pitch_group_dataset.py`가 생성한 `output/pitch_type_cv/dataset.csv`로
궤적 기반 3그룹(FASTBALL/BREAKING/OFFSPEED) 분류기를 학습하고, 홀드아웃 경기로 검증한다.

성공 기준: 3그룹 랜덤 베이스라인(33%)을 유의미하게 상회하는지 확인 (완벽한 정확도가 목표가 아님).
```

**셀 2 (code):**
```python
import os
import sys

ROOT = os.path.dirname(os.getcwd())
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from pitch_type_cv.group_classifier import predict_group, save_classifier, train_classifier
from pitch_type_cv.trajectory_features import FEATURE_COLUMNS

DATASET_PATH = os.path.join(ROOT, "output", "pitch_type_cv", "dataset.csv")
OUT_DIR = os.path.join(ROOT, "output", "pitch_type_cv")

df = pd.read_csv(DATASET_PATH)
print(f"전체 샘플 수: {len(df)}")
df["group"].value_counts()
```

**셀 3 (markdown):**
```markdown
## 홀드아웃 경기 분리

`game_pk` 중 하나를 통째로 홀드아웃으로 뺀다 (경기 내 데이터 누수를 막기 위해 투구 단위가 아닌
경기 단위로 분리).
```

**셀 4 (code):**
```python
game_pks = df["game_pk"].unique()
holdout_game_pk = game_pks[-1]
print(f"홀드아웃 경기: {holdout_game_pk} (전체 {len(game_pks)}경기 중)")

train_df = df[df["game_pk"] != holdout_game_pk].reset_index(drop=True)
holdout_df = df[df["game_pk"] == holdout_game_pk].reset_index(drop=True)

print(f"학습 샘플: {len(train_df)}  홀드아웃 샘플: {len(holdout_df)}")
```

**셀 5 (code):**
```python
model = train_classifier(train_df[FEATURE_COLUMNS + ["group"]], train_df["group"].tolist())

y_true = holdout_df["group"].tolist()
y_pred = [predict_group(model, row.to_dict())[0] for _, row in holdout_df[FEATURE_COLUMNS].iterrows()]

accuracy = accuracy_score(y_true, y_pred)
baseline = 1 / 3
print(f"홀드아웃 정확도: {accuracy:.3f}  (랜덤 베이스라인: {baseline:.3f})")
print(classification_report(y_true, y_pred))
```

**셀 6 (markdown):**
```markdown
## 혼동행렬 & 그룹별 정확도 시각화
```

**셀 7 (code):**
```python
os.makedirs(OUT_DIR, exist_ok=True)
labels = ["FASTBALL", "BREAKING", "OFFSPEED"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
ax.set_xlabel("예측")
ax.set_ylabel("실제")
ax.set_title(f"홀드아웃 혼동행렬 (game_pk={holdout_game_pk})")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=150)
plt.show()
```

**셀 8 (code):**
```python
per_group_acc = (
    pd.DataFrame({"true": y_true, "pred": y_pred})
    .assign(correct=lambda d: d["true"] == d["pred"])
    .groupby("true")["correct"].mean()
    .reindex(labels)
)

fig, ax = plt.subplots(figsize=(5, 4))
per_group_acc.plot(kind="bar", ax=ax, color="#3b82f6")
ax.axhline(baseline, color="red", linestyle="--", label="랜덤 베이스라인 (33%)")
ax.set_ylabel("정확도")
ax.set_title("그룹별 정확도")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "group_accuracy.png"), dpi=150)
plt.show()
```

**셀 9 (code):**
```python
model_path = os.path.join(OUT_DIR, "group_classifier.pkl")
save_classifier(model, model_path)
print(f"모델 저장 완료: {model_path}")
```

**셀 10 (markdown):**
```markdown
## 해석

- `accuracy`가 33%(랜덤 베이스라인)를 유의미하게 상회하면 궤적 기반 분류 방향이 통한다는 신호.
- 파일럿 규모(5-10경기)이므로 특정 그룹 샘플 부족으로 정확도가 불안정할 수 있음 — 그룹별 정확도
  그래프에서 표본 수가 적은 그룹은 참고용으로만 본다.
- 다음 단계(별도 스펙): 데이터 규모 확대, 다음 구종 예측 파이프라인과의 연결, Streamlit 통합.
```

- [ ] **Step 2: 노트북 실행 확인 (실제 데이터가 있을 때)**

```bash
venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_pitch_group_classifier.ipynb
```
Expected: 에러 없이 종료, `output/pitch_type_cv/`에 `confusion_matrix.png`, `group_accuracy.png`,
`group_classifier.pkl` 생성. `output/pitch_type_cv/dataset.csv`가 아직 없다면(Task 8을 실제
데이터로 아직 안 돌렸다면) `FileNotFoundError`가 나는 게 정상 — Task 10에서 처리.

- [ ] **Step 3: 커밋**

```bash
git add notebooks/01_pitch_group_classifier.ipynb
git commit -m "feat: 구종 그룹 분류기 학습·검증 노트북 추가"
```

---

### Task 10: 파일럿 실행 (수동, 실제 데이터 필요)

**이 태스크는 코드 작성이 아니라 실행 절차다.** Task 1-9까지는 전부 페이크/합성 데이터로 검증된
코드이며, 실제 정확도를 확인하려면 진짜 경기 영상이 필요하다.

- [ ] **Step 1: 파일럿 경기 5-10개 선정**

Fox 중계 + YouTube에 전체 경기 영상이 있는 `game_pk`를 조사해 `scripts/build_pitch_group_dataset.py`의
`GAME_LIST`에 채운다. 기존 고정 데모 경기(`game_pk=775300`, `https://youtu.be/gMm3EODDb6w`)를
1번째 항목으로 포함하면 이미 검증된 영상이라 안전하다.

- [ ] **Step 2: tesseract 설치 확인**

```bash
which tesseract || brew install tesseract
```

- [ ] **Step 3: 데이터셋 생성 실행**

```bash
venv/bin/python3 scripts/build_pitch_group_dataset.py
```
로그에서 경기별 OCR/Statcast 개수 불일치, 궤적 부족 제외 건수를 확인한다.
`output/pitch_type_cv/dataset.csv` 생성 확인.

- [ ] **Step 4: 노트북 실행 및 결과 확인**

```bash
venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/01_pitch_group_classifier.ipynb
```
홀드아웃 정확도가 33% 랜덤 베이스라인을 유의미하게 상회하는지 확인한다.

- [ ] **Step 5: 결과를 트러블슈팅/실험 기록에 남길지 판단**

정확도가 기대와 다르거나(예: 특정 그룹만 심하게 낮음, 궤적 감지 실패율이 높음) 원인 파악에 시간이
걸렸다면 프로젝트 `CLAUDE.md`의 트러블슈팅 기록 기준에 따라 `docs/TROUBLESHOOTING.md`에 기록한다.

---

## Self-Review 결과

- **스펙 커버리지**: 목표 1-5(데이터셋 구성, 특징 추출, 3그룹 분류기 학습, 홀드아웃 검증, 노트북
  시각화) 모두 Task 1-9에서 구현. 에러 처리 표의 5개 상황 모두 Task 5-7에서 테스트로 커버
  (다운로드 실패/OCR 0개는 Task 7의 예외 격리로, 궤적 부족/개수 불일치/구종 매핑 제외는 Task 5-6의
  순수 로직으로 커버). 비범위 항목(다음 구종 예측, Streamlit 통합, 7세분류, 딥러닝)은 어느 태스크에도
  포함하지 않음.
- **플레이스홀더 스캔**: 없음. `GAME_LIST = []`는 실제 데이터 대기 상태를 나타내는 정상적인 초기값이며
  Task 10에서 채우는 절차를 명시함.
- **타입 일관성**: `pitch_type_to_group`, `compute_trajectory_features`, `FEATURE_COLUMNS`,
  `extract_trajectory_window`, `GameSpec`, `build_dataset`/`build_dataset_for_game` 시그니처를
  모든 태스크에서 동일하게 사용. `group_classifier.py`는 `load_model`(YOLO, 기존 코드)과의 이름
  충돌을 피하기 위해 `save_classifier`/`load_classifier`로 명명함.
