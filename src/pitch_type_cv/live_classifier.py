"""
영상 한 투구 → 구종 판정. 앱이 Statcast API 없이 실측을 내는 경로다.

앱은 지금 실측을 Statcast API에서 받는다. 그건 정답이지만 API가 있어야만 나온다.
이 모듈은 같은 값을 **영상만으로** 낸다 — 그게 이 프로젝트가 증명하려던 것이다.

범위는 FASTBALL vs BREAKING 2분류다. OFFSPEED는 중계 영상 궤적으로 갈리지 않는다는
것이 실측으로 확정됐다 (ADR-0012, TS-024).

판정에 실패하면 실패로 보고한다. 궤적이 안 잡히는 투구가 14%인데, 그때 아무 구종이나
돌려주면 앱은 그걸 판정으로 표시한다. PitchVerdict가 falsy라 실수로 쓰기 어렵다.
"""
import os
from dataclasses import dataclass, field

import pandas as pd

from pitch_type_cv.trajectory_features import (
    FEATURE_COLUMNS,
    box_sizes_for_chain,
    compute_trajectory_features,
)

TWO_CLASSES = ["FASTBALL", "BREAKING"]

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TWO_CLASS_MODEL_PATH = os.path.join(
    _ROOT, "output", "pitch_type_cv", "two_class_classifier.pkl"
)

# 고정 창은 안 된다는 것이 실측으로 확인됐다 (TS-025). Savant 클립은 투구가 항상 같은
# 위치에 오도록 잘려 있지만, 전체 중계 영상에서 OCR 오버레이 시각과 실제 투구 시각의
# 간격은 0.4초에서 7.4초까지 흩어진다. 폭 1.4초짜리 창을 어디에 놓아도 절반 이상을
# 놓쳐서, 창 위치 5개를 훑어도 판정률이 30% 근처에서 평평했다.
#
# 그래서 넓게 찾고 사슬 알고리즘이 공을 고르게 한다. 넓은 창 실측 확보율은 80%로
# 실험실(86.3%)에 근접한다. 다만 너무 넓히면 직전 투구나 리플레이를 잡으므로
# t-4.0 ~ t+0.5초로 묶는다 (실측 오프셋 8개 중 7개가 이 안에 들어온다).
DEFAULT_LOOKBACK_START_SEC = 4.0
DEFAULT_LOOKBACK_END_SEC = -0.5

# 사슬을 찾은 뒤 프레임 인덱스를 이 값에 맞춰 평행이동한다.
#
# 학습 데이터의 end_frame은 21±4로 좁게 몰려 있다(IQR 19~22). 클립이 일정하게 잘려
# 있어서 생긴 값이라, 창 원점이 흔들리는 앱에서는 같은 의미를 갖지 못한다. 넓은 창을
# 쓰면 end_frame이 수백까지 커져 모델이 학습에서 못 본 값을 받는다.
#
# 프레임 인덱스의 절대값에 의존하는 특징은 end_frame 하나뿐이다 — frame_span, 속도비,
# 수직가속(2차 계수), 박스 기울기는 전부 평행이동 불변이다. 그래서 사슬 끝을 학습
# 중앙값에 맞추는 것으로 충분하다.
#
# 이 처리로 end_frame은 앱에서 상수가 된다. 즉 그 특징의 판별력(중요도 0.042)을
# 포기하는 것이다. 앱에는 신뢰할 창 원점이 없으므로 애초에 정보가 없고, 학습 중앙값은
# 그중 가장 덜 해로운 대입값이다.
TRAINING_END_FRAME = 21

# 사슬 임계값. 학습 데이터셋을 만든 값과 같아야 한다 — 다르면 같은 영상에서 다른 궤적이
# 나오고, 모델은 자기가 못 본 분포를 받는다 (build_pitch_group_clips_dataset.py).
CHAIN_MAX_JUMP_PX = 60
CHAIN_MIN_TOTAL_MOVE_PX = 30


@dataclass(frozen=True)
class PitchVerdict:
    """구종 판정 결과. 실패는 falsy라 `if verdict:`로 걸러진다."""

    group: str | None
    probabilities: dict[str, float] = field(default_factory=dict)
    n_points: int = 0
    reason: str = "ok"

    @property
    def ok(self) -> bool:
        return self.group is not None

    def __bool__(self) -> bool:
        return self.ok

    @property
    def confidence(self) -> float:
        return max(self.probabilities.values()) if self.probabilities else 0.0

    @classmethod
    def failed(cls, reason: str, n_points: int) -> "PitchVerdict":
        return cls(group=None, probabilities={}, n_points=n_points, reason=reason)


def classify_framed_trajectory(model, framed: list[tuple[int, float, float, float]]) -> PitchVerdict:
    """
    (frame_idx, x, y, box_size) 궤적으로 구종을 판정한다.

    입력 형식은 build_clip_dataset의 extractor 계약과 같다 — 학습과 추론이 같은
    특징 계산을 타야 분포가 어긋나지 않는다.
    """
    if not framed:
        return PitchVerdict.failed("no_trajectory", 0)

    features = compute_trajectory_features(
        [(x, y) for _f, x, y, _s in framed],
        frame_indices=[f for f, _x, _y, _s in framed],
        box_sizes=[s for _f, _x, _y, s in framed],
    )
    if features is None:
        return PitchVerdict.failed("too_few_points", len(framed))

    # 이름 있는 DataFrame으로 넘긴다. 리스트로 주면 sklearn이 컬럼 순서에만 의존하고,
    # FEATURE_COLUMNS 순서가 바뀌는 날 조용히 다른 특징으로 예측한다.
    row = pd.DataFrame([features])[FEATURE_COLUMNS]
    group = str(model.predict(row)[0])
    proba = model.predict_proba(row)[0]
    return PitchVerdict(
        group=group,
        probabilities={str(c): float(p) for c, p in zip(model.classes_, proba)},
        n_points=len(framed),
    )


def align_chain_frames(chain: list[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    """
    사슬의 프레임 인덱스를 평행이동해 끝 프레임을 학습 중앙값에 맞춘다.

    간격은 보존된다 — 속도·가속 특징이 간격에서 나오므로 여기가 흔들리면 안 된다.
    좌표도 건드리지 않는다. 바뀌는 특징은 end_frame 하나다.
    """
    if not chain:
        return []
    shift = TRAINING_END_FRAME - chain[-1][0]
    return [(frame + shift, x, y) for frame, x, y in chain]


def verdict_from_chain(model, chain, candidates_by_frame) -> PitchVerdict:
    """
    longest_moving_chain_frames의 사슬과 감지 후보를 받아 판정한다.

    사슬은 (frame, x, y)만 담고 박스 크기는 후보 쪽에 남아 있다. 좌표로 다시 맞춰야
    하는데, 한 프레임에 정지 오탐과 실제 공이 함께 있을 수 있어 인덱스로는 못 맞춘다.
    """
    if not chain:
        return PitchVerdict.failed("no_trajectory", 0)

    # 박스를 먼저 뽑는다. box_sizes_for_chain이 frame_idx를 키로 후보를 찾으므로
    # 재색인 뒤에 부르면 전부 못 찾아 0이 된다 — 조용히 박스 특징만 죽는 형태다.
    sizes = box_sizes_for_chain(chain, candidates_by_frame)
    framed = [(f, x, y, size)
              for (f, x, y), size in zip(align_chain_frames(chain), sizes)]
    return classify_framed_trajectory(model, framed)


def classify_video_pitch(
    model,
    detector,
    video_path: str,
    timestamp_sec: float,
    lookback_start_sec: float = DEFAULT_LOOKBACK_START_SEC,
    lookback_end_sec: float = DEFAULT_LOOKBACK_END_SEC,
    imgsz: int = 960,
    conf: float = 0.05,
) -> PitchVerdict:
    """
    영상의 한 시점 주변에서 공 궤적을 찾아 구종을 판정한다.

    감지 설정(imgsz 960 / conf 0.05)은 학습 때와 같아야 한다 (ADR-0010).
    ultralytics를 부르므로 순수 함수가 아니다 — 단위 테스트는 위 두 함수로 한다.
    """
    from pitch_type_cv.trajectory_features import (
        extract_trajectory_candidates,
        longest_moving_chain_frames,
    )

    candidates = extract_trajectory_candidates(
        video_path, timestamp_sec, detector,
        lookback_start_sec=lookback_start_sec,
        lookback_end_sec=lookback_end_sec,
        imgsz=imgsz, conf=conf,
    )
    if not candidates:
        return PitchVerdict.failed("no_detections", 0)

    chain = longest_moving_chain_frames(
        candidates, CHAIN_MAX_JUMP_PX, CHAIN_MIN_TOTAL_MOVE_PX
    )
    return verdict_from_chain(model, chain, candidates)
