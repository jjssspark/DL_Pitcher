"""영상 한 투구 → 구종 판정 경로 테스트.

앱이 Statcast API 없이 실측을 내는 경로다. 여기서 중요한 것은 정확도가 아니라
**실패를 실패로 보고하는가**다 — 궤적을 못 잡았을 때 조용히 아무 구종이나
돌려주면 앱은 그걸 판정으로 표시한다.
"""
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from pitch_type_cv.live_classifier import (
    TRAINING_END_FRAME,
    TWO_CLASSES,
    PitchVerdict,
    align_chain_frames,
    classify_framed_trajectory,
    verdict_from_chain,
)
from pitch_type_cv.trajectory_features import FEATURE_COLUMNS

# 학습 데이터와 같은 형식: (frame_idx, x, y, box_size)
VALID_FRAMED = [
    (20, 100.0, 100.0, 10.0),
    (21, 110.0, 120.0, 11.0),
    (23, 120.0, 145.0, 13.0),
    (24, 130.0, 175.0, 14.0),
]


@pytest.fixture
def model() -> GradientBoostingClassifier:
    """FEATURE_COLUMNS 전체로 학습한 최소 2분류 모델."""
    rows = []
    for i in range(20):
        fast = {c: float(i) for c in FEATURE_COLUMNS}
        slow = {c: float(i + 100) for c in FEATURE_COLUMNS}
        rows.append({**fast, "group": "FASTBALL"})
        rows.append({**slow, "group": "BREAKING"})
    df = pd.DataFrame(rows)
    m = GradientBoostingClassifier(random_state=0)
    m.fit(df[FEATURE_COLUMNS], df["group"])
    return m


def test_returns_a_group_and_probabilities_for_a_valid_trajectory(model):
    verdict = classify_framed_trajectory(model, VALID_FRAMED)

    assert verdict.ok
    assert verdict.group in TWO_CLASSES
    assert set(verdict.probabilities) == set(TWO_CLASSES)
    assert verdict.probabilities[verdict.group] == pytest.approx(
        max(verdict.probabilities.values())
    )
    assert sum(verdict.probabilities.values()) == pytest.approx(1.0)


def test_reports_failure_when_no_trajectory_was_found(model):
    verdict = classify_framed_trajectory(model, [])

    assert not verdict.ok
    assert verdict.group is None
    assert verdict.probabilities == {}
    assert verdict.reason == "no_trajectory"


def test_reports_failure_when_too_few_points(model):
    """compute_trajectory_features가 min_points 미만에서 None을 준다."""
    verdict = classify_framed_trajectory(model, VALID_FRAMED[:2])

    assert not verdict.ok
    assert verdict.reason == "too_few_points"
    assert verdict.n_points == 2


def test_keeps_the_point_count_for_display(model):
    verdict = classify_framed_trajectory(model, VALID_FRAMED)

    assert verdict.n_points == len(VALID_FRAMED)


def test_verdict_from_chain_pairs_coordinates_with_box_sizes(model):
    """
    사슬은 (frame, x, y)만 담고 박스 크기는 감지 후보 쪽에 있다. 둘을 좌표로
    다시 맞추지 못하면 박스 특징이 조용히 0이 되어 학습 분포를 벗어난다.
    """
    chain = [(f, x, y) for f, x, y, _s in VALID_FRAMED]
    candidates = [
        (f, [(x, y, 0.5, s, s)]) for f, x, y, s in VALID_FRAMED
    ]

    verdict = verdict_from_chain(model, chain, candidates)

    assert verdict.ok
    assert verdict.n_points == len(VALID_FRAMED)


def test_failure_verdict_is_falsy_so_callers_cannot_use_it_by_accident(model):
    assert not PitchVerdict.failed("no_trajectory", 0)
    assert classify_framed_trajectory(model, VALID_FRAMED)


# ── 넓은 창 탐색 후 프레임 재색인 (TS-025) ──

def test_align_puts_the_chain_end_at_the_training_median():
    """넓은 창에서는 프레임 인덱스가 수백까지 간다. 학습이 본 위치로 되돌린다."""
    chain = [(300, 10.0, 10.0), (304, 20.0, 30.0), (308, 30.0, 60.0)]

    aligned = align_chain_frames(chain)

    assert aligned[-1][0] == TRAINING_END_FRAME


def test_align_preserves_gaps_between_frames():
    """평행이동이라 간격이 보존돼야 한다 — 안 그러면 속도 특징이 바뀐다."""
    chain = [(300, 10.0, 10.0), (304, 20.0, 30.0), (308, 30.0, 60.0)]

    aligned = align_chain_frames(chain)

    assert [f for f, _x, _y in aligned] == [
        TRAINING_END_FRAME - 8, TRAINING_END_FRAME - 4, TRAINING_END_FRAME
    ]


def test_align_leaves_coordinates_untouched():
    chain = [(300, 10.0, 10.0), (304, 20.0, 30.0), (308, 30.0, 60.0)]

    aligned = align_chain_frames(chain)

    assert [(x, y) for _f, x, y in aligned] == [(10.0, 10.0), (20.0, 30.0), (30.0, 60.0)]


def test_align_handles_an_empty_chain():
    assert align_chain_frames([]) == []


def test_shift_invariant_features_are_unchanged_by_alignment(model):
    """
    재색인이 바꿔도 되는 것은 end_frame 하나뿐이다. 나머지가 흔들리면 앱과 학습의
    특징 분포가 어긋나고, 그건 화면에서 안 보인다.
    """
    from pitch_type_cv.trajectory_features import compute_trajectory_features

    far = [(300, 10.0, 10.0, 5.0), (304, 20.0, 30.0, 6.0), (308, 30.0, 60.0, 7.0)]
    near = [(f - 287, x, y, s) for f, x, y, s in far]  # 같은 궤적, 원점만 다름

    a = compute_trajectory_features(
        [(x, y) for _f, x, y, _s in far],
        frame_indices=[f for f, _x, _y, _s in far],
        box_sizes=[s for _f, _x, _y, s in far],
    )
    b = compute_trajectory_features(
        [(x, y) for _f, x, y, _s in near],
        frame_indices=[f for f, _x, _y, _s in near],
        box_sizes=[s for _f, _x, _y, s in near],
    )

    differing = {k for k in a if a[k] != pytest.approx(b[k], abs=1e-9)}
    assert differing == {"end_frame"}
