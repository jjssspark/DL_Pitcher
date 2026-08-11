"""영상 시각 -> 투구 인덱스 매핑 검증.

이 매핑이 틀리면 화면의 구종·구속·예측이 통째로 다른 투구를 가리킨다.
가장 중요한 성질은 "앵커가 없으면 기존 균등 분할과 완전히 같다"는 것 —
OCR 판독이 실패한 경기에서도 앱이 지금처럼 동작해야 한다.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from timeline_anchor import (  # noqa: E402
    index_at_time,
    pitcher_pitch_counts,
    resolve_anchors,
    time_at_index,
)

DURATION = 8230.7


def make_pitches(pitcher_ids: list[int]) -> list[dict]:
    return [{"pitcher_id": pid} for pid in pitcher_ids]


def test_counts_restart_when_pitcher_changes():
    pitches = make_pitches([1, 1, 1, 2, 2, 1])
    assert pitcher_pitch_counts(pitches) == [1, 2, 3, 1, 2, 4]


def test_without_anchors_matches_uniform_division():
    n = 320
    for t in (0.0, 25.7, 100.0, 4000.0, DURATION):
        expected = max(0, min(int(t / DURATION * n), n - 1))
        assert index_at_time(t, [], DURATION, n) == expected


def test_interpolates_between_anchors():
    # 100초에 10번, 200초에 30번 -> 150초면 정확히 중간인 20번
    anchors = [(100.0, 10), (200.0, 30)]
    assert index_at_time(150.0, anchors, DURATION, 320) == 20


def test_anchor_times_return_their_own_index():
    anchors = [(100.0, 10), (200.0, 30)]
    assert index_at_time(100.0, anchors, DURATION, 320) == 10
    assert index_at_time(200.0, anchors, DURATION, 320) == 30


def test_index_never_leaves_valid_range():
    anchors = [(100.0, 10)]
    assert index_at_time(-50.0, anchors, DURATION, 320) == 0
    assert index_at_time(DURATION * 2, anchors, DURATION, 320) == 319


def test_resolve_picks_candidate_nearest_to_uniform_guess():
    # 투수 1이 3구, 투수 2가 3구. P:2 는 인덱스 1과 4 둘 다 해당한다.
    pitches = make_pitches([1, 1, 1, 2, 2, 2])
    # 영상 후반부 관측이면 뒤쪽 후보(4)를 골라야 한다
    anchors = resolve_anchors([(DURATION * 0.75, 2)], pitches, DURATION)
    assert anchors == [(DURATION * 0.75, 4)]


def test_resolve_drops_readings_that_go_backwards():
    # OCR 오독은 실측에서 전부 감소 방향이었다 (16 -> 3, 16 -> 2)
    pitches = make_pitches([1] * 40)
    counters = [(1000.0, 10), (2000.0, 3), (3000.0, 20)]
    anchors = resolve_anchors(counters, pitches, DURATION)
    assert [idx for _t, idx in anchors] == [9, 19]      # P:N 은 1부터, 인덱스는 0부터


def test_resolve_keeps_readings_that_follow_a_misread():
    # 오독 하나가 뒤따르는 정상 관측을 막으면 안 된다. 앞에서부터 그리디로 고르면
    # 12 를 채택한 뒤 6, 7, 8 이 전부 버려진다 (실측에서 282개 중 129개가 이렇게
    # 탈락했다). 가장 긴 사슬은 5, 6, 7, 8 이다.
    pitches = make_pitches([1] * 40)
    counters = [(1000.0, 5), (2000.0, 12), (3000.0, 6), (4000.0, 7), (5000.0, 8)]
    anchors = resolve_anchors(counters, pitches, DURATION)
    assert [idx for _t, idx in anchors] == [4, 5, 6, 7]


def test_resolve_prefers_the_earlier_reading_when_chains_tie():
    # 길이가 같으면 앞선 관측을 쓴다. 오독은 실측에서 전부 감소 방향이었으므로
    # 뒤에 온 작은 값(3)이 아니라 먼저 온 값(10)을 남기는 쪽이 맞다.
    pitches = make_pitches([1] * 40)
    counters = [(1000.0, 10), (2000.0, 3), (3000.0, 20)]
    anchors = resolve_anchors(counters, pitches, DURATION)
    assert [idx for _t, idx in anchors] == [9, 19]


def test_resolve_ignores_unreadable_entries():
    pitches = make_pitches([1] * 40)
    anchors = resolve_anchors([(1000.0, None), (2000.0, 5)], pitches, DURATION)
    assert anchors == [(2000.0, 4)]


def test_resolve_ignores_counter_with_no_matching_pitch():
    pitches = make_pitches([1] * 5)          # 최대 P:5
    assert resolve_anchors([(1000.0, 99)], pitches, DURATION) == []


@pytest.mark.parametrize("bad_duration", [0.0, -1.0])
def test_degenerate_duration_does_not_crash(bad_duration):
    assert index_at_time(10.0, [], bad_duration, 320) == 0
    assert resolve_anchors([(1.0, 1)], make_pitches([1]), bad_duration) == []


def test_empty_pitch_list_yields_no_anchors():
    assert resolve_anchors([(1000.0, 5)], [], DURATION) == []


# ── 역함수: 이동할 때 영상도 같은 매핑으로 옮겨야 한다 ──────────────────

def test_time_at_index_without_anchors_matches_uniform():
    n = 320
    for idx in (0, 1, 100, 319):
        expected = (idx / n) * DURATION
        assert time_at_index(idx, [], DURATION, n) == pytest.approx(expected)


def test_round_trip_returns_the_same_index():
    anchors = [(100.0, 10), (200.0, 30), (400.0, 50)]
    for idx in (0, 10, 20, 30, 40, 50):
        t = time_at_index(idx, anchors, DURATION, 320)
        assert index_at_time(t, anchors, DURATION, 320) == idx


def test_time_at_anchor_index_returns_its_own_time():
    anchors = [(100.0, 10), (200.0, 30)]
    assert time_at_index(10, anchors, DURATION, 320) == pytest.approx(100.0)
    assert time_at_index(30, anchors, DURATION, 320) == pytest.approx(200.0)


def test_time_stays_inside_the_video():
    anchors = [(100.0, 10)]
    assert time_at_index(-5, anchors, DURATION, 320) >= 0.0
    assert time_at_index(9999, anchors, DURATION, 320) <= DURATION
