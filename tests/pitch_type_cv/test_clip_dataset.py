import math

from pitch_type_cv.clip_dataset import build_clip_dataset
from pitch_type_cv.savant_clips import PitchClip

# 3점 이상이고 실제로 이동하는 궤적 — compute_trajectory_features가 특징을 돌려준다.
# 추출기는 (frame_idx, x, y)를 돌려준다. 좌표만으로는 경과 프레임을 알 수 없기 때문이다.
VALID_TRAJECTORY = [
    (20, 100.0, 100.0), (21, 110.0, 120.0), (23, 120.0, 145.0), (24, 130.0, 175.0),
]


def _clip(play_id: str = "p1", pitch_type: str = "FF", game_pk: int = 1) -> PitchClip:
    return PitchClip(play_id=play_id, pitch_type=pitch_type, game_pk=game_pk)


def test_keeps_pitch_with_valid_trajectory():
    df = build_clip_dataset([_clip()], lambda clip: VALID_TRAJECTORY)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["has_trajectory"]
    assert row["group"] == "FASTBALL"
    assert row["play_id"] == "p1"
    assert row["pitch_type"] == "FF"
    assert row["game_pk"] == 1
    assert row["duration_frames"] == 4


def test_keeps_row_for_pitch_without_trajectory():
    """
    궤적을 못 잡은 투구도 행으로 남긴다. 선택 편향을 재려면 분모가 '궤적이 잡힌 투구'가
    아니라 '전체 투구'여야 한다 — 빠른 공일수록 사슬이 끊기므로 버리면 편향이 감춰진다.
    """
    df = build_clip_dataset([_clip()], lambda clip: [(5, 1.0, 1.0)])
    assert len(df) == 1
    assert not df.iloc[0]["has_trajectory"]
    assert math.isnan(df.iloc[0]["curvature_ratio"])
    assert df.iloc[0]["group"] == "FASTBALL"


def test_passes_frame_indices_through_to_features():
    """
    추출기가 준 frame_idx가 특징 계산까지 도달해야 한다. 중간에서 좌표만 넘기면
    frame_span이 조용히 '점 개수 - 1'로 퇴화하고, 감지가 빠진 프레임이 사라진다.
    """
    df = build_clip_dataset([_clip()], lambda clip: VALID_TRAJECTORY)
    row = df.iloc[0]

    assert row["frame_span"] == 4      # 20 -> 24 (프레임 22 누락)
    assert row["end_frame"] == 24
    assert row["duration_frames"] == 4


def test_skips_pitch_type_outside_three_groups():
    """KN(너클볼) 등 3그룹에 속하지 않는 구종은 분류 문제의 대상이 아니다."""
    df = build_clip_dataset([_clip(pitch_type="KN")], lambda clip: VALID_TRAJECTORY)
    assert df.empty


def test_maps_knuckle_curve_to_breaking():
    """KC 매핑 누락으로 너클커브가 통째로 빠지던 회귀를 조립 단계에서도 막는다."""
    df = build_clip_dataset([_clip(pitch_type="KC")], lambda clip: VALID_TRAJECTORY)
    assert df.iloc[0]["group"] == "BREAKING"


def test_preserves_game_pk_for_holdout_split():
    """game_pk 단위 홀드아웃이 목적이므로 경기 식별자가 행마다 살아 있어야 한다."""
    clips = [_clip("p1", game_pk=10), _clip("p2", game_pk=20)]
    df = build_clip_dataset(clips, lambda clip: VALID_TRAJECTORY)
    assert sorted(df["game_pk"].tolist()) == [10, 20]


def test_returns_empty_frame_for_no_clips():
    assert build_clip_dataset([], lambda clip: VALID_TRAJECTORY).empty


def test_continues_after_extraction_failure():
    """한 투구의 영상 처리가 실패해도 나머지 경기를 통째로 잃지 않아야 한다."""
    def flaky(clip: PitchClip):
        if clip.play_id == "bad":
            raise RuntimeError("디코딩 실패")
        return VALID_TRAJECTORY

    df = build_clip_dataset([_clip("bad"), _clip("good")], flaky).set_index("play_id")
    assert sorted(df.index) == ["bad", "good"]
    assert not df.loc["bad", "has_trajectory"]
    assert df.loc["good", "has_trajectory"]
