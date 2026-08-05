import logging

import pandas as pd

from pitch_type_cv.dataset import pair_timestamps_with_statcast, build_dataset_for_game


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
