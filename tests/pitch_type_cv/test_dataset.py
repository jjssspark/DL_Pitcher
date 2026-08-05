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


def test_build_dataset_for_game_high_ocr_agreement_returns_normal_rows(caplog):
    statcast_df = pd.DataFrame({"pitch_type": ["FF", "FF", "SL", "SL", "CH"]})
    trajectories = {
        1.0: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        2.0: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        3.0: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        4.0: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        5.0: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
    }
    pitch_data = [
        {"pitch_type": "4-Seam Fastball", "speed": 95},
        {"pitch_type": "4-Seam Fastball", "speed": 96},
        {"pitch_type": "Slider", "speed": 85},
        {"pitch_type": "Slider", "speed": 84},
        {"pitch_type": "Changeup", "speed": 83},
    ]

    with caplog.at_level(logging.INFO):
        df = build_dataset_for_game(
            game_pk=775300,
            youtube_url="https://youtu.be/fake",
            fetch_statcast=lambda game_pk: statcast_df,
            resolve_video=lambda url: "fake_video.mp4",
            scan_overlays=lambda video_path: ([1.0, 2.0, 3.0, 4.0, 5.0], pitch_data),
            extract_trajectory=lambda video_path, ts: trajectories[ts],
        )

    assert len(df) == 5
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_build_dataset_for_game_low_ocr_agreement_returns_empty_and_warns(caplog):
    statcast_df = pd.DataFrame({"pitch_type": ["FF", "FF", "FF", "FF", "FF"]})
    trajectories = {ts: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)] for ts in [1.0, 2.0, 3.0, 4.0, 5.0]}
    # Statcast는 전부 FASTBALL이지만 OCR은 대부분 BREAKING을 읽음 → 일치율 낮음(페어링 밀림 의심)
    pitch_data = [
        {"pitch_type": "Slider", "speed": 85},
        {"pitch_type": "Slider", "speed": 85},
        {"pitch_type": "Slider", "speed": 85},
        {"pitch_type": "Slider", "speed": 85},
        {"pitch_type": "4-Seam Fastball", "speed": 95},
    ]

    with caplog.at_level(logging.INFO):
        df = build_dataset_for_game(
            game_pk=775300,
            youtube_url="https://youtu.be/fake",
            fetch_statcast=lambda game_pk: statcast_df,
            resolve_video=lambda url: "fake_video.mp4",
            scan_overlays=lambda video_path: ([1.0, 2.0, 3.0, 4.0, 5.0], pitch_data),
            extract_trajectory=lambda video_path, ts: trajectories[ts],
        )

    assert df.empty
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("775300" in r.message for r in warnings)


def test_build_dataset_for_game_too_few_comparable_pairs_skips_validation():
    statcast_df = pd.DataFrame({"pitch_type": ["FF", "SL", "CH"]})
    trajectory = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    pitch_data = [
        {"pitch_type": None, "speed": None},
        {"pitch_type": None, "speed": None},
        {"pitch_type": None, "speed": None},
    ]

    df = build_dataset_for_game(
        game_pk=775300,
        youtube_url="https://youtu.be/fake",
        fetch_statcast=lambda game_pk: statcast_df,
        resolve_video=lambda url: "fake_video.mp4",
        scan_overlays=lambda video_path: ([1.0, 2.0, 3.0], pitch_data),
        extract_trajectory=lambda video_path, ts: trajectory,
    )

    assert len(df) == 3


def test_build_dataset_for_game_logs_exclusion_diagnostics(caplog):
    statcast_df = pd.DataFrame({"pitch_type": ["FF", "KN", "SL"]})  # KN은 매핑 안 됨
    trajectories = {
        1.0: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],  # 정상 보존
        2.0: [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],  # 매핑 제외 대상이라 궤적 미사용
        3.0: [(0.0, 0.0)],  # 궤적 포인트 부족으로 제외
    }

    with caplog.at_level(logging.INFO):
        df = build_dataset_for_game(
            game_pk=775300,
            youtube_url="https://youtu.be/fake",
            fetch_statcast=lambda game_pk: statcast_df,
            resolve_video=lambda url: "fake_video.mp4",
            scan_overlays=lambda video_path: ([1.0, 2.0, 3.0], []),
            extract_trajectory=lambda video_path, ts: trajectories[ts],
        )

    assert len(df) == 1

    summary_logs = [
        r for r in caplog.records
        if r.levelno >= logging.INFO and "페어링" in r.message
    ]
    assert len(summary_logs) == 1
    message = summary_logs[0].message
    assert "3" in message  # n_paired
    assert "1" in message  # 매핑제외/궤적부족제외/최종 보존 각각 1개


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
