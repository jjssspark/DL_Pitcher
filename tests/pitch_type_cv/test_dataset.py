import logging

import pandas as pd

from pitch_type_cv.dataset import GameSpec, build_dataset, build_dataset_for_game

TRAJ = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]


def _build(statcast_types, timestamps, pitch_data, trajectories=None, game_pk=775300):
    """테스트 공통 호출 — 궤적은 지정 없으면 전부 유효한 3점짜리를 준다."""
    return build_dataset_for_game(
        game_pk=game_pk,
        youtube_url="https://youtu.be/fake",
        fetch_statcast=lambda pk: pd.DataFrame({"pitch_type": statcast_types}),
        resolve_video=lambda url: "fake_video.mp4",
        scan_overlays=lambda video_path: (timestamps, pitch_data),
        extract_trajectory=(
            (lambda video_path, ts: trajectories[ts]) if trajectories
            else (lambda video_path, ts: TRAJ)
        ),
    )


def test_labels_come_from_ocr_reading_not_statcast():
    """
    라벨은 OCR이 읽은 오버레이 구종이다. Statcast 순서와 어긋나도 OCR을 따라야 한다 —
    순서 페어링을 없앤 것이 이 설계의 핵심이다.
    """
    df = _build(
        statcast_types=["FF", "FF", "FF"],          # Statcast는 전부 패스트볼
        timestamps=[1.0, 2.0, 3.0],
        pitch_data=[
            {"pitch_type": "Slider", "speed": 85},   # OCR은 브레이킹으로 읽음
            {"pitch_type": "Fastball", "speed": 96},
            {"pitch_type": "Changeup", "speed": 83},
        ],
    )

    assert list(df["group"]) == ["BREAKING", "FASTBALL", "OFFSPEED"]


def test_keeps_all_detections_when_statcast_count_differs():
    """OCR 감지 수가 Statcast 투구 수보다 적어도 앞에서부터 잘라내지 않는다."""
    df = _build(
        statcast_types=["FF"] * 10,
        timestamps=[1.0, 2.0],
        pitch_data=[
            {"pitch_type": "Fastball", "speed": 96},
            {"pitch_type": "Fastball", "speed": 95},
        ],
    )

    assert len(df) == 2


def test_excludes_unreadable_pitch_names():
    df = _build(
        statcast_types=["FF", "FF"],
        timestamps=[1.0, 2.0],
        pitch_data=[
            {"pitch_type": "Fastball", "speed": 96},
            {"pitch_type": None, "speed": None},      # OCR 판독 실패
        ],
    )

    assert len(df) == 1
    assert df.iloc[0]["group"] == "FASTBALL"


def test_excludes_short_trajectories():
    df = _build(
        statcast_types=["FF", "SL"],
        timestamps=[1.0, 2.0],
        pitch_data=[
            {"pitch_type": "Fastball", "speed": 96},
            {"pitch_type": "Slider", "speed": 85},
        ],
        trajectories={1.0: TRAJ, 2.0: [(0.0, 0.0)]},  # 두 번째는 포인트 부족
    )

    assert len(df) == 1
    assert df.iloc[0]["group"] == "FASTBALL"


def test_produces_expected_columns_and_game_pk():
    df = _build(
        statcast_types=["FF"],
        timestamps=[1.0],
        pitch_data=[{"pitch_type": "Fastball", "speed": 96}],
    )

    assert (df["game_pk"] == 775300).all()
    assert df.iloc[0]["timestamp_sec"] == 1.0
    assert {"group", "game_pk", "timestamp_sec", "curvature_ratio"} <= set(df.columns)


def test_warns_when_group_share_diverges_from_statcast(caplog):
    """OCR이 브레이킹볼만 읽는데 실제는 전부 패스트볼이면 판독 편향을 경고해야 한다."""
    with caplog.at_level(logging.INFO):
        df = _build(
            statcast_types=["FF"] * 10,
            timestamps=[1.0, 2.0, 3.0],
            pitch_data=[{"pitch_type": "Slider", "speed": 85}] * 3,
        )

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("편향" in r.message for r in warnings)
    # 경고일 뿐 중단은 아니다 — 라벨 정확성이 이 지표에 걸려 있지 않다
    assert len(df) == 3


def test_no_warning_when_group_share_matches_statcast(caplog):
    with caplog.at_level(logging.INFO):
        _build(
            statcast_types=["FF", "FF", "SL", "SL"],
            timestamps=[1.0, 2.0, 3.0, 4.0],
            pitch_data=[
                {"pitch_type": "Fastball", "speed": 96},
                {"pitch_type": "Fastball", "speed": 95},
                {"pitch_type": "Slider", "speed": 85},
                {"pitch_type": "Slider", "speed": 84},
            ],
        )

    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_logs_exclusion_diagnostics(caplog):
    with caplog.at_level(logging.INFO):
        _build(
            statcast_types=["FF", "FF", "FF"],
            timestamps=[1.0, 2.0, 3.0],
            pitch_data=[
                {"pitch_type": "Fastball", "speed": 96},   # 보존
                {"pitch_type": None, "speed": None},       # 판독 실패
                {"pitch_type": "Fastball", "speed": 94},   # 궤적 부족
            ],
            trajectories={1.0: TRAJ, 2.0: TRAJ, 3.0: [(0.0, 0.0)]},
        )

    summaries = [r for r in caplog.records if "최종" in r.message]
    assert len(summaries) == 1
    assert "감지 3개" in summaries[0].message


def test_build_dataset_skips_failing_game_and_keeps_others(caplog):
    def fetch_statcast(game_pk: int) -> pd.DataFrame:
        if game_pk == 111:
            raise RuntimeError("Statcast 조회 실패")
        return pd.DataFrame({"pitch_type": ["FF"]})

    with caplog.at_level(logging.WARNING):
        df = build_dataset(
            games=[GameSpec(game_pk=111, youtube_url="https://youtu.be/bad"),
                   GameSpec(game_pk=222, youtube_url="https://youtu.be/good")],
            fetch_statcast=fetch_statcast,
            resolve_video=lambda url: "fake_video.mp4",
            scan_overlays=lambda video_path: (
                [1.0], [{"pitch_type": "Fastball", "speed": 96}]
            ),
            extract_trajectory=lambda video_path, ts: TRAJ,
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
