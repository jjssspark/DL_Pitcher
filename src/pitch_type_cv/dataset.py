"""OCR 타임스탐프와 Statcast 정답을 페어링해 학습용 데이터셋을 조립한다."""
import logging
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from pitch_type_cv.pitch_group_map import ocr_pitch_name_to_group, pitch_type_to_group
from pitch_type_cv.trajectory_features import compute_trajectory_features

logger = logging.getLogger(__name__)

MIN_COMPARABLE_PAIRS_FOR_VALIDATION = 5
MIN_OCR_STATCAST_AGREEMENT_RATE = 0.5


def pair_timestamps_with_statcast(
    timestamps: list[float], pitch_types: list[str]
) -> list[tuple[float, str]]:
    """
    i번째 OCR 타임스탐프를 i번째 Statcast pitch_type과 순서대로 페어링한다.
    개수가 다르면 짧은 쪽 길이까지만 앞에서부터 매칭하고 경고를 남긴다.
    """
    n = min(len(timestamps), len(pitch_types))
    if len(timestamps) != len(pitch_types):
        logger.warning(
            "OCR 타임스탐프(%d개)와 Statcast 투구 수(%d개)가 달라 앞에서부터 %d개만 매칭합니다.",
            len(timestamps), len(pitch_types), n,
        )
    return list(zip(timestamps[:n], pitch_types[:n]))


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

    timestamps, pitch_data = scan_overlays(video_path)
    pairs = pair_timestamps_with_statcast(timestamps, pitch_types)
    ocr_pitch_data = pitch_data[:len(pairs)]

    n_paired = len(pairs)
    n_excluded_unmapped = 0
    n_excluded_short_trajectory = 0
    n_kept = 0
    comparable_pairs = 0
    agreeing_pairs = 0

    rows = []
    for i, (timestamp_sec, pitch_type) in enumerate(pairs):
        group = pitch_type_to_group(pitch_type)

        # ocr_pitch_data가 pairs보다 짧아도(예: scan_overlays 구현체가 빈 리스트를 주는 경우)
        # 안전하게 "비교 불가"로 취급한다 — 인덱스 밖이면 빈 dict를 쓴다.
        ocr_entry = ocr_pitch_data[i] if i < len(ocr_pitch_data) else {}
        ocr_group = ocr_pitch_name_to_group(ocr_entry.get("pitch_type"))
        if ocr_group is not None:
            comparable_pairs += 1
            if ocr_group == group:
                agreeing_pairs += 1

        if group is None:
            n_excluded_unmapped += 1
            continue

        trajectory = extract_trajectory(video_path, timestamp_sec)
        features = compute_trajectory_features(trajectory)
        if features is None:
            n_excluded_short_trajectory += 1
            continue

        n_kept += 1
        rows.append({
            **features,
            "group": group,
            "game_pk": game_pk,
            "timestamp_sec": timestamp_sec,
        })

    if comparable_pairs >= MIN_COMPARABLE_PAIRS_FOR_VALIDATION:
        agreement_rate = agreeing_pairs / comparable_pairs
        logger.info(
            "game_pk=%s OCR-Statcast 구종 그룹 일치율: %.1f%% (%d/%d 비교 가능)",
            game_pk, agreement_rate * 100, agreeing_pairs, comparable_pairs,
        )
        if agreement_rate < MIN_OCR_STATCAST_AGREEMENT_RATE:
            logger.warning(
                "game_pk=%s OCR-Statcast 구종 그룹 일치율이 %.1f%%로 낮아 "
                "페어링 밀림이 의심됩니다. 이 경기 데이터를 제외합니다.",
                game_pk, agreement_rate * 100,
            )
            return pd.DataFrame()
    else:
        logger.info(
            "game_pk=%s 비교 가능한 OCR 샘플이 %d개로 부족해 일치율 검증을 건너뜁니다.",
            game_pk, comparable_pairs,
        )

    logger.info(
        "game_pk=%s 페어링 %d개 → 매핑제외 %d개, 궤적부족제외 %d개, 최종 %d개 보존",
        game_pk, n_paired, n_excluded_unmapped, n_excluded_short_trajectory, n_kept,
    )

    return pd.DataFrame(rows)


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
