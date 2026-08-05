"""OCR이 읽은 중계 오버레이 구종을 라벨로 삼아 학습용 데이터셋을 조립한다."""
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from pitch_type_cv.pitch_group_map import ocr_pitch_name_to_group, pitch_type_to_group
from pitch_type_cv.trajectory_features import compute_trajectory_features

logger = logging.getLogger(__name__)

# OCR 라벨 비율이 Statcast 실제 비율과 이만큼 넘게 벌어지면 경고한다.
# 중단시키지는 않는다 — 검증된 임계값이 아니고, 라벨 정확성이 이 값에 걸려 있지도 않다.
GROUP_SHARE_WARN_DIVERGENCE = 0.25


def _group_shares(groups: list[str | None]) -> dict[str, float]:
    """None(판독 실패·미매핑)을 뺀 그룹별 비율."""
    valid = [g for g in groups if g is not None]
    if not valid:
        return {}
    counts = Counter(valid)
    return {g: c / len(valid) for g, c in counts.items()}


def _log_group_distribution_check(
    game_pk: int, ocr_groups: list[str | None], statcast_groups: list[str | None]
) -> None:
    """
    OCR 라벨 분포를 Statcast 실제 분포와 대조해 기록한다.

    투구 단위 페어링이 아니라 **분포 단위** 비교라 순서 밀림에 영향받지 않는다.
    OCR이 특정 구종을 체계적으로 오독하는 경우(예: 패스트볼을 브레이킹볼로)를 잡는 게 목적이다.
    """
    ocr_shares = _group_shares(ocr_groups)
    statcast_shares = _group_shares(statcast_groups)
    if not ocr_shares or not statcast_shares:
        return

    logger.info(
        "game_pk=%s 그룹 비율 — OCR %s / Statcast %s",
        game_pk,
        {g: f"{s:.0%}" for g, s in sorted(ocr_shares.items())},
        {g: f"{s:.0%}" for g, s in sorted(statcast_shares.items())},
    )

    worst_group, worst = max(
        (
            (g, abs(ocr_shares.get(g, 0.0) - statcast_shares.get(g, 0.0)))
            for g in set(ocr_shares) | set(statcast_shares)
        ),
        key=lambda pair: pair[1],
    )
    if worst > GROUP_SHARE_WARN_DIVERGENCE:
        logger.warning(
            "game_pk=%s %s 비율이 Statcast와 %.0f%%p 차이납니다. "
            "OCR 구종 판독이 편향됐을 수 있습니다.",
            game_pk, worst_group, worst * 100,
        )


def build_dataset_for_game(
    game_pk: int,
    youtube_url: str,
    fetch_statcast: Callable[[int], pd.DataFrame],
    resolve_video: Callable[[str], str],
    scan_overlays: Callable[[str], tuple[list[float], list[dict]]],
    extract_trajectory: Callable[[str, float], list[tuple[float, float]]],
) -> pd.DataFrame:
    """
    한 경기의 영상에서 (궤적 특징, 구종 그룹) 라벨링된 데이터셋 행을 조립한다.

    라벨은 **OCR이 읽은 중계 오버레이 구종**을 그대로 쓴다. 타임스탬프와 구종이 같은
    오버레이에서 함께 나오므로 순서 밀림이 원천적으로 없다. 이전에는 Statcast를 정답으로
    삼고 i번째 OCR ↔ i번째 Statcast로 붙였는데, OCR이 투구를 하나 놓칠 때마다 그 뒤가
    전부 한 칸씩 밀려 실측 일치율이 무작위 수준(34%)까지 떨어졌다.

    Statcast는 라벨이 아니라 그룹 비율 대조(진단)용으로만 쓴다.
    구종을 못 읽었거나 매핑 안 되는 샘플, 궤적 포인트가 부족한 샘플은 제외한다.
    """
    video_path = resolve_video(youtube_url)
    timestamps, pitch_data = scan_overlays(video_path)

    if len(timestamps) != len(pitch_data):
        logger.warning(
            "game_pk=%s 타임스탐프(%d개)와 OCR 구종(%d개) 개수가 달라 짧은 쪽까지만 씁니다.",
            game_pk, len(timestamps), len(pitch_data),
        )
    ocr_groups = [ocr_pitch_name_to_group(entry.get("pitch_type")) for entry in pitch_data]

    statcast_df = fetch_statcast(game_pk)
    _log_group_distribution_check(
        game_pk,
        ocr_groups,
        [pitch_type_to_group(p) for p in statcast_df["pitch_type"].tolist()],
    )

    n_detected = len(timestamps)
    n_excluded_unreadable = 0
    n_excluded_short_trajectory = 0

    rows = []
    for timestamp_sec, group in zip(timestamps, ocr_groups):
        if group is None:
            n_excluded_unreadable += 1
            continue

        trajectory = extract_trajectory(video_path, timestamp_sec)
        features = compute_trajectory_features(trajectory)
        if features is None:
            n_excluded_short_trajectory += 1
            continue

        rows.append({
            **features,
            "group": group,
            "game_pk": game_pk,
            "timestamp_sec": timestamp_sec,
        })

    logger.info(
        "game_pk=%s 감지 %d개 → 구종판독실패 %d개, 궤적부족 %d개, 최종 %d개 보존",
        game_pk, n_detected, n_excluded_unreadable, n_excluded_short_trajectory, len(rows),
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
