"""
투구별 Savant 클립에서 학습용 데이터셋을 조립한다.

OCR 경로(dataset.py)와 나뉘어 있는 이유:
  - 라벨을 StatsAPI에서 직접 받는다. OCR 판독률 85% 상한과 순서 밀림이 없다.
  - 영상 단위가 '경기'가 아니라 '투구'다. 타임스탬프 탐색이 필요 없고 고정 윈도우로 자른다.

궤적을 못 잡은 투구도 행으로 남긴다. 선택 편향의 분모가 '궤적이 잡힌 투구'가 되면
편향 자체가 측정에서 사라지기 때문이다 — 학습 시에는 has_trajectory로 거른다.
"""
import logging
from typing import Callable

import pandas as pd

from pitch_type_cv.pitch_group_map import pitch_type_to_group
from pitch_type_cv.savant_clips import PitchClip
from pitch_type_cv.trajectory_features import FEATURE_COLUMNS, compute_trajectory_features

logger = logging.getLogger(__name__)

_EMPTY_FEATURES = {column: float("nan") for column in FEATURE_COLUMNS}


def build_clip_dataset(
    clips: list[PitchClip],
    extract_trajectory: Callable[[PitchClip], list[tuple[float, float]]],
) -> pd.DataFrame:
    """
    투구 목록에서 (궤적 특징, 구종 그룹) 데이터셋을 조립한다.

    3그룹에 매핑되지 않는 구종(KN 등)만 제외한다. 궤적 추출이 실패하거나 포인트가
    부족한 투구는 특징을 NaN으로 두고 has_trajectory=False로 남긴다.
    """
    rows = []
    for clip in clips:
        group = pitch_type_to_group(clip.pitch_type)
        if group is None:
            continue

        try:
            trajectory = extract_trajectory(clip)
        except Exception as exc:
            # 개별 클립의 다운로드·디코딩 실패로 경기 전체를 잃지 않는다.
            logger.warning("투구 %s 궤적 추출 실패: %s", clip.play_id, exc)
            trajectory = []

        features = compute_trajectory_features(trajectory)
        rows.append({
            **(features if features is not None else _EMPTY_FEATURES),
            "group": group,
            "game_pk": clip.game_pk,
            "play_id": clip.play_id,
            "pitch_type": clip.pitch_type,
            "has_trajectory": features is not None,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    kept = int(df["has_trajectory"].sum())
    logger.info("투구 %d개 중 궤적 확보 %d개 (%.1f%%)", len(df), kept, 100 * kept / len(df))
    return df
