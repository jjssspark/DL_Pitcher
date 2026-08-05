"""OCR 타임스탐프와 Statcast 정답을 페어링해 학습용 데이터셋을 조립한다."""
import logging

logger = logging.getLogger(__name__)


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
