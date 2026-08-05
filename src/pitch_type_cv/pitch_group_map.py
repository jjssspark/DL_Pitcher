"""Statcast pitch_type 코드를 3그룹(FASTBALL/BREAKING/OFFSPEED)으로 매핑한다."""

PITCH_GROUP_MAP: dict[str, str] = {
    "FF": "FASTBALL", "SI": "FASTBALL", "FC": "FASTBALL",
    "SL": "BREAKING", "CU": "BREAKING",
    "CH": "OFFSPEED", "FS": "OFFSPEED",
}


def pitch_type_to_group(pitch_type: str) -> str | None:
    """Statcast pitch_type을 3그룹으로 매핑. 매핑되지 않는 코드(KN 등)는 None."""
    return PITCH_GROUP_MAP.get(pitch_type)


OCR_PITCH_NAME_GROUP_MAP: dict[str, str] = {
    "4-Seam Fastball": "FASTBALL",
    "2-Seam Fastball": "FASTBALL",
    "Sinker": "FASTBALL",
    "Cutter": "FASTBALL",
    "Slider": "BREAKING",
    "Sweeper": "BREAKING",
    "Curveball": "BREAKING",
    "Knuckle Curve": "BREAKING",
    "Changeup": "OFFSPEED",
    "Splitter": "OFFSPEED",
}


def ocr_pitch_name_to_group(pitch_name: str | None) -> str | None:
    """OCR로 읽은 구종 이름(pose_detector.PITCH_MAP의 표시명)을 3그룹으로 매핑. None이거나 매핑 안 되면 None."""
    if pitch_name is None:
        return None
    return OCR_PITCH_NAME_GROUP_MAP.get(pitch_name)
