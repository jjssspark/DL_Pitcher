"""Statcast pitch_type 코드를 3그룹(FASTBALL/BREAKING/OFFSPEED)으로 매핑한다."""

PITCH_GROUP_MAP: dict[str, str] = {
    "FF": "FASTBALL", "SI": "FASTBALL", "FC": "FASTBALL",
    "SL": "BREAKING", "CU": "BREAKING",
    "CH": "OFFSPEED", "FS": "OFFSPEED",
}


def pitch_type_to_group(pitch_type: str) -> str | None:
    """Statcast pitch_type을 3그룹으로 매핑. 매핑되지 않는 코드(KN 등)는 None."""
    return PITCH_GROUP_MAP.get(pitch_type)
