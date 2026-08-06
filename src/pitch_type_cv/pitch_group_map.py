"""Statcast pitch_type 코드를 3그룹(FASTBALL/BREAKING/OFFSPEED)으로 매핑한다."""

PITCH_GROUP_MAP: dict[str, str] = {
    "FF": "FASTBALL", "SI": "FASTBALL", "FC": "FASTBALL",
    # ST(스위퍼)는 2023년 Statcast에 신설된 코드다. 2024 WS G1 한 경기에만 25구(9%) 나와,
    # 빠뜨리면 실제 브레이킹볼이 통째로 학습셋에서 제외된다.
    # KC(너클커브)와 FO(포크볼)는 OCR 맵에는 있었는데 여기엔 없었다. OCR 라벨을 쓸 때는
    # 드러나지 않았지만 라벨 소스를 StatsAPI로 바꾸면 조용히 전량 탈락한다 —
    # 후보 7경기 실측 KC 93구, FO 16구 (2026-08-06 조사).
    "SL": "BREAKING", "CU": "BREAKING", "ST": "BREAKING", "KC": "BREAKING",
    "CH": "OFFSPEED", "FS": "OFFSPEED", "FO": "OFFSPEED",
}


def pitch_type_to_group(pitch_type: str) -> str | None:
    """Statcast pitch_type을 3그룹으로 매핑. 매핑되지 않는 코드(KN 등)는 None."""
    return PITCH_GROUP_MAP.get(pitch_type)


OCR_PITCH_NAME_GROUP_MAP: dict[str, str] = {
    # pose_detector.PITCH_MAP은 오버레이 텍스트 "FASTBALL"을 일반명 "Fastball"로 바꾼다.
    # 3그룹에서는 FF/SI/FC가 모두 FASTBALL이라 구질 세부를 몰라도 그룹은 확정된다.
    "Fastball": "FASTBALL",
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
