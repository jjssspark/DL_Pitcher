from pitch_type_cv.pitch_group_map import pitch_type_to_group, ocr_pitch_name_to_group


def test_maps_fastball_family():
    assert pitch_type_to_group("FF") == "FASTBALL"
    assert pitch_type_to_group("SI") == "FASTBALL"
    assert pitch_type_to_group("FC") == "FASTBALL"


def test_maps_breaking_family():
    assert pitch_type_to_group("SL") == "BREAKING"
    assert pitch_type_to_group("CU") == "BREAKING"


def test_maps_offspeed_family():
    assert pitch_type_to_group("CH") == "OFFSPEED"
    assert pitch_type_to_group("FS") == "OFFSPEED"


def test_returns_none_for_unmapped_pitch_type():
    assert pitch_type_to_group("KN") is None
    assert pitch_type_to_group("") is None


def test_ocr_maps_fastball_family_name():
    assert ocr_pitch_name_to_group("4-Seam Fastball") == "FASTBALL"
    assert ocr_pitch_name_to_group("Sinker") == "FASTBALL"


def test_ocr_maps_breaking_family_name():
    assert ocr_pitch_name_to_group("Slider") == "BREAKING"
    assert ocr_pitch_name_to_group("Curveball") == "BREAKING"


def test_ocr_maps_offspeed_family_name():
    assert ocr_pitch_name_to_group("Changeup") == "OFFSPEED"
    assert ocr_pitch_name_to_group("Splitter") == "OFFSPEED"


def test_ocr_returns_none_for_none_input():
    assert ocr_pitch_name_to_group(None) is None


def test_ocr_returns_none_for_unrecognized_name():
    assert ocr_pitch_name_to_group("Knuckleball") is None


def test_ocr_maps_generic_fastball_name():
    """
    pose_detector.PITCH_MAP은 오버레이 텍스트 "FASTBALL"을 표시명 "Fastball"로 바꾼다.
    3그룹 분류에서는 FF/SI/FC가 모두 FASTBALL이라 일반명도 모호하지 않다.
    실측(2024 WS G1)에서 OCR 판독 237건 중 74건이 이 이름이었다.
    """
    assert ocr_pitch_name_to_group("Fastball") == "FASTBALL"


def test_statcast_maps_sweeper_to_breaking():
    """ST(스위퍼)는 브레이킹볼이다. 2024 WS G1 한 경기에만 25구(9%) 나온다."""
    assert pitch_type_to_group("ST") == "BREAKING"


def test_statcast_maps_knuckle_curve_to_breaking():
    """
    KC(너클커브)는 브레이킹볼이다. OCR 맵에는 "Knuckle Curve"가 있었지만 Statcast 맵에는
    빠져 있었다 — 라벨 소스를 StatsAPI로 바꾸면 조용히 전량 탈락한다.
    후보 7경기 실측 93구 (2026-08-06 조사).
    """
    assert pitch_type_to_group("KC") == "BREAKING"


def test_statcast_maps_forkball_to_offspeed():
    """FO(포크볼)는 스플리터 계열이라 OFFSPEED다. 후보 7경기 실측 16구."""
    assert pitch_type_to_group("FO") == "OFFSPEED"
