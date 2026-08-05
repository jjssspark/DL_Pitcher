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
    assert ocr_pitch_name_to_group("Fastball") is None
