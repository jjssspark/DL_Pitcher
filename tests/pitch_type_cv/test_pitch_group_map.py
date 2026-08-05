from pitch_type_cv.pitch_group_map import pitch_type_to_group


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
