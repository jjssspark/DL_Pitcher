import logging

from pitch_type_cv.dataset import pair_timestamps_with_statcast


def test_pairs_equal_length_lists_fully():
    result = pair_timestamps_with_statcast([1.0, 2.0, 3.0], ["FF", "SL", "CH"])
    assert result == [(1.0, "FF"), (2.0, "SL"), (3.0, "CH")]


def test_truncates_to_shorter_list_when_counts_mismatch():
    result = pair_timestamps_with_statcast([1.0, 2.0, 3.0, 4.0], ["FF", "SL"])
    assert result == [(1.0, "FF"), (2.0, "SL")]


def test_logs_warning_on_count_mismatch(caplog):
    with caplog.at_level(logging.WARNING):
        pair_timestamps_with_statcast([1.0, 2.0, 3.0], ["FF"])

    assert any("3" in r.message and "1" in r.message for r in caplog.records)
