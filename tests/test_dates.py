from datetime import UTC, datetime, timedelta

import pytest

from app.hn_client import is_in_window, split_window, utc_lookback_window


def test_utc_window_is_half_open() -> None:
    now = datetime(2026, 9, 16, 18, 0, tzinfo=UTC)
    start, end = utc_lookback_window(now, 7)
    assert start == datetime(2026, 9, 9, 18, 0, tzinfo=UTC)
    assert end == now
    assert is_in_window(start, start, end)
    assert not is_in_window(end, start, end)
    just_before_end = end - timedelta(seconds=1)
    assert is_in_window(just_before_end, start, end)
    before_start = start - timedelta(seconds=1)
    assert not is_in_window(before_start, start, end)


def test_naive_now_is_rejected() -> None:
    naive = datetime(2026, 9, 16, 18, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        utc_lookback_window(naive, 7)


def test_non_positive_lookback_is_rejected() -> None:
    now = datetime(2026, 9, 16, 18, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="positive"):
        utc_lookback_window(now, 0)


def test_split_window_is_contiguous_and_half_open() -> None:
    start = datetime(2026, 9, 9, 18, 0, tzinfo=UTC)
    end = datetime(2026, 9, 12, 18, 0, tzinfo=UTC)
    buckets = split_window(start, end, 3)
    assert buckets[0][0] == start
    assert buckets[-1][1] == end
    for index, (bucket_start, bucket_end) in enumerate(buckets):
        assert is_in_window(bucket_start, start, end)
        assert not is_in_window(bucket_end, bucket_start, bucket_end)
        if index > 0:
            assert buckets[index - 1][1] == bucket_start
