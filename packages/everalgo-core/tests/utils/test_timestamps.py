"""Tests for ``everalgo.utils.timestamps``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

import everalgo.utils
from everalgo.utils import (
    format_utc_date,
    format_utc_time,
    parse_utc_date,
    timestamps,
)

_WEDNESDAY_14_00 = 1_712_757_600_000  # 2024-04-10T14:00:00Z
_TUESDAY_22_13_20 = 1_700_000_000_000  # 2023-11-14T22:13:20Z
_TUESDAY_03_00 = 1_699_930_800_000  # 2023-11-14T03:00:00Z
_TUESDAY_12_00 = 1_699_963_200_000  # 2023-11-14T12:00:00Z
_MONDAY_10_00 = 1_699_869_600_000  # 2023-11-13T10:00:00Z
_ONE_MINUTE_MS = 60_000


def test_utc_time_is_minute_precision_with_a_utc_label() -> None:
    assert format_utc_time(_WEDNESDAY_14_00) == "2024-04-10 14:00 UTC"


def test_utc_time_truncates_seconds_rather_than_rounding() -> None:
    assert format_utc_time(_WEDNESDAY_14_00 + _ONE_MINUTE_MS - 1) == "2024-04-10 14:00 UTC"
    assert format_utc_time(_WEDNESDAY_14_00 + _ONE_MINUTE_MS) == "2024-04-10 14:01 UTC"


def test_utc_time_uses_a_24_hour_clock() -> None:
    assert format_utc_time(_TUESDAY_22_13_20) == "2023-11-14 22:13 UTC"
    assert format_utc_time(_TUESDAY_03_00) == "2023-11-14 03:00 UTC"
    assert format_utc_time(_TUESDAY_12_00) == "2023-11-14 12:00 UTC"


def test_utc_time_appends_the_utc_weekday_on_request() -> None:
    assert format_utc_time(_WEDNESDAY_14_00, weekday=True) == "2024-04-10 14:00 UTC (Wednesday)"
    assert format_utc_time(_MONDAY_10_00, weekday=True) == "2023-11-13 10:00 UTC (Monday)"


def test_utc_time_weekday_follows_utc_not_local_midnight() -> None:
    late_tuesday_utc = _TUESDAY_22_13_20 + 90 * _ONE_MINUTE_MS  # 23:43 UTC, already Wednesday in UTC+8
    assert format_utc_time(late_tuesday_utc, weekday=True) == "2023-11-14 23:43 UTC (Tuesday)"


def test_utc_date_from_milliseconds() -> None:
    assert format_utc_date(_WEDNESDAY_14_00) == "2024-04-10"


def test_utc_date_from_an_aware_datetime_converts_to_utc_first() -> None:
    east_of_utc = timezone(timedelta(hours=8))
    assert format_utc_date(datetime(2024, 4, 11, 1, 0, tzinfo=east_of_utc)) == "2024-04-10"


def test_utc_date_rejects_a_naive_datetime() -> None:
    with pytest.raises(ValueError, match="naive"):
        format_utc_date(datetime(2024, 4, 10))


def test_parse_utc_date_returns_aware_utc_midnight() -> None:
    assert parse_utc_date("2024-04-10") == datetime(2024, 4, 10, tzinfo=UTC)


def test_parse_utc_date_round_trips_with_format_utc_date() -> None:
    parsed = parse_utc_date("2024-02-28")
    assert parsed is not None
    assert format_utc_date(parsed + timedelta(days=1)) == "2024-02-29"


@pytest.mark.parametrize(
    "text",
    [
        "2024-02-30",  # not a real date
        "2024-4-1",  # not zero-padded
        "2024-04-10T00:00:00Z",  # carries a time
        "2024-04-10 ",  # trailing whitespace
        "2024-04-10\n",  # a trailing newline that ``$`` would let through
        "٢٠٢٤-٠٤-١٠",  # non-ASCII digits  # noqa: RUF001
        "",
    ],
)
def test_parse_utc_date_rejects_anything_but_a_real_zero_padded_date(text: str) -> None:
    assert parse_utc_date(text) is None


def test_the_seconds_precision_message_timestamp_is_gone() -> None:
    """``format_message_timestamp`` (``YYYY-MM-DD HH:MM:SS``, no zone label) is removed with its last caller."""
    assert not hasattr(everalgo.utils, "format_message_timestamp")
    assert not hasattr(timestamps, "format_message_timestamp")


def test_the_atomic_fact_label_is_gone() -> None:
    """The transitional 12-hour label is removed outright, leaving ``format_utc_time`` the one shape for an instant."""
    assert not hasattr(everalgo.utils, "format_atomic_fact_time")
    assert not hasattr(timestamps, "format_atomic_fact_time")
