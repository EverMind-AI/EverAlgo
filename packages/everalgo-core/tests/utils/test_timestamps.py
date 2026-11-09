"""Tests for ``everalgo.utils.timestamps``."""

from __future__ import annotations

import locale
from datetime import UTC, datetime, timedelta, timezone
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pytest

import everalgo.utils
from everalgo.utils import (
    format_date,
    format_time,
    format_utc_date,
    format_utc_time,
    parse_utc_date,
    timestamps,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_WEDNESDAY_14_00 = 1_712_757_600_000  # 2024-04-10T14:00:00Z
_TUESDAY_22_13_20 = 1_700_000_000_000  # 2023-11-14T22:13:20Z
_TUESDAY_03_00 = 1_699_930_800_000  # 2023-11-14T03:00:00Z
_TUESDAY_12_00 = 1_699_963_200_000  # 2023-11-14T12:00:00Z
_MONDAY_10_00 = 1_699_869_600_000  # 2023-11-13T10:00:00Z
_ONE_MINUTE_MS = 60_000
_UTC_PLUS_8 = timezone(timedelta(hours=8))


def test_time_is_minute_precision_with_a_utc_label() -> None:
    assert format_time(_WEDNESDAY_14_00) == "2024-04-10 14:00 UTC"


def test_time_truncates_seconds_rather_than_rounding() -> None:
    assert format_time(_WEDNESDAY_14_00 + _ONE_MINUTE_MS - 1) == "2024-04-10 14:00 UTC"
    assert format_time(_WEDNESDAY_14_00 + _ONE_MINUTE_MS) == "2024-04-10 14:01 UTC"


def test_time_uses_a_24_hour_clock() -> None:
    assert format_time(_TUESDAY_22_13_20) == "2023-11-14 22:13 UTC"
    assert format_time(_TUESDAY_03_00) == "2023-11-14 03:00 UTC"
    assert format_time(_TUESDAY_12_00) == "2023-11-14 12:00 UTC"


def test_time_appends_the_weekday_on_request() -> None:
    assert format_time(_WEDNESDAY_14_00, weekday=True) == "2024-04-10 14:00 UTC (Wednesday)"
    assert format_time(_MONDAY_10_00, weekday=True) == "2023-11-13 10:00 UTC (Monday)"


def test_time_weekday_follows_utc_by_default_not_local_midnight() -> None:
    late_tuesday_utc = _TUESDAY_22_13_20 + 90 * _ONE_MINUTE_MS  # 23:43 UTC, already Wednesday in UTC+8
    assert format_time(late_tuesday_utc, weekday=True) == "2023-11-14 23:43 UTC (Tuesday)"


def test_time_in_another_zone_moves_the_date_clock_and_weekday_and_labels_the_offset() -> None:
    late_tuesday_utc = _TUESDAY_22_13_20 + 90 * _ONE_MINUTE_MS
    assert format_time(late_tuesday_utc, tz=_UTC_PLUS_8, weekday=True) == "2023-11-15 07:43 UTC+08:00 (Wednesday)"


@pytest.mark.parametrize(
    ("tz", "expected"),
    [
        (timezone(timedelta(hours=5, minutes=30)), "2024-04-10 19:30 UTC+05:30"),
        (timezone(timedelta(hours=-5)), "2024-04-10 09:00 UTC-05:00"),
        (timezone(timedelta(hours=-3, minutes=-30)), "2024-04-10 10:30 UTC-03:30"),
        (timezone(timedelta(0)), "2024-04-10 14:00 UTC"),
    ],
)
def test_time_labels_half_hour_negative_and_zero_offsets(tz: timezone, expected: str) -> None:
    assert format_time(_WEDNESDAY_14_00, tz=tz) == expected


def test_time_in_a_daylight_saving_zone_shows_the_offset_in_force_then() -> None:
    new_york = ZoneInfo("America/New_York")
    assert format_time(1_705_330_800_000, tz=new_york) == "2024-01-15 10:00 UTC-05:00"  # 15:00Z, winter
    assert format_time(1_721_055_600_000, tz=new_york) == "2024-07-15 11:00 UTC-04:00"  # 15:00Z, summer


@pytest.fixture
def chinese_time_locale() -> Iterator[None]:
    """Switch ``LC_TIME`` to a Chinese locale for the test, as a host calling ``locale.setlocale`` would."""
    saved = locale.setlocale(locale.LC_TIME)
    for name in ("zh_CN.UTF-8", "zh_CN.utf8", "zh_CN"):
        try:
            locale.setlocale(locale.LC_TIME, name)
            break
        except locale.Error:
            continue
    else:
        pytest.skip("no Chinese locale installed")
    try:
        yield
    finally:
        locale.setlocale(locale.LC_TIME, saved)


@pytest.mark.usefixtures("chinese_time_locale")
def test_the_weekday_stays_english_whatever_the_process_locale() -> None:
    assert datetime(2024, 4, 10, tzinfo=UTC).strftime("%A") != "Wednesday"  # the locale is really switched
    assert format_time(_WEDNESDAY_14_00, weekday=True) == "2024-04-10 14:00 UTC (Wednesday)"


def test_date_from_milliseconds() -> None:
    assert format_date(_WEDNESDAY_14_00) == "2024-04-10"


def test_date_from_milliseconds_in_another_zone_reads_its_calendar() -> None:
    late_tuesday_utc = _TUESDAY_22_13_20 + 90 * _ONE_MINUTE_MS
    assert format_date(late_tuesday_utc) == "2023-11-14"
    assert format_date(late_tuesday_utc, tz=_UTC_PLUS_8) == "2023-11-15"


def test_date_from_an_aware_datetime_converts_to_utc_first() -> None:
    assert format_date(datetime(2024, 4, 11, 1, 0, tzinfo=_UTC_PLUS_8)) == "2024-04-10"


def test_date_rejects_a_naive_datetime() -> None:
    with pytest.raises(ValueError, match="naive"):
        format_date(datetime(2024, 4, 10))


def test_parse_utc_date_returns_aware_utc_midnight() -> None:
    assert parse_utc_date("2024-04-10") == datetime(2024, 4, 10, tzinfo=UTC)


def test_parse_utc_date_round_trips_with_format_date() -> None:
    parsed = parse_utc_date("2024-02-28")
    assert parsed is not None
    assert format_date(parsed + timedelta(days=1)) == "2024-02-29"


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


def test_the_utc_named_aliases_still_render_as_before_but_warn() -> None:
    """Packages released against core 0.6 call these; any core below 2.0 must keep them working."""
    with pytest.warns(DeprecationWarning, match="use format_time"):
        assert format_utc_time(_WEDNESDAY_14_00, weekday=True) == "2024-04-10 14:00 UTC (Wednesday)"
    with pytest.warns(DeprecationWarning, match="use format_date"):
        assert format_utc_date(_WEDNESDAY_14_00) == "2024-04-10"


def test_the_seconds_precision_message_timestamp_is_gone() -> None:
    """``format_message_timestamp`` (``YYYY-MM-DD HH:MM:SS``, no zone label) is removed with its last caller."""
    assert not hasattr(everalgo.utils, "format_message_timestamp")
    assert not hasattr(timestamps, "format_message_timestamp")


def test_the_atomic_fact_label_is_gone() -> None:
    """The transitional 12-hour label is removed outright, leaving ``format_time`` the one shape for an instant."""
    assert not hasattr(everalgo.utils, "format_atomic_fact_time")
    assert not hasattr(timestamps, "format_atomic_fact_time")
