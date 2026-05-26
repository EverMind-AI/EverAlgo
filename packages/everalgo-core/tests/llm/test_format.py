"""Unit tests for everalgo.llm.format — time-formatting helpers."""

from __future__ import annotations

import re

from everalgo.llm.format import format_message_timestamp, format_natural_language_time

# Fixed reference timestamp: 2023-11-14T22:13:20Z (Tuesday, 10:13 PM UTC)
_TS_MS = 1_700_000_000_000

# A morning timestamp: 2023-11-14T03:00:00Z (Tuesday, 3:00 AM UTC)
# 1_700_000_000_000 - (22 * 3600 + 13 * 60 + 20) * 1000 + 3 * 3600 * 1000
_TS_AM_MS = _TS_MS - (22 * 3_600 + 13 * 60 + 20) * 1_000 + 3 * 3_600 * 1_000

# Midnight: 2023-11-14T00:00:00Z
_TS_MIDNIGHT_MS = _TS_MS - (22 * 3_600 + 13 * 60 + 20) * 1_000

# Noon: 2023-11-14T12:00:00Z
_TS_NOON_MS = _TS_MS - (22 * 3_600 + 13 * 60 + 20) * 1_000 + 12 * 3_600 * 1_000

# Another weekday — 2023-11-13T10:00:00Z is a Monday
_TS_MONDAY_MS = _TS_MS - (1 * 24 * 3_600 + 12 * 3_600 + 13 * 60 + 20) * 1_000


# ---------------------------------------------------------------------------
# format_message_timestamp
# ---------------------------------------------------------------------------


def test_format_message_timestamp_shape() -> None:
    result = format_message_timestamp(_TS_MS)
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", result), f"unexpected shape: {result!r}"


def test_format_message_timestamp_known_value() -> None:
    assert format_message_timestamp(_TS_MS) == "2023-11-14 22:13:20"


def test_format_message_timestamp_no_timezone_marker() -> None:
    """The new evercore-aligned format drops the ``Z`` tz marker; UTC is implicit by convention."""
    result = format_message_timestamp(_TS_MS)
    assert "Z" not in result
    assert "+" not in result
    assert "T" not in result  # space-separated, not ``T``


def test_format_message_timestamp_second_boundary() -> None:
    assert format_message_timestamp(_TS_MS + 1_000) == "2023-11-14 22:13:21"


# ---------------------------------------------------------------------------
# format_natural_language_time — EN (default)
# ---------------------------------------------------------------------------


def test_format_natural_language_time_default_is_en() -> None:
    default = format_natural_language_time(_TS_MS)
    explicit_en = format_natural_language_time(_TS_MS, lang="en")
    assert default == explicit_en


def test_format_natural_language_time_en_known_value() -> None:
    result = format_natural_language_time(_TS_MS, lang="en")
    assert result == "November 14, 2023 (Tuesday) at 10:13 PM UTC"


def test_format_natural_language_time_en_am() -> None:
    result = format_natural_language_time(_TS_AM_MS, lang="en")
    assert "03:00 AM UTC" in result
    assert "November 14, 2023" in result
    assert "(Tuesday)" in result


def test_format_natural_language_time_en_midnight() -> None:
    result = format_natural_language_time(_TS_MIDNIGHT_MS, lang="en")
    assert "12:00 AM UTC" in result


def test_format_natural_language_time_en_noon() -> None:
    result = format_natural_language_time(_TS_NOON_MS, lang="en")
    assert "12:00 PM UTC" in result


def test_format_natural_language_time_en_weekday() -> None:
    result = format_natural_language_time(_TS_MONDAY_MS, lang="en")
    assert "(Monday)" in result


# ---------------------------------------------------------------------------
# format_natural_language_time — ZH
# ---------------------------------------------------------------------------


def test_format_natural_language_time_zh_known_value() -> None:
    result = format_natural_language_time(_TS_MS, lang="zh")
    assert result == "2023 年 11 月 14 日（星期二）下午 10:13 UTC"  # noqa: RUF001


def test_format_natural_language_time_zh_am() -> None:
    result = format_natural_language_time(_TS_AM_MS, lang="zh")
    assert "上午" in result
    assert "3:00" in result
    assert "星期二" in result


def test_format_natural_language_time_zh_midnight() -> None:
    result = format_natural_language_time(_TS_MIDNIGHT_MS, lang="zh")
    assert "上午" in result
    assert "12:00" in result


def test_format_natural_language_time_zh_noon() -> None:
    result = format_natural_language_time(_TS_NOON_MS, lang="zh")
    assert "下午" in result
    assert "12:00" in result


def test_format_natural_language_time_zh_weekday() -> None:
    result = format_natural_language_time(_TS_MONDAY_MS, lang="zh")
    assert "星期一" in result


# ---------------------------------------------------------------------------
# format_atomic_fact_time — evercore EventLog format
# ---------------------------------------------------------------------------


def test_format_atomic_fact_time_known_value() -> None:
    from everalgo.llm.format import format_atomic_fact_time

    # _TS_MS = 2023-11-14 22:13:20 UTC → 10:13 PM Tuesday
    assert format_atomic_fact_time(_TS_MS) == "November 14, 2023(Tuesday) at 10:13 PM"


def test_format_atomic_fact_time_no_space_before_paren() -> None:
    """Distinguishing feature vs format_natural_language_time: no space before ``(``."""
    from everalgo.llm.format import format_atomic_fact_time

    result = format_atomic_fact_time(_TS_MS)
    assert ", 2023(Tuesday)" in result
    assert ", 2023 (Tuesday)" not in result


def test_format_atomic_fact_time_no_utc_suffix() -> None:
    """Distinguishing feature vs format_natural_language_time: no trailing ``UTC``."""
    from everalgo.llm.format import format_atomic_fact_time

    result = format_atomic_fact_time(_TS_MS)
    assert not result.endswith("UTC")


def test_format_atomic_fact_time_zero_padded_hour() -> None:
    """Single-digit hours are zero-padded (strftime ``%I``)."""
    from everalgo.llm.format import format_atomic_fact_time

    # _TS_AM_MS = 2023-11-14 03:00:00 UTC → 03:00 AM
    assert "03:00 AM" in format_atomic_fact_time(_TS_AM_MS)


def test_format_atomic_fact_time_noon() -> None:
    from everalgo.llm.format import format_atomic_fact_time

    assert "12:00 PM" in format_atomic_fact_time(_TS_NOON_MS)


def test_format_atomic_fact_time_weekday() -> None:
    from everalgo.llm.format import format_atomic_fact_time

    # _TS_MONDAY_MS → Monday
    assert "(Monday)" in format_atomic_fact_time(_TS_MONDAY_MS)


# ---------------------------------------------------------------------------
# format_iso_timestamp — Python isoformat() with UTC offset
# ---------------------------------------------------------------------------


def test_format_iso_timestamp_known_value() -> None:
    """Output is byte-for-byte ``datetime_utils.to_iso_format``."""
    from everalgo.llm.format import format_iso_timestamp

    assert format_iso_timestamp(_TS_MS) == "2023-11-14T22:13:20+00:00"


def test_format_iso_timestamp_has_t_delimiter_and_utc_offset() -> None:
    """Distinguishing features vs ``format_message_timestamp`` (space-separated, no offset)."""
    from everalgo.llm.format import format_iso_timestamp

    result = format_iso_timestamp(_TS_MS)
    assert "T" in result
    assert result.endswith("+00:00")
    assert " " not in result


def test_format_iso_timestamp_midnight() -> None:
    from everalgo.llm.format import format_iso_timestamp

    assert format_iso_timestamp(_TS_MIDNIGHT_MS) == "2023-11-14T00:00:00+00:00"
