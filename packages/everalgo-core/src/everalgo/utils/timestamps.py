"""Time rendering and parsing for prompts; every package goes through here rather than formatting its own.

Every instant a model reads has one shape, ``YYYY-MM-DD HH:MM <zone>``, to the minute, where ``<zone>`` is ``UTC`` or
the offset from it (``UTC+08:00``): it is the shape the Episode prompt asks the model to write absolute times in, so
the times the model reads and the times stored in narrative text compare as equal strings. Dates the model writes and
code computes are ``YYYY-MM-DD``.

The caller picks the timezone an instant is shown in, as a ``tzinfo`` (default UTC); the date, the clock and the
weekday are all read in that zone, since "yesterday" means the day before in the user's own calendar. Weekday names
are English whatever the process locale, so a host that calls ``locale.setlocale`` does not change the prompt.
"""

from __future__ import annotations

__all__ = [
    "format_date",
    "format_time",
    "format_utc_date",
    "format_utc_time",
    "parse_utc_date",
]

import re
import warnings
from datetime import UTC, datetime, timedelta, tzinfo

from everalgo.utils.errors import InvalidInputError

_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}", re.ASCII)
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_ONE_MINUTE = timedelta(minutes=1)


def format_time(timestamp_ms: int, *, tz: tzinfo = UTC, weekday: bool = False) -> str:
    """Render an instant for a prompt, e.g. ``2024-04-10 14:00 UTC`` or ``2024-04-10 22:00 UTC+08:00``.

    Seconds are dropped, not rounded: ``14:00:59`` renders as ``14:00``. The clock is 24-hour, since a 12-hour
    clock leaves noon and midnight ambiguous.

    Args:
        timestamp_ms: Milliseconds since the epoch.
        tz: The zone to show the instant in. The label is ``UTC`` when its offset at that instant is zero, else
            ``UTC+HH:MM`` / ``UTC-HH:MM``, so a zone with daylight saving shows the offset in force then.
        weekday: Append the weekday in ``tz``, e.g. ``2024-04-10 14:00 UTC (Wednesday)``. Pass it where the model
            resolves relative times ("last Friday") against this instant; without it the model miscounts weeks.

    Returns:
        The instant in ``tz``, to the minute, with its zone label.
    """
    local = datetime.fromtimestamp(timestamp_ms / 1000, tz=tz)
    text = f"{local:%Y-%m-%d %H:%M} {_zone_label(local)}"
    return f"{text} ({_WEEKDAYS[local.weekday()]})" if weekday else text


def format_date(value: int | datetime, *, tz: tzinfo = UTC) -> str:
    """Render a calendar date, e.g. ``2024-04-10``; the inverse of :func:`parse_utc_date` when ``tz`` is UTC.

    Args:
        value: Milliseconds since the epoch, or an aware datetime such as one :func:`parse_utc_date` returned or one
            computed from it.
        tz: The zone whose calendar names the date. Leave it UTC for a datetime from :func:`parse_utc_date`: that is
            midnight UTC, which another zone may place on the day before.

    Returns:
        The date of ``value`` in ``tz``.

    Raises:
        InvalidInputError: If ``value`` is a naive datetime, which names no instant.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise InvalidInputError(
                "format_date got a naive datetime, which names no instant: "
                "attach a timezone (for example tzinfo=UTC) before calling"
            )
        local = value.astimezone(tz)
    else:
        local = datetime.fromtimestamp(value / 1000, tz=tz)
    return f"{local:%Y-%m-%d}"


def format_utc_time(timestamp_ms: int, *, weekday: bool = False) -> str:
    """Deprecated alias of :func:`format_time` in UTC, kept for packages released against core 0.6.

    Warns:
        DeprecationWarning: Always; call ``format_time(timestamp_ms, weekday=...)`` instead.
    """
    warnings.warn("format_utc_time is deprecated; use format_time", DeprecationWarning, stacklevel=2)
    return format_time(timestamp_ms, weekday=weekday)


def format_utc_date(value: int | datetime) -> str:
    """Deprecated alias of :func:`format_date` in UTC, kept for packages released against core 0.6.

    Warns:
        DeprecationWarning: Always; call ``format_date(value)`` instead.

    Raises:
        InvalidInputError: If ``value`` is a naive datetime, as :func:`format_date` raises.
    """
    warnings.warn("format_utc_date is deprecated; use format_date", DeprecationWarning, stacklevel=2)
    return format_date(value)


def parse_utc_date(text: str) -> datetime | None:
    """Parse a strict ``YYYY-MM-DD`` date, e.g. one a model wrote.

    Args:
        text: The candidate date. Zero padding is required; time parts, surrounding whitespace and non-ASCII
            digits are rejected.

    Returns:
        Midnight of that date as an aware UTC datetime, or ``None`` when ``text`` is not of that shape or names
        no real date (``2024-02-30``).
    """
    if not _DATE_PATTERN.fullmatch(text):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None


def _zone_label(local: datetime) -> str:
    """``UTC`` for a zero offset, else ``UTC+HH:MM`` / ``UTC-HH:MM``; sub-minute offsets are truncated."""
    offset = local.utcoffset() or timedelta(0)
    if not offset:
        return "UTC"
    sign = "+" if offset > timedelta(0) else "-"
    hours, minutes = divmod(abs(offset) // _ONE_MINUTE, 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"
