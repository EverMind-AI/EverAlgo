"""Time-formatting helpers for LLM prompt rendering.

Four flavours:
- ``format_message_timestamp`` — UTC anchor for inline conversation-line
  prefixes (e.g. ``[2023-11-14 22:13:20] Alice: ...``). Language-agnostic.
- ``format_iso_timestamp`` — ISO 8601 with explicit UTC offset
  (e.g. ``"2024-01-01T06:00:00+00:00"``). Used in JSON-shaped prompt blocks
  whose timestamp field the LLM must parse and reason about (Episode prompt
  conversation block).
- ``format_atomic_fact_time`` — AtomicFact-flavoured timestamp (e.g. ``March 10, 2024(Sunday) at
  02:00 PM``). No space before the weekday parenthesis, no ``UTC`` suffix. Used in AtomicFact prompts
  where the LLM copies the value verbatim into output JSON.
- ``format_natural_language_time`` — human-readable label for LLM time-of-day
  reasoning (e.g. ``Conversation start time:`` / ``TIME:``). Supports EN + ZH;
  callers pick via ``lang`` (default ``"en"`` matches the current EverAlgo
  default-EN prompt policy).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

__all__ = [
    "Lang",
    "format_atomic_fact_time",
    "format_iso_timestamp",
    "format_message_timestamp",
    "format_natural_language_time",
]

Lang = Literal["en", "zh"]


def format_message_timestamp(timestamp_ms: int) -> str:
    """``YYYY-MM-DD HH:MM:SS`` UTC for conversation-line prefixes (e.g. ``2023-11-14 22:13:20``).

    Language-agnostic. Space-separated, no ``T`` delimiter, no trailing ``Z``.
    UTC is implicit by convention (callers always pass ms-since-epoch UTC).
    """
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def format_iso_timestamp(timestamp_ms: int) -> str:
    """ISO 8601 with explicit UTC offset (e.g. ``"2024-01-01T06:00:00+00:00"``).

    Emits whatever Python's :py:meth:`datetime.datetime.isoformat` returns on a timezone-aware UTC
    datetime — ``T`` delimiter and ``+00:00`` suffix included. Distinct from
    :func:`format_message_timestamp` (``[YYYY-MM-DD HH:MM:SS]`` prefix, no T, no offset),
    which is used for boundary history lines; this function is used in Episode prompt JSON blocks
    where the LLM is asked to reason about absolute dates.
    """
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()


def format_atomic_fact_time(timestamp_ms: int) -> str:
    """Timestamp formatter for AtomicFact ``{TIME}`` / ``{{TIME}}`` prompt placeholder.

    Format: ``"March 10, 2024(Sunday) at 02:00 PM"`` — no space before the weekday parenthesis,
    no ``UTC`` suffix, zero-padded day/hour. The AtomicFact LLM copies this value verbatim into
    its output JSON ``time`` field. Distinct from :func:`format_natural_language_time`
    (Episode format, which includes a space before the parenthesis and a ``UTC`` suffix).
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    return dt.strftime("%B %d, %Y(%A) at %I:%M %p")


def format_natural_language_time(timestamp_ms: int, *, lang: Lang = "en") -> str:
    """Human-readable timestamp for LLM time-of-day reasoning labels.

    EN: ``November 14, 2023 (Tuesday) at 10:13 PM UTC``
    ZH: ``2023 年 11 月 14 日 (星期二) 下午 10:13 UTC`` (uses CJK parentheses in the actual output)
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    if lang == "zh":
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][dt.weekday()]
        hour_12 = dt.hour % 12 or 12
        ampm = "下午" if dt.hour >= 12 else "上午"
        return f"{dt.year} 年 {dt.month} 月 {dt.day} 日（{weekday}）{ampm} {hour_12}:{dt.minute:02d} UTC"  # noqa: RUF001
    return dt.strftime("%B %d, %Y (%A) at %I:%M %p UTC")
