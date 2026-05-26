"""Time-formatting helpers for LLM prompt rendering.

Two flavours:
- ``format_message_timestamp`` — ISO 8601 UTC anchor for inline conversation-line
  prefixes (e.g. ``[2023-11-14T22:13:20Z] Alice: ...``). Language-agnostic.
- ``format_natural_language_time`` — human-readable label for LLM time-of-day
  reasoning (e.g. ``Conversation start time:`` / ``TIME:``). Supports EN + ZH;
  callers pick via ``lang`` (default ``"en"`` matches the current EverAlgo
  default-EN prompt policy).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

__all__ = ["Lang", "format_message_timestamp", "format_natural_language_time"]

Lang = Literal["en", "zh"]


def format_message_timestamp(timestamp_ms: int) -> str:
    """ISO 8601 UTC for conversation-line prefixes (e.g. ``2023-11-14T22:13:20Z``).

    Language-agnostic. Used wherever an LLM prompt needs a machine-readable
    time anchor in front of a message body.
    """
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    return dt.strftime("%B %-d, %Y (%A) at %-I:%M %p UTC")
