"""Chat messages as the text the boundary prompts and the hard limits see.

The batch prompt numbers its records, puts a time-gap line between consecutive records, escapes XML delimiters,
and indents continuation lines so message text cannot introduce a top-level record. Empty messages retain their
indices as placeholders. The step prompt keeps the original unnumbered lines and drops empty messages, and labels
the time gap before the new message. Both paths label a gap the same way (``_calculate_time_gap``). Both show
each message's time in the caller's zone (``render_chat_line``'s ``tz``). Token counting reads the raw
``speaker: text`` view.
"""

from __future__ import annotations

from datetime import UTC
from html import escape
from typing import TYPE_CHECKING

from everalgo.utils import count_tokens, render_chat_line, speaker

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import tzinfo

    from everalgo.types import ChatMessage


def _format_messages_with_indices(messages: Sequence[ChatMessage], *, tz: tzinfo = UTC) -> str:
    """Render escaped records with 0-based indices and a ``<time_gap>`` line between consecutive records.

    Empty messages keep their index as a placeholder and never raise for a missing timestamp; a gap next to a
    missing (0) timestamp reads ``No timestamp information available``.

    Raises:
        InvalidInputError: If a message with text has no timestamp (0).
    """
    lines: list[str] = []
    for index, message in enumerate(messages):
        if index:
            lines.append(f"<time_gap>{_calculate_time_gap(messages[index - 1], message)}</time_gap>")
        lines.append(f"[{index}] {_batch_record(message, tz=tz)}")
    return "\n".join(lines)


def _batch_record(message: ChatMessage, *, tz: tzinfo) -> str:
    """Keep message text inside its XML block and continuation lines inside their numbered record.

    Raises:
        InvalidInputError: If a message with text has no timestamp (0).
    """
    if not message.content:
        return "[empty message]"
    return "\n  ".join(escape(render_chat_line(message, tz=tz), quote=False).splitlines())


def _format_messages_step(messages: Sequence[ChatMessage], *, tz: tzinfo = UTC) -> str:
    """Render one line per message with text, unnumbered: the step prompt answers a single ``should_end``.

    Raises:
        InvalidInputError: If a message with text has no timestamp (0).
    """
    return "\n".join(render_chat_line(message, tz=tz) for message in messages if message.content)


def _calculate_time_gap(previous: ChatMessage, new: ChatMessage) -> str:
    """Bucket the time from ``previous`` to ``new`` into a prompt-ready label.

    Timestamps are milliseconds since the epoch; ``0`` reads as missing. The labels are the reference run's
    (everalgo-boundary 0.2.1), unit always plural: ``1 minutes``.
    """
    if previous.timestamp == 0 or new.timestamp == 0:
        return "No timestamp information available"
    total_seconds = (new.timestamp - previous.timestamp) // 1000
    if total_seconds < 0:
        return "Time gap: Messages appear to be out of order"
    if total_seconds < 60:
        return f"Time gap: {total_seconds} seconds (immediate response)"
    if total_seconds < 3600:
        return f"Time gap: {total_seconds // 60} minutes (recent conversation)"
    if total_seconds < 86_400:
        return f"Time gap: {total_seconds // 3600} hours (same day, but significant pause)"
    return f"Time gap: {total_seconds // 86_400} days (long gap, likely new conversation)"


def _count_history_tokens(messages: Sequence[ChatMessage]) -> int:
    """Sum of each message's ``speaker: text`` token count (per message, not over the joined text)."""
    total = 0
    for message in messages:
        label = speaker(message)
        total += count_tokens(f"{label}: {message.content}" if label else message.content)
    return total
