"""Chat messages as the text the boundary prompts and the hard limits see.

The batch prompt numbers its lines so the model can answer with indices; the step prompt answers one yes/no
question and gets the same lines unnumbered. Each line is :func:`~everalgo.utils.render_chat_line`'s, and both drop
messages with no text. Token counting reads the ``speaker: text`` view, one message at a time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.utils import count_tokens, render_chat_line, speaker

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.types import ChatMessage


def _format_messages_with_indices(messages: Sequence[ChatMessage]) -> str:
    """Render each message as ``[N] `` and its line, N 1-based over ``messages``; text-less messages are dropped.

    A dropped message keeps its number, so the indices the model answers with still address ``messages``.

    Raises:
        ValueError: If a message with text has no timestamp (0).
    """
    return "\n".join(
        f"[{index}] {render_chat_line(message)}" for index, message in enumerate(messages, start=1) if message.content
    )


def _format_messages_step(messages: Sequence[ChatMessage]) -> str:
    """Render one line per message with text, unnumbered: the step prompt answers a single ``should_end``.

    Raises:
        ValueError: If a message with text has no timestamp (0).
    """
    return "\n".join(render_chat_line(message) for message in messages if message.content)


def _count_history_tokens(messages: Sequence[ChatMessage]) -> int:
    """Sum of each message's ``speaker: text`` token count (per message, not over the joined text)."""
    total = 0
    for message in messages:
        label = speaker(message)
        total += count_tokens(f"{label}: {message.content}" if label else message.content)
    return total
