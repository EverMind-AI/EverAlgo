"""Input helpers shared by the user-memory operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.types.chat import ChatMessage

if TYPE_CHECKING:
    from everalgo.types.conversation import MemCell

__all__ = ["chat_messages", "nonblank"]


def chat_messages(memcell: MemCell) -> list[ChatMessage]:
    """Filter ``memcell.items`` to ``ChatMessage`` entries only.

    User-memory extractors silently skip ``ToolCallRequest`` / ``ToolCallResult`` — this is an
    explicit contract supporting the agent → user-memory pipeline (see EverAlgo AGENTS.md). The
    caller need not pre-filter; an AgentMemCell-shaped ``MemCell`` is acceptable input.
    """
    return [item for item in memcell.items if isinstance(item, ChatMessage)]


def nonblank(text: str | None) -> str | None:
    """``text``, or None when it is None or only whitespace."""
    return text if text is not None and text.strip() else None
