"""Shared rendering helpers for user-memory extractors.

This module is internal (``_render``-prefixed) — not part of the public API of ``everalgo.user_memory``.
``render_content`` is re-exported from ``everalgo.types._render`` (moved to core in Stage 4 so that
``everalgo-agent-memory`` can import it without creating a cross-package dependency on user-memory).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.types._render import render_content
from everalgo.types.chat import ChatMessage

if TYPE_CHECKING:
    from everalgo.types.conversation import MemCell

__all__ = ["chat_messages", "render_content"]


def chat_messages(memcell: MemCell) -> list[ChatMessage]:
    """Filter ``memcell.items`` to ``ChatMessage`` entries only.

    User-memory extractors silently skip ``ToolCallRequest`` / ``ToolCallResult`` — this is an
    explicit contract supporting the agent → user-memory pipeline (see EverAlgo AGENTS.md). The
    caller need not pre-filter; an AgentMemCell-shaped ``MemCell`` is acceptable input.
    """
    return [item for item in memcell.items if isinstance(item, ChatMessage)]
