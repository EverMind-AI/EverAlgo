"""Conversation wire types shared by chat and agent pipelines — ``MemCell`` and ``ConversationItem``."""

from __future__ import annotations

__all__ = ["ConversationItem", "MemCell"]

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from everalgo.types.agent import ToolCallRequest, ToolCallResult
from everalgo.types.chat import ChatMessage

ConversationItem = Annotated[
    ChatMessage | ToolCallRequest | ToolCallResult,
    Field(discriminator="kind"),
]
"""One ordered step in a conversation or agent trajectory; discriminated by ``kind``."""


class MemCell(BaseModel):
    """A coherent slice produced by boundary detection.

    ``items`` is ordered; chat scenarios contain only ``ChatMessage`` while agent scenarios may also carry
    ``ToolCallRequest`` / ``ToolCallResult``. User-memory extractors silently skip non-``ChatMessage`` items.
    ``timestamp`` is the Unix epoch ms of the slice's last ``ChatMessage``; tool items after it do not move it.
    """

    items: list[ConversationItem]
    timestamp: int  # Unix epoch ms of the slice's last ChatMessage

    model_config = ConfigDict(extra="ignore")
