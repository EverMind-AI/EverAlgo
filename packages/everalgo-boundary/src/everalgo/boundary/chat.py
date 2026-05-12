"""Chat-style MemCell extractor — slice a message stream by topic boundaries."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.boundary._tokenize import count_tokens
from everalgo.boundary.prompts.en.chat import CHAT_BOUNDARY_DETECT_PROMPT_EN
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import MemCell, Message

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


class ChatMemCellExtractor:
    """Detect MemCell boundaries in a chat-style message stream.

    Stateless: no ``__init__``, no instance state. Thread/async safe — instances are interchangeable.
    Customize per call via ``llm=`` and ``prompt=`` arguments.

    Algorithm (minimal reference impl):
        1. Render LLM prompt with the message stream + token budget hint.
        2. Call LLM, parse JSON ``{"split_at": int | null}``.
        3. Build one or two MemCells from the split.

    For production-grade boundary detection (multi-split / token-aware force_split / boundary_reason
    classification), replace the prompt via ``prompt=`` argument or monkey-patch the module constant.
    """

    async def adetect(
        self,
        messages: list[Message],
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[MemCell]:
        """Async main implementation: ask LLM for boundary split point.

        Parameters
        ----------
        messages : list[Message]
            Ordered chat messages (user/assistant turns).
        llm : LLMClient or None, optional
            Per-call LLM override. Falls back through scoped (``use(...)``) and default
            (``configure(...)``); raises ``LLMNotConfiguredError`` if all None.
        prompt : str or None, optional
            Per-call prompt override. Defaults to ``CHAT_BOUNDARY_DETECT_PROMPT_EN``.

        Returns
        -------
        list[MemCell]
            At least one cell. The minimal ref impl produces either 1 cell (no split) or 2 cells
            (one split point).
        """
        client = everalgo.llm.resolve(llm)
        rendered = render_prompt(
            CHAT_BOUNDARY_DETECT_PROMPT_EN,
            prompt,
            messages=_format_messages_for_prompt(messages),
            token_count=count_tokens(_concat_messages(messages)),
        )
        response = await client.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
        )
        return _build_memcells_from_llm_response(response.content, messages)

    detect = async_to_sync(adetect)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helper functions — stateless utilities (per AGENTS.md §5).


def _concat_messages(messages: list[Message]) -> str:
    """Concatenate messages into a single prompt-friendly string."""
    return "\n".join(f"[{m.role.value}] {m.content}" for m in messages)


def _format_messages_for_prompt(messages: list[Message]) -> str:
    """Format messages with index prefix (LLM uses index for split_at)."""
    return "\n".join(f"{i}. [{m.role.value}] {m.content}" for i, m in enumerate(messages))


def _build_memcells_from_llm_response(raw: str, messages: list[Message]) -> list[MemCell]:
    """Parse LLM JSON ``{"split_at": int | null}`` and build MemCell list."""
    parsed = json.loads(raw)
    split_at = parsed.get("split_at")
    if split_at is None:
        return [_make_memcell(messages, suffix="0")]
    return [
        _make_memcell(messages[:split_at], suffix="0"),
        _make_memcell(messages[split_at:], suffix="1"),
    ]


def _make_memcell(slice_msgs: list[Message], *, suffix: str) -> MemCell:
    """Build a MemCell with deterministic id derived from timestamp + suffix."""
    timestamp = slice_msgs[-1].timestamp if slice_msgs else 0
    return MemCell(
        id=f"mc_{timestamp}_{suffix}",
        messages=slice_msgs,
        timestamp=timestamp,
    )
