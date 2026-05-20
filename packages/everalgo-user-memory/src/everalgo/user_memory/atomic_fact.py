"""Extract atomic facts (single verifiable assertions) from a conversation slice or free text."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from asgiref.sync import async_to_sync
from pydantic import BaseModel, Field, field_validator

from everalgo.llm.format import format_message_timestamp, format_natural_language_time
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import AtomicFact, MemCell
from everalgo.user_memory._render import chat_messages, render_content
from everalgo.user_memory.prompts.en.atomic_fact import ATOMIC_FACT_PROMPT
from everalgo.user_memory.prompts.en.atomic_fact_from_text import ATOMIC_FACT_FROM_TEXT_PROMPT_EN

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured outputs schemas for atomic fact LLM extraction.
# ---------------------------------------------------------------------------


class _AtomicFactsBlock(BaseModel):
    """Nested atomic facts data returned by the LLM."""

    time: str = Field(..., description="The conversation/text start time as an exact string from input")
    atomic_fact: list[str] = Field(..., description="List of atomic fact sentences")

    @field_validator("atomic_fact", mode="before")
    @classmethod
    def _filter_non_strings(cls, v: object) -> list[str]:
        """Filter out non-string items and empty strings; robust against LLM quirks."""
        if not isinstance(v, list):
            raise TypeError(f"atomic_fact must be a list, got {type(v).__name__}")
        return [item for item in v if isinstance(item, str) and item.strip()]


class _AtomicFactLLMResponse(BaseModel):
    """Structured outputs schema for atomic fact extraction."""

    atomic_facts: _AtomicFactsBlock = Field(..., description="Nested block containing time and atomic_fact list")


class AtomicFactExtractor:
    """Extract zero or more atomic facts from one MemCell.

    Non-ChatMessage items in memcell.items are silently skipped (agent → user-memory contract).
    Each string in the LLM ``atomic_facts.atomic_fact`` list becomes one :class:`AtomicFact` entity.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        memcell: MemCell,
        *,
        sender_id: str | None,
        prompt: str | None = None,
    ) -> list[AtomicFact]:
        """Extract atomic facts for ``sender_id`` from ``memcell``.

        Args:
            memcell: Source slice from boundary detection.
            sender_id: Owner tag stamped on each resulting AtomicFact; pass ``None`` for generic
                (whole-memcell) facts that do not bind to any user. The prompt itself does not
                consume sender_id.
            prompt: Prompt override; ``None`` uses the bundled default.

        Raises:
            LLMError: From the LLM call.
            ValueError: If the LLM returns no parsed structured output.
        """
        rendered = render_prompt(
            ATOMIC_FACT_PROMPT,
            prompt,
            INPUT_TEXT=_render_input_text(memcell),
            TIME=_format_time_label(memcell.timestamp),
        )

        response = await self._llm.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format=_AtomicFactLLMResponse,
        )
        parsed = cast("_AtomicFactLLMResponse | None", response.parsed)
        if parsed is None:
            raise ValueError("LLM returned no parsed structured output")
        return _build_atomic_facts(parsed.atomic_facts, sender_id=sender_id, memcell=memcell)

    extract = async_to_sync(aextract)

    async def aextract_from_text(
        self,
        text: str,
        *,
        timestamp: int,
        prompt: str | None = None,
    ) -> list[str]:
        """Extract atomic facts from a piece of text.

        Generic primitive: caller decides text source (episode body, summary, third-party
        document, email body, etc.) — function does not bind to any text type. Single LLM call.

        Args:
            text: Source text. Substituted into the prompt at the ``{{TEXT}}`` placeholder.
            timestamp: Unix epoch milliseconds. Rendered to a human-readable English form
                and used as the TIME anchor for resolving relative time expressions in
                ``text`` (e.g. "yesterday" -> "yesterday (March 9, 2024)").
            prompt: Optional prompt template override; default uses
                ``ATOMIC_FACT_FROM_TEXT_PROMPT_EN``.

        Returns:
            List of atomic-fact sentences. Empty list if text yields no facts.

        Raises:
            ValueError: If the LLM returns no parsed structured output.
            LLMError: Propagated from LLM client.
        """
        time_str = format_natural_language_time(timestamp)
        template = prompt if prompt is not None else ATOMIC_FACT_FROM_TEXT_PROMPT_EN
        rendered = template.replace("{{TEXT}}", text).replace("{{TIME}}", time_str)

        response = await self._llm.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format=_AtomicFactLLMResponse,
        )
        parsed = cast("_AtomicFactLLMResponse | None", response.parsed)
        if parsed is None:
            raise ValueError("LLM returned no parsed structured output")
        facts = parsed.atomic_facts.atomic_fact
        logger.debug("aextract_from_text extracted %d facts", len(facts))
        return facts

    extract_from_text = async_to_sync(aextract_from_text)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helpers.


def _render_input_text(memcell: MemCell) -> str:
    """Render ChatMessage items as ``[<ts>] <speaker>: <text>`` lines for the ``{INPUT_TEXT}`` placeholder.

    Mirror EverCore main ``atomic_fact_extractor.py:255-262`` which prepends the
    ISO timestamp of each message — this anchors message-level time signals into
    the LLM context so atomic_fact extraction can preserve when each event
    happened (critical for LoCoMo temporal questions).
    """
    lines: list[str] = []
    for m in chat_messages(memcell):
        text = render_content(m.content)
        if not text:
            continue
        speaker = m.sender_name or m.sender_id
        ts_str = format_message_timestamp(m.timestamp)
        lines.append(f"[{ts_str}] {speaker}: {text}")
    return "\n".join(lines)


def _format_time_label(timestamp_ms: int) -> str:
    """Render timestamp as ``March 10, 2024 (Sunday) at 2:00 PM UTC``."""
    return format_natural_language_time(timestamp_ms)


def _build_atomic_facts(block: _AtomicFactsBlock, *, sender_id: str | None, memcell: MemCell) -> list[AtomicFact]:
    """Split ``atomic_facts.atomic_fact`` list into individual AtomicFact entities."""
    time_label = block.time if block.time else _format_time_label(memcell.timestamp)
    out: list[AtomicFact] = []
    for item in block.atomic_fact:
        if not item.strip():
            continue
        out.append(
            AtomicFact.model_validate(
                {
                    "owner_id": sender_id,
                    "fact": item.strip(),
                    "timestamp": memcell.timestamp,
                    "time_label": time_label,
                }
            )
        )
    return out
