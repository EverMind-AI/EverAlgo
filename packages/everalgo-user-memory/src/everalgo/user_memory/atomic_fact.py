"""Extract atomic facts (single verifiable assertions) from a conversation slice or free text."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

from everalgo.llm.format import format_atomic_fact_time, format_message_timestamp
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import AtomicFact, MemCell
from everalgo.user_memory._render import chat_messages, render_content
from everalgo.user_memory.prompts.en.atomic_fact import ATOMIC_FACT_PROMPT
from everalgo.user_memory.prompts.en.atomic_fact_from_text import ATOMIC_FACT_FROM_TEXT_PROMPT_EN

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


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
            TIME=format_atomic_fact_time(memcell.items[0].timestamp),
        )

        block = await _call_llm_for_atomic_facts(self._llm, rendered, memcell.timestamp)
        return _build_atomic_facts(block, sender_id=sender_id)

    extract = async_to_sync(aextract)

    async def aextract_from_text(
        self,
        text: str,
        *,
        timestamp: int,
        prompt: str | None = None,
    ) -> list[AtomicFact]:
        """Extract atomic facts from a piece of text.

        Generic primitive: caller decides text source (episode body, summary, third-party
        document, email body, etc.) — function does not bind to any text type. Single LLM call.

        Args:
            text: Source text. Substituted into the prompt at ``{{EPISODE_TEXT}}`` (current default) or ``{{TEXT}}`` (legacy).
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
        time_str = format_atomic_fact_time(timestamp)
        template = prompt if prompt is not None else ATOMIC_FACT_FROM_TEXT_PROMPT_EN
        rendered = template.replace("{{EPISODE_TEXT}}", text).replace("{{TEXT}}", text).replace("{{TIME}}", time_str)

        block = await _call_llm_for_atomic_facts(self._llm, rendered, timestamp)
        facts: list[str] = block["atomic_fact"]
        logger.debug("aextract_from_text extracted %d facts", len(facts))
        return _build_atomic_facts(block, sender_id=None)

    extract_from_text = async_to_sync(aextract_from_text)
    """Sync bridge — only callable from non-event-loop contexts."""


# ---------------------------------------------------------------------------
# LLM callsite — brace-balanced JSON extraction + 5-retry (mirror b150b32).
# ---------------------------------------------------------------------------


async def _call_llm_for_atomic_facts(llm: LLMClient, rendered: str, timestamp: int) -> dict[str, Any]:
    """Call LLM and return validated atomic-fact inner block.

    Accepts two top-level keys as aliases of the same inner schema
    ``{"time": "...", "atomic_fact": [...]}``:

    - ``event_log`` — alias used by ``EVENT_LOG_PROMPT``.
    - ``atomic_facts`` — alias used by ``ATOMIC_FACT_FROM_TEXT_PROMPT_EN`` (algo default).

    Both prompts ship from ``prompts/en/atomic_fact_from_text.py``; the caller picks via
    the ``prompt=`` argument on :meth:`AtomicFactExtractor.aextract_from_text`. Tolerating
    both keys here lets a single algo callsite back both schemas without forcing a public-API rename.

    Raises:
        ValueError: If no JSON found, neither key is present, or ``atomic_fact`` is not a list.
    """
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    json_str = _extract_json_object(text)
    data: dict[str, Any] = json.loads(json_str)
    if "event_log" in data:
        raw_block = data["event_log"]
    elif "atomic_facts" in data:
        raw_block = data["atomic_facts"]
    else:
        raise ValueError(f"event_log/atomic_facts key missing from LLM response: {data!r}")
    if not isinstance(raw_block, dict):
        raise ValueError(  # noqa: TRY004
            f"event_log/atomic_facts must be a dict, got {type(raw_block).__name__}: {raw_block!r}"
        )
    block: dict[str, Any] = cast("dict[str, Any]", raw_block)
    if "time" not in block or not block["time"]:
        raise ValueError(f"time key missing from LLM response: {data!r}")
    raw_facts = block.get("atomic_fact", [])
    if not isinstance(raw_facts, list):
        raise ValueError(f"atomic_fact must be a list, got {type(raw_facts).__name__}: {raw_facts!r}")  # noqa: TRY004
    # Filter out non-string and empty items (mirror _AtomicFactsBlock._filter_non_strings).
    block["atomic_fact"] = [item for item in cast("list[Any]", raw_facts) if isinstance(item, str) and item.strip()]  # type: ignore[redundant-cast]
    block["timestamp"] = timestamp
    return block


def _extract_json_object(text: str) -> str:
    """First balanced {{...}} block in text (brace-balanced parser for nested JSON)."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in atomic_fact LLM response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Unbalanced JSON in atomic_fact LLM response: {text[:200]!r}")


# Module-level helpers.


def _render_input_text(memcell: MemCell) -> str:
    """Render ChatMessage items as ``[<ts>] <speaker>: <text>`` lines for the ``{INPUT_TEXT}`` placeholder.

    Prepends each message's ISO timestamp to anchor message-level time signals into the LLM context
    so atomic_fact extraction can preserve when each event happened.
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


def _build_atomic_facts(block: dict[str, Any], *, sender_id: str | None) -> list[AtomicFact]:
    """Split ``atomic_facts.atomic_fact`` list into individual AtomicFact entities."""
    time = block["timestamp"]
    out: list[AtomicFact] = []
    for item in block.get("atomic_fact", []):
        if not isinstance(item, str) or not item.strip():
            continue
        out.append(
            AtomicFact.model_validate(
                {
                    "owner_id": sender_id,
                    "content": item.strip(),
                    "timestamp": time,
                }
            )
        )
    return out
