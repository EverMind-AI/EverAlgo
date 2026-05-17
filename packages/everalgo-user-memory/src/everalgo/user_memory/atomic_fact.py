"""Extract atomic facts (single verifiable assertions) from a conversation slice."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

from everalgo.llm.format import format_natural_language_time
from everalgo.llm.parse import parse_llm_json_object
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import AtomicFact, MemCell
from everalgo.user_memory._render import chat_messages, render_content
from everalgo.user_memory.prompts.en.atomic_fact import ATOMIC_FACT_PROMPT

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
            json.JSONDecodeError: If all parse strategies fail.
            ValueError: On schema validation failure (missing required fields or empty list).
        """
        rendered = render_prompt(
            ATOMIC_FACT_PROMPT,
            prompt,
            INPUT_TEXT=_render_input_text(memcell),
            TIME=_format_time_label(memcell.timestamp),
        )

        response = await self._llm.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
        )
        data = _parse_llm_response(response.content)
        atomic_facts_block = _validate_atomic_facts(data)
        return _build_atomic_facts(atomic_facts_block, sender_id=sender_id, memcell=memcell)

    extract = async_to_sync(aextract)


# Module-level helpers.


def _render_input_text(memcell: MemCell) -> str:
    """Render ChatMessage items as the raw conversation transcript for the ``{INPUT_TEXT}`` placeholder."""
    lines: list[str] = []
    for m in chat_messages(memcell):
        text = render_content(m.content)
        if not text:
            continue
        speaker = m.sender_name or m.sender_id
        lines.append(f"{speaker}: {text}")
    return "\n".join(lines)


def _format_time_label(timestamp_ms: int) -> str:
    """Render timestamp as ``March 10, 2024 (Sunday) at 2:00 PM UTC``."""
    return format_natural_language_time(timestamp_ms)


def _parse_llm_response(raw: str) -> object:
    """Parse LLM JSON response.

    Schema-specific regex is tried first as a main-path optimisation (targets the nested
    ``{"atomic_facts": {"time": ..., "atomic_fact": [...]}}`` shape); falls back to the shared
    three-tier parser (fence → direct loads → outermost braces).

    Raises:
        ValueError: If all strategies fail to find a valid JSON object.
    """
    match = re.search(
        r'\{[^{}]*"atomic_facts"[^{}]*\{[^{}]*"time"[^{}]*"atomic_fact"[^{}]*\}[^{}]*\}',
        raw,
        re.DOTALL,
    )
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return parse_llm_json_object(raw)


def _validate_atomic_facts(data: object) -> dict[str, Any]:
    """Validate ``atomic_facts`` schema; raise :class:`ValueError` on any violation."""
    if not isinstance(data, dict):
        raise ValueError("LLM response is not a JSON object")  # noqa: TRY004
    data_dict = cast("dict[str, Any]", data)
    block_raw = data_dict.get("atomic_facts")
    if not isinstance(block_raw, dict):
        raise ValueError("Missing 'atomic_facts' field in LLM response")  # noqa: TRY004
    block = cast("dict[str, Any]", block_raw)
    if "time" not in block or not block["time"]:
        raise ValueError("Missing time field in atomic_facts")
    if "atomic_fact" not in block:
        raise ValueError("Missing atomic_fact field in atomic_facts")
    atomic_fact_raw = block["atomic_fact"]
    if not isinstance(atomic_fact_raw, list):
        raise ValueError(f"atomic_fact is not a list: {type(atomic_fact_raw)}")  # noqa: TRY004
    atomic_fact_list = cast("list[object]", atomic_fact_raw)
    if not atomic_fact_list:
        raise ValueError("atomic_fact list is empty")
    return block


def _build_atomic_facts(block: dict[str, Any], *, sender_id: str | None, memcell: MemCell) -> list[AtomicFact]:
    """Split ``atomic_facts.atomic_fact`` list into individual AtomicFact entities."""
    time_label = block["time"] if isinstance(block.get("time"), str) else _format_time_label(memcell.timestamp)
    facts_list = cast("list[object]", block["atomic_fact"])
    out: list[AtomicFact] = []
    for item in facts_list:
        if not isinstance(item, str) or not item.strip():
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
