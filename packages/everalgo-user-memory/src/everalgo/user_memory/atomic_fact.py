"""AtomicFact extractor — aligned with new-release opensource.

Algorithm follows opensource ``atomic_fact_extractor.py`` (was ``event_log_extractor.py`` in older
releases). Prompt ported verbatim from ``atomic_fact_prompts.py``; uses double-brace placeholders
(``{{INPUT_TEXT}}`` / ``{{TIME}}``) rendered via :py:meth:`str.replace`.

Algorithm details:
    - 5-retry loop wrapping the whole extraction; raises RuntimeError after exhaustion.
    - 4-strategy JSON parse:
        1. ```json``` code block.
        2. Any ``` code block (skip language identifier).
        3. Regex ``{atomic_facts{time,atomic_fact}}``.
        4. Direct ``json.loads``.
    - Schema validation: atomic_facts / time / atomic_fact present, atomic_fact is non-empty list.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt_replace
from everalgo.types import AtomicFact, MemCell
from everalgo.user_memory.prompts.en.atomic_fact import ATOMIC_FACT_PROMPT

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)

_MAX_LLM_RETRIES = 5


class AtomicFactExtractor:
    """Extract atomic facts (single verifiable assertions) from a single MemCell.

    Stateless callable class. Mirrors new-release opensource ``AtomicFactExtractor`` — one LLM call (with
    5-retry on parse / schema failure), 4-strategy JSON parse, output is
    ``{atomic_facts: {time, atomic_fact: list[str]}}``, we explode ``atomic_fact`` into individual
    :class:`AtomicFact` entities.
    """

    async def aextract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[AtomicFact]:
        """Ask the LLM to enumerate atomic facts; raises RuntimeError after 5 retries."""
        client = everalgo.llm.resolve(llm)
        rendered = render_prompt_replace(
            ATOMIC_FACT_PROMPT,
            prompt,
            {
                "{{INPUT_TEXT}}": _render_input_text(memcell),
                "{{TIME}}": _format_time_label(memcell.timestamp),
            },
        )

        last_error: Exception | None = None
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                response = await client.chat(
                    messages=[LLMChatMessage(role="user", content=rendered)],
                    response_format={"type": "json_object"},
                )
                data = _parse_llm_response(response.content)
                atomic_facts_block = _validate_atomic_facts(data)
                return _build_atomic_facts(atomic_facts_block, memcell)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                last_error = e
                logger.warning("atomic_fact retry %d/%d: %s", attempt + 1, _MAX_LLM_RETRIES, e)
                if attempt == _MAX_LLM_RETRIES - 1:
                    raise RuntimeError(
                        f"AtomicFactExtractor: all {_MAX_LLM_RETRIES} retries exhausted "
                        f"(JSON parse / schema mismatch). Last error: {last_error}"
                    ) from last_error
                continue

        return []  # unreachable

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helpers.


def _render_input_text(memcell: MemCell) -> str:
    """Render messages as the raw conversation transcript expected by ``{{INPUT_TEXT}}``."""
    lines: list[str] = []
    for m in memcell.messages:
        if not m.content:
            continue
        speaker = m.sender_name or m.role.value
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)


def _format_time_label(timestamp_ms: int) -> str:
    """Render timestamp in opensource example form ``March 10, 2024(Sunday) at 2:00 PM UTC``."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    return dt.strftime("%B %-d, %Y(%A) at %-I:%M %p UTC")


def _parse_llm_response(raw: str) -> object:  # noqa: C901 — 4 distinct parse strategies match opensource shape
    """4-strategy JSON parse ported from opensource line 92-151.

    Strategies (in order):
        1. ```json``` code block.
        2. Any ``` code block (skip language identifier on the first token).
        3. Regex ``{atomic_facts{time,atomic_fact}}``.
        4. Direct ``json.loads``.

    Raises :class:`ValueError` if all strategies fail (caller catches and retries).
    """
    if "```json" in raw:
        start = raw.find("```json") + 7
        end = raw.find("```", start)
        if end > start:
            try:
                return json.loads(raw[start:end].strip())
            except json.JSONDecodeError:
                pass
    if "```" in raw:
        start = raw.find("```") + 3
        head = raw[start : start + 10].strip().split()
        if head and head[0].isalpha():
            newline = raw.find("\n", start)
            if newline > 0:
                start = newline + 1
        end = raw.find("```", start)
        if end > start:
            try:
                return json.loads(raw[start:end].strip())
            except json.JSONDecodeError:
                pass
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
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        raise ValueError("Unable to parse LLM response into valid JSON format") from e


def _validate_atomic_facts(data: object) -> dict[str, Any]:
    """Validate new-release ``atomic_facts`` schema.

    All violations raise :class:`ValueError` (caller catches + retries) — matches opensource convention
    of using ValueError uniformly for both schema-shape and value-missing failures.
    """
    if not isinstance(data, dict):
        raise ValueError("LLM response is not a JSON object")  # noqa: TRY004 — opensource uses ValueError uniformly
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


def _build_atomic_facts(block: dict[str, Any], memcell: MemCell) -> list[AtomicFact]:
    """Split new-release ``atomic_facts.atomic_fact`` list into individual AtomicFact entities."""
    owner_id = _derive_owner_id(memcell)
    time_label = block["time"] if isinstance(block.get("time"), str) else _format_time_label(memcell.timestamp)
    facts_list = cast("list[object]", block["atomic_fact"])
    out: list[AtomicFact] = []
    for item in facts_list:
        if not isinstance(item, str) or not item.strip():
            continue
        out.append(
            AtomicFact.model_validate(
                {
                    "id": f"af_{uuid.uuid4().hex[:12]}",
                    "owner_id": owner_id,
                    "fact": item.strip(),
                    "timestamp": memcell.timestamp,
                    "parent_type": "memcell",
                    "parent_id": memcell.event_id or "",
                    "time_label": time_label,
                }
            )
        )
    return out


def _derive_owner_id(memcell: MemCell) -> str:
    if memcell.participants:
        return memcell.participants[0]
    for m in memcell.messages:
        if m.sender_id:
            return m.sender_id
    return "u_default"
