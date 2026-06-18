"""Extract anticipated commitments (Foresight) from a conversation slice."""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

from everalgo.llm.format import format_message_timestamp
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Foresight, MemCell
from everalgo.user_memory._render import chat_messages, render_content
from everalgo.user_memory.prompts.en.foresight import FORESIGHT_GENERATION_PROMPT

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


_FORESIGHT_TEMPERATURE = 0.3
_FORESIGHT_MAX_COUNT = 10
_FORESIGHT_MIN_COUNT = 4  # warn-only floor; LLM may legitimately produce fewer


class ForesightExtractor:
    """Extract zero or more foresights from one MemCell.

    Non-ChatMessage items in memcell.items are silently skipped (agent → user-memory contract).
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        memcell: MemCell,
        *,
        sender_id: str,
        prompt: str | None = None,
    ) -> list[Foresight]:
        """Extract foresights for ``sender_id`` from ``memcell``.

        Args:
            memcell: Source slice from boundary detection.
            sender_id: Must be one of memcell's chat senders; not inferred.
            prompt: Prompt override; ``None`` uses the bundled default.

        Raises:
            LLMError: From the LLM call.
            json.JSONDecodeError: On unparseable response.
        """
        user_name = _resolve_user_name(memcell, sender_id)
        rendered = render_prompt(
            FORESIGHT_GENERATION_PROMPT,
            prompt,
            USER_ID=sender_id,
            USER_NAME=user_name,
            CONVERSATION_TEXT=_render_conversation(memcell),
        )

        start_time_fallback = _format_start_time_from_timestamp(memcell.items[0].timestamp)

        foresight_items = await _call_llm_for_foresight(self._llm, rendered, temperature=_FORESIGHT_TEMPERATURE)
        foresights = _build_foresights_from_items(
            foresight_items,
            memcell=memcell,
            sender_id=sender_id,
            start_time_fallback=start_time_fallback,
        )
        if len(foresights) > _FORESIGHT_MAX_COUNT:
            foresights = foresights[:_FORESIGHT_MAX_COUNT]
        elif 0 < len(foresights) < _FORESIGHT_MIN_COUNT:
            logger.warning("foresight count below soft floor: %d", len(foresights))
        return foresights

    extract = async_to_sync(aextract)


# ---------------------------------------------------------------------------
# LLM callsite — brace-balanced JSON extraction + 5-retry (mirror b150b32).
# ---------------------------------------------------------------------------


async def _call_llm_for_foresight(
    llm: LLMClient, rendered: str, *, temperature: float | None = None
) -> list[dict[str, Any]]:
    """Call LLM and return validated foresights list.

    Uses brace-balanced extraction because ``foresights`` is nested (list of dicts).

    Raises:
        ValueError: If no JSON found or ``foresights`` key is missing/not a list.
    """
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)], temperature=temperature)
    text = response.content
    json_str = _extract_json_object(text)
    data = json.loads(json_str)
    if "foresights" not in data:
        raise ValueError(f"foresights key missing from LLM response: {data!r}")
    items = data["foresights"]
    if not isinstance(items, list):
        raise ValueError(f"foresights must be a list, got {type(items).__name__}: {items!r}")  # noqa: TRY004
    return cast("list[dict[str, Any]]", items)


def _extract_json_object(text: str) -> str:
    """First balanced {{...}} block in text (brace-balanced parser for nested JSON)."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in foresight LLM response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Unbalanced JSON in foresight LLM response: {text[:200]!r}")


# Module-level helpers.


def _resolve_user_name(memcell: MemCell, sender_id: str) -> str:
    """Look up ``sender_id``'s ``sender_name`` from ChatMessage items; fall back to ``sender_id`` literal."""
    for m in chat_messages(memcell):
        if m.sender_id == sender_id and m.sender_name:
            return m.sender_name
    return sender_id


def _render_conversation(memcell: MemCell) -> str:
    """Render ChatMessage items as ``[YYYY-MM-DDTHH:MM:SSZ] speaker: content`` lines."""
    lines: list[str] = []
    for m in chat_messages(memcell):
        text = render_content(m.content)
        if not text:
            continue
        speaker = m.sender_name or m.sender_id
        time_str = format_message_timestamp(m.timestamp)
        lines.append(f"[{time_str}] {speaker}: {text}")
    return "\n".join(lines)


def _format_start_time_from_timestamp(timestamp_ms: int) -> str:
    """Render MemCell timestamp as ``YYYY-MM-DD`` for foresight start_time fallback."""
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime("%Y-%m-%d")


def _clean_date_string(date_str: object) -> str | None:
    """Normalize to ``YYYY-MM-DD`` — keep digits + hyphens, validate regex + constructibility. Returns ``None`` if invalid."""
    if not isinstance(date_str, str) or not date_str:
        return None
    cleaned = re.sub(r"[^\d\-]", "", date_str)
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", cleaned):
        return None
    try:
        year, month, day = map(int, cleaned.split("-"))
        datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None
    return cleaned


def _calculate_end_time_from_duration(start_time: str, duration_days: int) -> str | None:
    """Compute ``end_time = start_time + duration_days`` in ``YYYY-MM-DD``."""
    try:
        start_date = datetime.strptime(start_time, "%Y-%m-%d").replace(tzinfo=UTC)
        end_date = start_date + timedelta(days=duration_days)
    except ValueError:
        return None
    return end_date.strftime("%Y-%m-%d")


def _calculate_duration_days(start_time: str, end_time: str) -> int | None:
    """Compute ``end_time - start_time`` in days."""
    try:
        start_date = datetime.strptime(start_time, "%Y-%m-%d").replace(tzinfo=UTC)
        end_date = datetime.strptime(end_time, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError:
        return None
    return (end_date - start_date).days


def _build_foresights_from_items(
    items: list[dict[str, Any]],
    *,
    memcell: MemCell,
    sender_id: str,
    start_time_fallback: str,
) -> list[Foresight]:
    """Build Foresight entities from LLM response item list; apply date cleaning + mutual time computation."""
    out: list[Foresight] = []
    for item in items:
        if not isinstance(item, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        evidence = str(item.get("evidence") or "")

        item_start_time = _clean_date_string(item.get("start_time")) or start_time_fallback
        raw_end_time = item.get("end_time")
        item_end_time = _clean_date_string(raw_end_time) if raw_end_time else None
        raw_duration = item.get("duration_days")
        item_duration_days = int(raw_duration) if isinstance(raw_duration, (int, float)) else None

        # Mutual time computation
        if item_start_time:
            if item_duration_days is not None and not item_end_time:
                item_end_time = _calculate_end_time_from_duration(item_start_time, item_duration_days)
            elif item_end_time and item_duration_days is None:
                item_duration_days = _calculate_duration_days(item_start_time, item_end_time)

        out.append(
            Foresight(
                owner_id=sender_id,
                foresight=content,
                evidence=evidence,
                timestamp=memcell.timestamp,
                start_time=item_start_time,
                end_time=item_end_time,
                duration_days=item_duration_days,
            )
        )
    return out
