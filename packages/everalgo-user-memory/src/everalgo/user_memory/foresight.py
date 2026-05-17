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

        start_time_fallback = _format_start_time_from_timestamp(memcell.timestamp)

        response = await self._llm.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
            temperature=_FORESIGHT_TEMPERATURE,
        )
        foresights = _parse_and_build_foresights(
            response.content,
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


def _parse_and_build_foresights(
    raw: str,
    *,
    memcell: MemCell,
    sender_id: str,
    start_time_fallback: str,
) -> list[Foresight]:
    """Parse LLM foresight payload + apply date cleaning + mutual time computation.

    Accepts top-level JSON array OR ``{"foresights": [...]}`` wrapped form.
    """
    data = _parse_llm_response(raw)

    items: list[Any]
    if isinstance(data, list):
        items = cast("list[Any]", data)  # type: ignore[redundant-cast]
    elif isinstance(data, dict):
        wrapped = cast("dict[str, Any]", data).get("foresights")
        if isinstance(wrapped, list):
            items = cast("list[Any]", wrapped)  # type: ignore[redundant-cast]
        else:
            return []
    else:
        return []

    out: list[Foresight] = []
    for raw_item in items:
        if not isinstance(raw_item, dict):
            continue
        item = cast("dict[str, Any]", raw_item)
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        evidence_raw = item.get("evidence", "")
        evidence = evidence_raw if isinstance(evidence_raw, str) else ""

        item_start_time = _clean_date_string(item.get("start_time")) or start_time_fallback
        item_end_time = _clean_date_string(item.get("end_time"))
        item_duration_days = item.get("duration_days") if isinstance(item.get("duration_days"), int) else None

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


def _parse_llm_response(raw: str) -> object:
    r"""Parse LLM JSON response: `` ```json `` fence first, then direct ``json.loads``.

    Raises:
        json.JSONDecodeError: If both strategies fail.
    """
    if "```json" in raw:
        start = raw.find("```json") + 7
        end = raw.find("```", start)
        if end > start:
            try:
                return json.loads(raw[start:end].strip())
            except json.JSONDecodeError:
                pass
    return json.loads(raw)
