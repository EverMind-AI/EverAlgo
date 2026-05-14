"""Foresight extractor — opensource ``foresight_extractor.py`` algorithm in everalgo packaging.

Interface: stateless class with ``aextract(memcell) -> list[Foresight]``. Prompt ported verbatim from
opensource ``foresight_prompts.py``. The LLM emits a top-level JSON array of
``{content, evidence, start_time, end_time, duration_days}`` items.

Algorithm details ported from opensource ``foresight_extractor.py`` (locomo-benchmark, line 95-388):
    - 5-retry loop on JSON parse / empty-list failure (line 96-162); returns ``[]`` on exhaustion.
    - ``temperature=0.3`` explicit (line 120-121).
    - ``_clean_date_string`` (line 166-199): keep digits + hyphens, validate ``YYYY-MM-DD``, validate
      ``datetime`` constructibility.
    - Mutual time computation (line 338-387): start + duration → end; start + end → duration.
    - 4-10 count enforcement (line 138-147): at least 1 to pass retry, truncate at 10, warn if < 4.
    - ``start_time`` fallback from memcell.timestamp (line 326-336, ``%Y-%m-%d``).
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Foresight, MemCell
from everalgo.user_memory.prompts.en.foresight import FORESIGHT_GENERATION_PROMPT

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)

_MAX_LLM_RETRIES = 5
"""Maximum retries for LLM JSON / schema failures (matches opensource line 96)."""

_FORESIGHT_TEMPERATURE = 0.3
"""Temperature for foresight generation (matches opensource line 120)."""

_FORESIGHT_MAX_COUNT = 10
"""Hard upper bound on foresight items (matches opensource line 141-143)."""

_FORESIGHT_MIN_COUNT = 4
"""Soft lower bound — warn if generated count is below this (matches opensource line 144-147)."""


class ForesightExtractor:
    """Extract foresights (anticipated commitments) from a single MemCell.

    Stateless callable class. Algorithm mirrors opensource ``ForesightExtractor`` — 4-10 forward-looking
    associations grounded in the conversation, with 5-retry on JSON / empty-list failure,
    ``temperature=0.3``, date string cleaning, and mutual ``duration_days`` / ``end_time`` computation.

    Customize per call via ``llm=`` / ``prompt=`` arguments.
    """

    async def aextract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[Foresight]:
        """Ask the LLM to generate foresights for the MemCell.

        Returns an empty list after 5 retries on persistent JSON / empty-list failure (matches opensource
        line 159-161 — opensource also returns ``[]`` rather than raising). Infrastructure errors
        (LLMError / network / auth) propagate.
        """
        client = everalgo.llm.resolve(llm)
        owner_id = _derive_owner_id(memcell)
        user_name = _derive_user_name(memcell, owner_id)
        rendered = render_prompt(
            FORESIGHT_GENERATION_PROMPT,
            prompt,
            USER_ID=owner_id,
            USER_NAME=user_name,
            CONVERSATION_TEXT=_render_conversation(memcell),
        )

        start_time_fallback = _format_start_time_from_timestamp(memcell.timestamp)

        for attempt in range(_MAX_LLM_RETRIES):
            try:
                response = await client.chat(
                    messages=[LLMChatMessage(role="user", content=rendered)],
                    response_format={"type": "json_object"},
                    temperature=_FORESIGHT_TEMPERATURE,
                )
                foresights = _parse_and_build_foresights(
                    response.content,
                    memcell=memcell,
                    owner_id=owner_id,
                    start_time_fallback=start_time_fallback,
                )
                if not foresights:
                    raise ValueError("LLM returned empty foresight list")  # noqa: TRY301 — retry-loop semantics
                if len(foresights) > _FORESIGHT_MAX_COUNT:
                    foresights = foresights[:_FORESIGHT_MAX_COUNT]
                elif len(foresights) < _FORESIGHT_MIN_COUNT:
                    logger.warning("foresight count below soft floor: %d", len(foresights))
                return foresights  # noqa: TRY300 — opensource retry pattern: return out on success
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.warning("foresight retry %d/%d: %s", attempt + 1, _MAX_LLM_RETRIES, e)
                if attempt == _MAX_LLM_RETRIES - 1:
                    logger.exception("foresight generation failed after %d retries", _MAX_LLM_RETRIES)
                    return []
                continue

        return []  # unreachable, satisfies type checker

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helpers.


def _render_conversation(memcell: MemCell) -> str:
    """Render messages as ISO-timestamped ``[YYYY-MM-DDTHH:MM:SSZ] speaker: content`` lines."""
    lines: list[str] = []
    for m in memcell.messages:
        if not m.content:
            continue
        speaker = m.sender_name or m.role.value
        time_str = datetime.fromtimestamp(m.timestamp / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append(f"[{time_str}] {speaker}: {m.content}")
    return "\n".join(lines)


def _derive_owner_id(memcell: MemCell) -> str:
    if memcell.participants:
        return memcell.participants[0]
    for m in memcell.messages:
        if m.sender_id:
            return m.sender_id
    return "u_default"


def _derive_user_name(memcell: MemCell, owner_id: str) -> str:
    """Pick a human-readable user_name; falls back to owner_id when no sender_name is set."""
    for m in memcell.messages:
        if m.sender_name and (m.sender_id == owner_id or not m.sender_id):
            return m.sender_name
    return owner_id


def _format_start_time_from_timestamp(timestamp_ms: int) -> str:
    """Render MemCell timestamp as ``YYYY-MM-DD`` for foresight start_time fallback (opensource line 336)."""
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime("%Y-%m-%d")


def _clean_date_string(date_str: object) -> str | None:
    """Validate / normalize a date string to ``YYYY-MM-DD`` (opensource line 166-199).

    Keeps only digits and hyphens, requires exact ``YYYY-MM-DD`` regex match, and validates the date is
    actually constructible via :class:`datetime`. Returns ``None`` if invalid.
    """
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
    """Compute ``end_time = start_time + duration_days`` in ``YYYY-MM-DD`` (opensource line 338-362)."""
    try:
        start_date = datetime.strptime(start_time, "%Y-%m-%d").replace(tzinfo=UTC)
        end_date = start_date + timedelta(days=duration_days)
    except ValueError:
        return None
    return end_date.strftime("%Y-%m-%d")


def _calculate_duration_days(start_time: str, end_time: str) -> int | None:
    """Compute ``end_time - start_time`` in days (opensource line 364-387)."""
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
    owner_id: str,
    start_time_fallback: str,
) -> list[Foresight]:
    """Parse opensource foresight payload + apply date cleaning + mutual time computation.

    Accepts top-level JSON array (opensource canonical) OR ``{"foresights": [...]}`` wrapped form
    (json_object mode requires top-level object).
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

        # Mutual time computation (opensource line 258-269)
        if item_start_time:
            if item_duration_days is not None and not item_end_time:
                item_end_time = _calculate_end_time_from_duration(item_start_time, item_duration_days)
            elif item_end_time and item_duration_days is None:
                item_duration_days = _calculate_duration_days(item_start_time, item_end_time)

        out.append(
            Foresight(
                id=f"fs_{uuid.uuid4().hex[:12]}",
                owner_id=owner_id,
                foresight=content,
                evidence=evidence,
                timestamp=memcell.timestamp,
                start_time=item_start_time,
                end_time=item_end_time,
                duration_days=item_duration_days,
                parent_type="memcell",
                parent_id=memcell.event_id or "",
            )
        )
    return out


def _parse_llm_response(raw: str) -> object:
    r"""Parse LLM response using opensource multi-strategy (line 222-237).

    Strategies: ``\`\`\`json`` code block, then direct ``json.loads``. Raises
    :class:`json.JSONDecodeError` if both fail (caller catches and retries).
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
