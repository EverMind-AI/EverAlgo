"""Episode extractor — per-sender fan-out aligned with new-release opensource.

Algorithm shape
---------------
* When ``memcell.sender_ids`` is non-empty, the extractor iterates senders **serially** and issues one
  LLM call per sender with ``EPISODE_GENERATION_PROMPT`` (personal mode), so the returned
  ``list[Episode]`` carries one entry per sender with ``owner_id = sender_id``. This mirrors opensource
  ``memory_manager`` calling ``EpisodeMemoryExtractor.extract_memory`` once per user, but the fan-out is
  internalised here to keep EverAlgo's stateless ``aextract(memcell) -> list[Episode]`` contract.
* When ``memcell.sender_ids`` is empty / ``None``, the extractor falls back to one ``GROUP``-mode LLM
  call with ``GROUP_EPISODE_GENERATION_PROMPT`` (opensource ``use_group_prompt=True`` path); the lone
  Episode's ``owner_id`` is derived from ``participants`` / message ``sender_id``.

Inherited from opensource ``episode_memory_extractor.py`` (locomo-benchmark, line 262-303):
    - 5-retry loop on JSON parse / schema validation failure (raises after exhaustion).
    - Multi-strategy JSON parse: ```json``` code block → regex ``{...title...content...}`` → direct
      ``json.loads``.
    - ``title`` + ``content`` non-empty validation; ``summary`` fallback = ``content[:200]``.
    - ``user_name`` resolved from ``sender_id → sender_name`` map; missing → ``sender_id`` string
      (opensource ``participants_name_map.get(user_id, user_id)`` at line 261).

Caller override: pass ``prompt=`` per call. The override receives every formattable field
(``conversation_start_time``, ``conversation``, ``custom_instructions``, ``user_name``); placeholders the
template does not reference are silently dropped by :py:meth:`str.format`.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Episode, MemCell
from everalgo.user_memory.prompts.en.episode import (
    DEFAULT_CUSTOM_INSTRUCTIONS,
    EPISODE_GENERATION_PROMPT,
    GROUP_EPISODE_GENERATION_PROMPT,
)

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


_MAX_LLM_RETRIES = 5
"""Maximum retries for LLM JSON / schema failures (matches opensource line 264)."""


class EpisodeExtractor:
    """Extract per-sender Episode memories from a single MemCell.

    Stateless callable class. Returns ``list[Episode]`` with one entry per ``memcell.sender_ids`` element
    (personal mode); if ``sender_ids`` is empty, falls back to a single group-mode Episode.

    Customisation per call:
        - ``llm=`` — override the resolved LLM client.
        - ``prompt=`` — replace the default template (personal mode by default; group mode in the
          empty-sender_ids fallback). The override must remain compatible with ``str.format`` and any
          placeholders it references must be among ``conversation_start_time`` / ``conversation`` /
          ``custom_instructions`` / ``user_name``.
        - ``custom_instructions=`` — replace the default ``DEFAULT_CUSTOM_INSTRUCTIONS`` block.
    """

    async def aextract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
        custom_instructions: str | None = None,
    ) -> list[Episode]:
        """Extract one Episode per sender (or one group-mode Episode when ``sender_ids`` is empty).

        Raises ``RuntimeError`` after 5 retries on persistent JSON parse / schema failure
        (matches opensource line 298). Infrastructure errors (LLMError / network / auth) propagate.
        """
        client = everalgo.llm.resolve(llm)
        custom_instr = custom_instructions or DEFAULT_CUSTOM_INSTRUCTIONS
        conv_start = _format_conversation_start_time(memcell.timestamp)
        conversation = _render_conversation(memcell)

        sender_ids = memcell.sender_ids or []

        if not sender_ids:
            rendered = render_prompt(
                GROUP_EPISODE_GENERATION_PROMPT,
                prompt,
                conversation_start_time=conv_start,
                conversation=conversation,
                custom_instructions=custom_instr,
            )
            data = await _call_llm_with_retry(client, rendered)
            return [_build_episode(data, owner_id=_derive_owner_id(memcell), memcell=memcell)]

        name_map = _build_sender_name_map(memcell)
        episodes: list[Episode] = []
        for sender_id in sender_ids:
            user_name = name_map.get(sender_id, sender_id)
            rendered = render_prompt(
                EPISODE_GENERATION_PROMPT,
                prompt,
                conversation_start_time=conv_start,
                conversation=conversation,
                custom_instructions=custom_instr,
                user_name=user_name,
            )
            data = await _call_llm_with_retry(client, rendered)
            episodes.append(_build_episode(data, owner_id=sender_id, memcell=memcell))
        return episodes

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helpers.


async def _call_llm_with_retry(client: LLMClient, rendered: str) -> dict[str, Any]:
    """Run the opensource 5-retry / 3-tier-parse / title+content validation loop for one LLM call.

    Returns the parsed ``{"title": str, "content": str, "summary"?: str}`` dict on success.
    Raises :class:`RuntimeError` if all retries are exhausted (matches opensource line 298).
    """
    last_error: Exception | None = None
    for attempt in range(_MAX_LLM_RETRIES):
        try:
            response = await client.chat(
                messages=[LLMChatMessage(role="user", content=rendered)],
                response_format={"type": "json_object"},
            )
            parsed = _parse_llm_response(response.content)
            if "title" not in parsed or not parsed["title"]:
                raise ValueError("LLM response missing title field")  # noqa: TRY301 — retry-loop semantics
            if "content" not in parsed or not parsed["content"]:
                raise ValueError("LLM response missing content field")  # noqa: TRY301
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            last_error = e
            if attempt == _MAX_LLM_RETRIES - 1:
                raise RuntimeError(
                    f"EpisodeExtractor: all {_MAX_LLM_RETRIES} retries exhausted "
                    f"(JSON parse / schema mismatch). Last error: {last_error}"
                ) from last_error
            continue
        else:
            return parsed
    # Loop above either returns or raises; this line is unreachable but satisfies type-checkers.
    raise RuntimeError("EpisodeExtractor: retry loop exited without resolution")  # pragma: no cover


def _build_episode(data: dict[str, Any], *, owner_id: str, memcell: MemCell) -> Episode:
    """Assemble an :class:`Episode` from the parsed LLM payload and memcell metadata."""
    title = cast("str", data["title"])
    content = cast("str", data["content"])
    summary_raw = data.get("summary")
    summary = summary_raw if isinstance(summary_raw, str) and summary_raw.strip() else content[:200]
    return Episode.model_validate(
        {
            "id": f"ep_{uuid.uuid4().hex[:12]}",
            "owner_id": owner_id,
            "episode": content,
            "subject": title,
            "timestamp": memcell.timestamp,
            "parent_type": "memcell",
            "parent_id": memcell.event_id or "",
            "summary": summary,  # opensource emits / fallbacks to content[:200]; preserved via extra='allow'
        }
    )


def _build_sender_name_map(memcell: MemCell) -> dict[str, str]:
    """Build ``sender_id -> sender_name`` from messages (opensource ``get_sender_name_map`` + override)."""
    name_map: dict[str, str] = {}
    for m in memcell.messages:
        if m.sender_id and m.sender_name:
            name_map[m.sender_id] = m.sender_name
    return name_map


def _format_conversation_start_time(timestamp_ms: int) -> str:
    """Render the MemCell timestamp in opensource example form ``March 14, 2024 (Thursday) at 3:00 PM UTC``."""
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    return dt.strftime("%B %-d, %Y (%A) at %-I:%M %p UTC")


def _render_conversation(memcell: MemCell) -> str:
    """Render messages as ``[YYYY-MM-DD HH:MM:SS] speaker: content`` lines (opensource convention)."""
    lines: list[str] = []
    for m in memcell.messages:
        if not m.content:
            continue
        speaker = m.sender_name or m.role.value
        time_str = datetime.fromtimestamp(m.timestamp / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"[{time_str}] {speaker}: {m.content}")
    return "\n".join(lines)


def _parse_llm_response(raw: str) -> dict[str, Any]:
    """Parse LLM response using opensource multi-strategy (line 270-285).

    Strategies (in order):
        1. ```json ... ``` code block.
        2. Regex ``{...title...content...}`` for any embedded object.
        3. Direct ``json.loads`` on the whole response.

    Raises :class:`json.JSONDecodeError` if all strategies fail (caller catches and retries).
    """
    if "```json" in raw:
        start = raw.find("```json") + 7
        end = raw.find("```", start)
        if end > start:
            try:
                return cast("dict[str, Any]", json.loads(raw[start:end].strip()))
            except json.JSONDecodeError:
                pass
    match = re.search(r'\{[^{}]*"title"[^{}]*"content"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            return cast("dict[str, Any]", json.loads(match.group()))
        except json.JSONDecodeError:
            pass
    return cast("dict[str, Any]", json.loads(raw))


def _derive_owner_id(memcell: MemCell) -> str:
    """Pick a stable owner_id for the group-mode fallback (no sender_ids)."""
    if memcell.participants:
        return memcell.participants[0]
    for m in memcell.messages:
        if m.sender_id:
            return m.sender_id
    return "u_default"
