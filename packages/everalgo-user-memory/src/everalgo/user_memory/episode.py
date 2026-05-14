"""Episode extractor — opensource ``episode_memory_extractor.py`` algorithm in everalgo packaging.

Interface preserved per design conventions: stateless class, ``aextract(memcell) -> list[Episode]``, 3-layer
LLM injection. Prompt and output schema verbatim from opensource ``episode_mem_prompts.py``
(``GROUP_EPISODE_GENERATION_PROMPT`` as default; ``EPISODE_GENERATION_PROMPT`` personal mode available via
the ``prompt=`` per-call override).

Algorithm details ported from opensource ``episode_memory_extractor.py`` (locomo-benchmark, line 262-303):
    - 5-retry loop on JSON parse / schema validation failure (raises Exception after exhaustion).
    - Multi-strategy JSON parse: ```json``` code block → regex ``{...title...content...}`` → direct ``json.loads``.
    - title + content non-empty validation; summary fallback = content[:200].
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
    GROUP_EPISODE_GENERATION_PROMPT,
)

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


_MAX_LLM_RETRIES = 5
"""Maximum retries for LLM JSON / schema failures (matches opensource line 264)."""


class EpisodeExtractor:
    """Extract Episode memories from a single MemCell.

    Stateless callable class. Per opensource ``episode_memory_extractor.py``, the operator emits a single
    Episode per MemCell (LLM contract: ``{"title": str, "content": str, "summary"?: str}``). We wrap that
    in a list to preserve the everalgo ``-> list[Episode]`` shape.

    Customize per call via ``llm=`` / ``prompt=`` / ``custom_instructions=`` arguments. To use the
    personal-mode prompt (which requires ``{user_name}``), pass ``prompt=EPISODE_GENERATION_PROMPT`` and
    provide ``user_name`` via a caller-side prompt pre-substitution.
    """

    async def aextract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
        custom_instructions: str | None = None,
    ) -> list[Episode]:
        """Ask the LLM to extract a single Episode from the MemCell.

        Raises ``RuntimeError`` after 5 retries on persistent JSON parse / schema failure (matches
        opensource line 298). Infrastructure errors (LLMError / network / auth) propagate.
        """
        client = everalgo.llm.resolve(llm)
        rendered = render_prompt(
            GROUP_EPISODE_GENERATION_PROMPT,
            prompt,
            conversation_start_time=_format_conversation_start_time(memcell.timestamp),
            conversation=_render_conversation(memcell),
            custom_instructions=custom_instructions or DEFAULT_CUSTOM_INSTRUCTIONS,
        )

        data: dict[str, Any] | None = None
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
                data = parsed
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                last_error = e
                if attempt == _MAX_LLM_RETRIES - 1:
                    raise RuntimeError(
                        f"EpisodeExtractor: all {_MAX_LLM_RETRIES} retries exhausted "
                        f"(JSON parse / schema mismatch). Last error: {last_error}"
                    ) from last_error
                continue

        assert data is not None  # guaranteed by the loop above (break only on parsed != None)
        owner_id = _derive_owner_id(memcell)
        title = cast("str", data["title"])
        content = cast("str", data["content"])
        summary_raw = data.get("summary")
        summary = summary_raw if isinstance(summary_raw, str) and summary_raw.strip() else content[:200]
        episode = Episode.model_validate(
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
        return [episode]

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""


# Module-level helpers.


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
    """Pick a stable owner_id from MemCell participants; fall back to ``u_default``."""
    if memcell.participants:
        return memcell.participants[0]
    for m in memcell.messages:
        if m.sender_id:
            return m.sender_id
    return "u_default"
