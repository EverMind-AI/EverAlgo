"""Extract a single Episode for one sender from a MemCell."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

from everalgo.llm.format import format_message_timestamp, format_natural_language_time
from everalgo.llm.parse import parse_llm_json_object
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Episode, MemCell
from everalgo.user_memory._render import chat_messages, render_content
from everalgo.user_memory.prompts.en.episode import (
    DEFAULT_CUSTOM_INSTRUCTIONS,
    EPISODE_GENERATION_PROMPT,
    USER_EPISODE_GENERATION_PROMPT,
)

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


class EpisodeExtractor:
    """Extract one Episode for a given sender from a MemCell.

    Non-ChatMessage items in memcell.items are silently skipped (agent → user-memory contract).
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        memcell: MemCell,
        *,
        sender_id: str | None,
        prompt: str | None = None,
        custom_instructions: str | None = None,
    ) -> Episode:
        """Extract one Episode from ``memcell``.

        Args:
            memcell: Source slice from boundary detection.
            sender_id: Specific chat sender to centre the episode on (uses USER_EPISODE_GENERATION_PROMPT);
                pass ``None`` to extract one whole-memcell generic episode (uses EPISODE_GENERATION_PROMPT)
                — cheaper than per-user fan-out.
            prompt: Prompt override; ``None`` uses the bundled default.
            custom_instructions: Extra instruction block appended to the system prompt; ``None`` uses the default.

        Raises:
            LLMError: From the LLM call.
            json.JSONDecodeError: If all parse strategies fail.
            ValueError: If the LLM response is missing a non-empty ``title`` or ``content``.
        """
        custom_instr = custom_instructions or DEFAULT_CUSTOM_INSTRUCTIONS
        conv_start = _format_conversation_start_time(memcell.timestamp)
        conversation = _render_conversation(memcell)

        if sender_id is None:
            rendered = render_prompt(
                EPISODE_GENERATION_PROMPT,
                prompt,
                conversation_start_time=conv_start,
                conversation=conversation,
                custom_instructions=custom_instr,
            )
        else:
            user_name = _resolve_user_name(memcell, sender_id)
            rendered = render_prompt(
                USER_EPISODE_GENERATION_PROMPT,
                prompt,
                conversation_start_time=conv_start,
                conversation=conversation,
                custom_instructions=custom_instr,
                user_name=user_name,
            )

        response = await self._llm.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
        )
        data = _parse_llm_response(response.content)
        if "title" not in data or not data["title"]:
            raise ValueError("LLM response missing title field")
        if "content" not in data or not data["content"]:
            raise ValueError("LLM response missing content field")

        return _build_episode(data, sender_id=sender_id, memcell=memcell)

    extract = async_to_sync(aextract)


# Module-level helpers.


def _resolve_user_name(memcell: MemCell, sender_id: str) -> str:
    """Look up ``sender_id``'s ``sender_name`` from ChatMessage items; fall back to ``sender_id`` literal."""
    for m in chat_messages(memcell):
        if m.sender_id == sender_id and m.sender_name:
            return m.sender_name
    return sender_id


def _build_episode(data: dict[str, Any], *, sender_id: str | None, memcell: MemCell) -> Episode:
    """Assemble an :class:`Episode` from the parsed LLM payload and memcell metadata."""
    title = cast("str", data["title"])
    content = cast("str", data["content"])
    summary_raw = data.get("summary")
    summary = summary_raw if isinstance(summary_raw, str) and summary_raw.strip() else content[:200]
    return Episode.model_validate(
        {
            "owner_id": sender_id,
            "episode": content,
            "subject": title,
            "timestamp": memcell.timestamp,
            "summary": summary,  # preserved via extra='allow' without a schema bump
        }
    )


def _format_conversation_start_time(timestamp_ms: int) -> str:
    """Render the MemCell timestamp as ``March 14, 2024 (Thursday) at 3:00 PM UTC``."""
    return format_natural_language_time(timestamp_ms)


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


def _parse_llm_response(raw: str) -> dict[str, Any]:
    """Parse LLM JSON response.

    Schema-specific regex is tried first as a main-path optimisation (targets in-prose
    ``{"title": ..., "content": ...}`` fragments); falls back to the shared three-tier parser
    (fence → direct loads → outermost braces).

    Raises:
        ValueError: If all strategies fail.
    """
    match = re.search(r'\{[^{}]*"title"[^{}]*"content"[^{}]*\}', raw, re.DOTALL)
    if match:
        try:
            return cast("dict[str, Any]", json.loads(match.group()))
        except json.JSONDecodeError:
            pass
    return parse_llm_json_object(raw)
