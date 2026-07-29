"""Extract a single Episode for one sender from a MemCell."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync

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
            prompt: Prompt override; ``None`` uses the bundled default. Code unconditionally prepends a
                UTC timestamp prefix to ``content`` regardless of which prompt produced it (see
                ``_build_episode``), so a custom prompt need not produce one. The prefix is metadata
                about the slice, while times inside the narrative belong to the events; the two state
                the same moment when the first event is the opening message, which is expected.
            custom_instructions: Extra instruction block appended to the system prompt; ``None`` uses the default.

        Raises:
            LLMError: From the LLM call.
            ValueError: If the LLM returns no parsed structured output.
        """
        custom_instr = custom_instructions or DEFAULT_CUSTOM_INSTRUCTIONS
        conv_start = _format_prompt_time(memcell.items[0].timestamp)
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

        data = await _call_llm_for_episode(self._llm, rendered)
        return _build_episode(data, sender_id=sender_id, memcell=memcell)

    extract = async_to_sync(aextract)


# ---------------------------------------------------------------------------
# LLM callsite — regex JSON extraction + 5-retry (mirror b150b32 boundary pattern).
# ---------------------------------------------------------------------------


async def _call_llm_for_episode(llm: LLMClient, rendered: str) -> dict[str, Any]:
    """Call LLM and return validated episode dict.

    Uses brace-balanced extraction because the ``summary`` field may contain nested strings with
    punctuation. Raises ``ValueError`` on missing JSON or missing required keys.
    """
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    json_str = _extract_json_object(text)
    data: dict[str, Any] = json.loads(json_str)
    if "title" not in data or "content" not in data:
        raise ValueError(f"Episode LLM response missing required keys: {data!r}")
    if not str(data["content"]).strip():
        raise ValueError(f"Episode LLM response has empty content: {data!r}")
    return data


def _extract_json_object(text: str) -> str:
    """First balanced {{...}} block in text (brace-balanced parser for nested/complex JSON)."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in episode LLM response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Unbalanced JSON in episode LLM response: {text[:200]!r}")


# Module-level helpers.


def _resolve_user_name(memcell: MemCell, sender_id: str) -> str:
    """Look up ``sender_id``'s ``sender_name`` from ChatMessage items; fall back to ``sender_id`` literal."""
    for m in chat_messages(memcell):
        if m.sender_id == sender_id and m.sender_name:
            return m.sender_name
    return sender_id


def _build_episode(data: dict[str, Any], *, sender_id: str | None, memcell: MemCell) -> Episode:
    """Assemble an :class:`Episode` from the parsed LLM payload and memcell metadata.

    The body is prefixed with a code-built UTC timestamp rather than one written by the LLM: the model
    used to translate the injected anchor into the output language, and that translation step is what
    produced drifting formats and mistranslated AM/PM markers. ``answer``-stage consumers see only
    ``subject`` and ``episode``, so the body has to carry the timestamp itself.
    """
    title = str(data["title"])
    body = f"{_format_episode_prefix(memcell.items[0].timestamp)} — {data['content']!s}"
    summary_raw = data.get("summary")
    summary = summary_raw if isinstance(summary_raw, str) and summary_raw.strip() else body[:200]
    return Episode.model_validate(
        {
            "owner_id": sender_id,
            "episode": body,
            "subject": title,
            "timestamp": memcell.timestamp,
            "summary": summary,  # preserved via extra='allow' without a schema bump
        }
    )


def _format_prompt_time(timestamp_ms: int) -> str:
    """Render a timestamp for injection into the prompt, e.g. ``2026-05-29 12:25 UTC (Friday)``.

    24-hour clock: a 12-hour clock leaves noon and midnight ambiguous, and the LLM has been observed
    mistranslating ``12:21 PM`` into a Chinese "before noon" phrasing. The weekday label is load-bearing
    for relative-time reasoning ("last Friday") — see user-memory 0.3.1, which added it to fix
    off-by-one-week errors. Do not drop it.
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    return f"{dt:%Y-%m-%d %H:%M} UTC ({dt:%A})"


def _format_episode_prefix(timestamp_ms: int) -> str:
    """Render the timestamp prefix that code prepends to the episode body, e.g. ``2026-05-29 12:25 UTC``.

    Pure ASCII, so a narrative in any language can carry it verbatim and the LLM never has to translate
    it. No weekday: it would have to be fixed English and reads wrong inside a non-English narrative,
    while the UI can derive it from ``Episode.timestamp``.
    """
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    return f"{dt:%Y-%m-%d %H:%M} UTC"


def _render_conversation(memcell: MemCell) -> str:
    """Render ChatMessage items as pseudo-JSON per message.

    Each message becomes a pseudo-JSON object with ~16-space indent — field names quoted,
    values unquoted (the LLM tolerates the not-strictly-JSON syntax). The no-timestamp
    branch is retained for callers that omit timestamps.
    """
    lines: list[str] = []
    for m in chat_messages(memcell):
        text = render_content(m.content)
        if not text:
            continue
        speaker = m.sender_name or m.sender_id
        timestamp = m.timestamp
        if timestamp:
            lines.append(
                f"""
                {{
                    "timestamp": {_format_prompt_time(timestamp)},
                    "speaker": {speaker},
                    "content": {text}
                }}"""
            )
        else:
            lines.append(
                f"""
                {{
                    "speaker": {speaker},
                    "content": {text}
                }}"""
            )
    return "\n".join(lines)
