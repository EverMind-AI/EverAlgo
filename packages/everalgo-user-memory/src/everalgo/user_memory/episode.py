"""Extract a single Episode for one sender from a MemCell."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Episode, MemCell
from everalgo.user_memory._language import OutputLanguage, build_language_rule
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
        output_language: OutputLanguage | str | None = None,
    ) -> Episode:
        """Extract one Episode from ``memcell``.

        Args:
            memcell: Source slice from boundary detection.
            sender_id: Specific chat sender to centre the episode on (uses USER_EPISODE_GENERATION_PROMPT);
                pass ``None`` to extract one whole-memcell generic episode (uses EPISODE_GENERATION_PROMPT)
                — cheaper than per-user fan-out.
            prompt: Prompt override; ``None`` uses the bundled default. Code stores ``content`` verbatim,
                so the times a reader sees are the ones the prompt made the model write. A custom prompt
                therefore owns that contract: it must require an absolute, ``UTC``-labelled time for the
                events it narrates, or the stored episode carries no time a downstream LLM can read
                (``Episode.timestamp`` holds ms since epoch and is not rendered into answer context).
            custom_instructions: Extra instruction block appended to the system prompt; ``None`` uses the default.
            output_language: Language to write the episode in, as an :class:`OutputLanguage` member or
                equivalent string in any casing. Naming one removes the decision from the model, which measured zero wrong
                languages; leaving it ``None`` asks the model to follow the participants, which costs roughly
                one episode in thirteen and fails towards Chinese. See
                ``prompts/en/_language.py`` for the measurements.

        Raises:
            LLMError: From the LLM call.
            ValueError: If the LLM returns no parsed structured output, or ``output_language``
                names no supported language.
        """
        custom_instr = custom_instructions or DEFAULT_CUSTOM_INSTRUCTIONS
        conv_start = _format_prompt_time(memcell.items[0].timestamp)
        conversation = _render_conversation(memcell)
        language_rule = build_language_rule(output_language)

        if sender_id is None:
            rendered = render_prompt(
                EPISODE_GENERATION_PROMPT,
                prompt,
                conversation_start_time=conv_start,
                conversation=conversation,
                custom_instructions=custom_instr,
                language_rule=language_rule,
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
                language_rule=language_rule,
            )

        data = await _call_llm_for_episode(self._llm, rendered)
        return _build_episode(data, sender_id=sender_id, memcell=memcell)

    extract = async_to_sync(aextract)


# ---------------------------------------------------------------------------
# LLM callsite — regex JSON extraction, no retry (the client documents none either; retry is a
# caller concern per everalgo.llm.providers.openai_compat).
# ---------------------------------------------------------------------------


async def _call_llm_for_episode(llm: LLMClient, rendered: str) -> dict[str, Any]:
    """Call LLM and return validated episode dict.

    Uses brace-balanced extraction because the free-text fields may contain nested strings with
    punctuation. Raises ``ValueError`` on missing JSON, a missing required key, or an empty
    ``content`` / ``summary``. All three keys are required: ``summary`` reaches the caller as a
    display preview, and there is no honest value to substitute for one the model did not write.
    """
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    json_str = _extract_json_object(text)
    data: dict[str, Any] = json.loads(json_str)
    missing = [key for key in ("title", "content", "summary") if key not in data]
    if missing:
        raise ValueError(f"Episode LLM response missing required keys {missing}: {data!r}")
    for key in ("content", "summary"):
        if not str(data[key]).strip():
            raise ValueError(f"Episode LLM response has empty {key}: {data!r}")
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

    The body carries the times the model wrote, with no code-built prefix in front of them. The prompt
    pins their format (24-hour, ``UTC``-labelled), which is what the prefix was introduced to guarantee;
    a prefix on top of that restated the opening message's time a second time, since the prefix's value
    was ``items[0].timestamp`` and the narrative's own timeline starts at the same moment.
    """
    return Episode.model_validate(
        {
            "owner_id": sender_id,
            "episode": str(data["content"]),
            "subject": str(data["title"]),
            "summary": str(data["summary"]),
            "timestamp": memcell.timestamp,
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
