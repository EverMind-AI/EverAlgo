"""Extract a single Episode for one sender from a MemCell."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from asgiref.sync import async_to_sync

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Episode, MemCell
from everalgo.user_memory._language import (
    SOURCE_TEXT_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
from everalgo.user_memory._render import chat_messages, render_content
from everalgo.user_memory._width import ascii_width
from everalgo.user_memory.prompts.en.episode import (
    DEFAULT_CUSTOM_INSTRUCTIONS,
    EPISODE_GENERATION_PROMPT,
    SUMMARY_COMPRESS_PROMPT,
    USER_EPISODE_GENERATION_PROMPT,
)

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)

# Hard cap on the stored ``summary``, in ASCII-equivalent units. The 400-unit ceiling accommodates the prompt's
# ~50-English-word target with headroom while still catching pathological outputs. Production
# 2026-08: a qwen3-4b finetune restated the whole ``content`` as ``summary`` on ~22% of
# episodes (max 9038 chars) — the prompt alone does not hold the line, so the extractor does.
_SUMMARY_WIDTH_CAP = 400

# Input guard for the compress call; it only trims pathological model outputs.
_SUMMARY_COMPRESS_INPUT_MAX_CHARS = 8000

_ELLIPSIS = "\u2026"


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

        The returned ``summary`` is guaranteed no wider than ``_SUMMARY_WIDTH_CAP`` ASCII-equivalent
        units: an over-cap one is repaired by one compress call over that ``summary``, then by
        sentence-boundary truncation — never by failing the extraction (see
        ``_ensure_summary_within_cap``).
        """
        custom_instr = custom_instructions or DEFAULT_CUSTOM_INSTRUCTIONS
        conv_start = _format_prompt_time(memcell.items[0].timestamp)
        conversation = _render_conversation(memcell)
        language_rule = build_language_rule(output_language)
        logger.info(
            "extracting episode: %d items, %d rendered chars, template=%s, output_language=%s",
            len(memcell.items),
            len(conversation),
            "user-centred" if sender_id is not None else "generic",
            output_language,
        )

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
        compress_rule = language_rule if output_language is not None else SOURCE_TEXT_LANGUAGE_RULE
        data["summary"] = await _ensure_summary_within_cap(
            self._llm,
            summary=str(data["summary"]),
            language_rule=compress_rule,
        )
        episode = _build_episode(data, sender_id=sender_id, memcell=memcell)
        logger.info(
            "episode extracted: content %d units, summary %d units",
            ascii_width(episode.episode),
            ascii_width(str(getattr(episode, "summary", ""))),
        )
        return episode

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


# ---------------------------------------------------------------------------
# Summary width guard — three tiers, never raising: the summary is a display preview,
# and the caller's failure model retries the WHOLE extraction on an exception and drops
# the episode after the last attempt. Failing the main product over a cosmetic field is
# the one outcome this guard exists to prevent, so every tier degrades instead of raising.
# ---------------------------------------------------------------------------


async def _ensure_summary_within_cap(llm: LLMClient, *, summary: str, language_rule: str) -> str:
    """Return a summary no wider than ``_SUMMARY_WIDTH_CAP``.

    Tier 1: the extractor's own summary, when compliant (the normal path, free).
    Tier 2: one cheap compress call over the over-wide ``summary`` itself.
    Tier 3: sentence-boundary truncation, so an increment is compliant by construction
    even when the model refuses to be. The truncated preview ends with an ellipsis and
    loses coverage, never correctness: every kept sentence is one the model wrote.
    """
    original_width = ascii_width(summary)
    if original_width <= _SUMMARY_WIDTH_CAP:
        return summary
    # Tier boundaries are logged on purpose: the tier-2 rate IS the production model's
    # violation rate, and repaired-vs-truncated is the repair's success rate — neither
    # is observable anywhere else.
    logger.warning(
        "episode summary over cap (%d > %d units), attempting compress repair",
        original_width,
        _SUMMARY_WIDTH_CAP,
    )
    rewritten = await _compress_summary(llm, summary=summary, language_rule=language_rule)
    if rewritten is not None and ascii_width(rewritten) <= _SUMMARY_WIDTH_CAP:
        logger.info(
            "episode summary repaired by compress: %d -> %d units",
            original_width,
            ascii_width(rewritten),
        )
        return rewritten
    if rewritten is not None and ascii_width(rewritten) < ascii_width(summary):
        summary = rewritten
    logger.warning(
        "episode summary still %d units wide after compression, truncating to %d",
        ascii_width(summary),
        _SUMMARY_WIDTH_CAP,
    )
    return _truncate_at_sentence_boundary(summary, _SUMMARY_WIDTH_CAP)


async def _compress_summary(llm: LLMClient, *, summary: str, language_rule: str) -> str | None:
    """One repair call: shorten the over-wide ``summary``. ``None`` on any failure.

    Plain-text output (no JSON) keeps the failure surface small; the broad except is the
    point — no repair failure may ever cost the episode itself.
    """
    prompt = SUMMARY_COMPRESS_PROMPT.format(
        language_rule=language_rule,
        summary_text=summary[:_SUMMARY_COMPRESS_INPUT_MAX_CHARS],
    )
    try:
        response = await llm.chat(messages=[LLMChatMessage(role="user", content=prompt)])
        return response.content.strip().strip('"') or None
    except Exception as exc:
        logger.warning("summary compress call failed: %s", exc)
        return None


_SENTENCE_TERMINATORS = ".\u3002\uff0e\uff01\uff1f!?"


def _truncate_at_sentence_boundary(text: str, cap: int) -> str:
    """Longest prefix of whole sentences within ``cap`` width, ellipsis-terminated.

    The ellipsis REPLACES the final sentence terminator (never sits after it), so the
    cut reads as one mark, not two. Falls back to a bare width cut only when even the first sentence exceeds the budget
    (a sentence-less run); mid-word is still better than over-cap there, and the ellipsis
    marks the cut either way.
    """
    budget = max(cap - ascii_width(_ELLIPSIS), 1)
    width = 0
    last_sentence_end = 0
    hard_cut = len(text)
    for i, ch in enumerate(text):
        width += ascii_width(ch)
        if width > budget:
            hard_cut = i
            break
        if ch in _SENTENCE_TERMINATORS:
            last_sentence_end = i + 1
    if width <= budget:
        return text
    cut = last_sentence_end or hard_cut
    return text[:cut].rstrip().rstrip(_SENTENCE_TERMINATORS) + _ELLIPSIS


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
