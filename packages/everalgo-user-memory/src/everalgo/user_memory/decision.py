"""Extract committed decisions from a conversation slice."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

from everalgo.llm.format import format_message_timestamp
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Decision, MemCell
from everalgo.user_memory._language import OutputLanguage, build_language_rule
from everalgo.user_memory._render import chat_messages, render_content
from everalgo.user_memory.prompts.en.decision import DECISION_GENERATION_PROMPT

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


_DECISION_TEMPERATURE = 0.3
_DECISION_MAX_COUNT = 10


class DecisionExtractor:
    """Extract zero or more decisions from one MemCell.

    Runs once for the whole slice: there is no ``sender_id``, and every ``Decision.owner_id`` is
    ``None``. The caller binds owners later. Non-ChatMessage items in ``memcell.items`` are silently
    skipped (agent → user-memory contract). An empty list is a successful result, not a failure.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        memcell: MemCell,
        *,
        prompt: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> list[Decision]:
        """Extract decisions from ``memcell``.

        Args:
            memcell: Source slice from boundary detection.
            prompt: Prompt override; ``None`` uses the bundled default.
            output_language: Language to write the decisions in, as an :class:`OutputLanguage` member
                or equivalent string in any casing. Naming one removes the decision from the model, which
                measured zero wrong languages; leaving it ``None`` asks the model to follow the
                participants, which costs roughly one extraction in thirteen and fails towards Chinese.
                See ``prompts/en/_language.py`` for the measurements.

        Raises:
            LLMError: From the LLM call.
            json.JSONDecodeError: On unparseable response.
            ValueError: If no JSON object is found, ``decisions`` is missing or not a list, or
                ``output_language`` names no supported language.
        """
        rendered = render_prompt(
            DECISION_GENERATION_PROMPT,
            prompt,
            CONVERSATION_TEXT=_render_conversation(memcell),
            language_rule=build_language_rule(output_language),
        )
        items = await _call_llm_for_decisions(self._llm, rendered, temperature=_DECISION_TEMPERATURE)
        decisions = _build_decisions_from_items(items, memcell=memcell)
        if len(decisions) > _DECISION_MAX_COUNT:
            decisions = decisions[:_DECISION_MAX_COUNT]
        return decisions

    extract = async_to_sync(aextract)


async def _call_llm_for_decisions(
    llm: LLMClient, rendered: str, *, temperature: float | None = None
) -> list[dict[str, Any]]:
    """Call LLM and return the validated ``decisions`` list.

    Uses brace-balanced extraction because ``decisions`` is nested (list of dicts).

    Raises:
        ValueError: If no JSON found or ``decisions`` key is missing/not a list.
    """
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)], temperature=temperature)
    text = response.content
    json_str = _extract_json_object(text)
    data: dict[str, Any] = json.loads(json_str)
    if "decisions" not in data:
        raise ValueError(f"decisions key missing from LLM response: {data!r}")
    items = data["decisions"]
    if not isinstance(items, list):
        raise ValueError(f"decisions must be a list, got {type(items).__name__}: {items!r}")  # noqa: TRY004
    return cast("list[dict[str, Any]]", items)


def _extract_json_object(text: str) -> str:
    """First balanced {{...}} block in text (brace-balanced parser for nested JSON)."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in decision LLM response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Unbalanced JSON in decision LLM response: {text[:200]!r}")


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


def _parse_tags(raw_tags: Any) -> list[str]:
    """Keep stripped string tags; anything other than a list becomes ``[]``."""
    if not isinstance(raw_tags, list):
        return []
    typed = cast("list[Any]", raw_tags)
    return [str(tag).strip() for tag in typed if str(tag).strip()]


def _build_decisions_from_items(items: list[dict[str, Any]], *, memcell: MemCell) -> list[Decision]:
    """Build Decision DTOs from the LLM item list.

    ``owner_id`` is always ``None`` (whole-memcell generic path). ``timestamp`` is the memcell's.
    Empty title / decision / reason drops the item rather than failing the call.
    """
    out: list[Decision] = []
    for item in items:
        if not isinstance(item, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            continue
        title = str(item.get("title") or "").strip()
        decision = str(item.get("decision") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if not title or not decision or not reason:
            continue
        raw_impact = item.get("impact")
        impact = str(raw_impact).strip() if raw_impact is not None else ""
        tags = _parse_tags(item.get("tags"))
        out.append(
            Decision(
                owner_id=None,
                title=title,
                decision=decision,
                reason=reason,
                impact=impact or None,
                tags=tags,
                timestamp=memcell.timestamp,
            )
        )
    return out
