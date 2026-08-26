"""Synthesise engineering principles from a cluster of Decisions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Decision, Principle
from everalgo.user_memory._language import (
    PRINCIPLES_FROM_DECISIONS_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
from everalgo.user_memory.prompts.en.principle import PRINCIPLE_GENERATION_PROMPT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.llm.protocols import LLMClient


_PRINCIPLE_TEMPERATURE = 0.3
_PRINCIPLE_MAX_COUNT = 10


class PrincipleExtractor:
    """Synthesise zero or more Principles from one Decision cluster.

    Each input item is ``(entry_id, Decision)``. ``entry_id`` is the caller's storage id (EverOS
    markdown entry id later); this operator never invents one. An empty cluster returns ``[]``
    without calling the LLM.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        decisions: Sequence[tuple[str, Decision]],
        *,
        owner_id: str,
        prompt: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> list[Principle]:
        """Synthesise principles from ``decisions``.

        Args:
            decisions: Cluster members as ``(entry_id, Decision)`` pairs. Empty is a successful
                ``[]``. ``entry_id`` values must be non-empty and unique.
            owner_id: Bound onto every returned Principle. Must be non-empty.
            prompt: Prompt override; ``None`` uses the bundled default.
            output_language: Language to write the principles in. ``None`` inherits from the
                decisions rather than judging a conversation.

        Raises:
            LLMError: From the LLM call.
            json.JSONDecodeError: On unparseable response.
            ValueError: Empty ``owner_id`` or ``entry_id``, duplicate ``entry_id``, missing JSON /
                ``principles`` key, or unsupported ``output_language``.
        """
        if not owner_id.strip():
            raise ValueError("owner_id must be a non-empty string")
        pairs = _materialize_pairs(decisions)
        if not pairs:
            return []
        rendered = render_prompt(
            PRINCIPLE_GENERATION_PROMPT,
            prompt,
            DECISION_CLUSTER=_render_cluster(pairs),
            language_rule=build_language_rule(output_language, fallback=PRINCIPLES_FROM_DECISIONS_LANGUAGE_RULE),
        )
        items = await _call_llm_for_principles(self._llm, rendered, temperature=_PRINCIPLE_TEMPERATURE)
        principles = _build_principles_from_items(items, pairs=pairs, owner_id=owner_id)
        if len(principles) > _PRINCIPLE_MAX_COUNT:
            principles = principles[:_PRINCIPLE_MAX_COUNT]
        return principles

    extract = async_to_sync(aextract)


def _materialize_pairs(decisions: Sequence[tuple[str, Decision]]) -> list[tuple[str, Decision]]:
    """Strip entry ids; reject empties and duplicates."""
    pairs: list[tuple[str, Decision]] = []
    seen: set[str] = set()
    for entry_id, decision in decisions:
        eid = entry_id.strip()
        if not eid:
            raise ValueError("entry_id must be a non-empty string")
        if eid in seen:
            raise ValueError(f"duplicate entry_id {eid!r}")
        seen.add(eid)
        pairs.append((eid, decision))
    return pairs


def _render_decision(dc: Decision) -> str:
    """Render one Decision's fields for the prompt; omit owner_id and timestamp."""
    impact = dc.impact if dc.impact else "(none)"
    tags = ", ".join(dc.tags) if dc.tags else "(none)"
    return f"Title: {dc.title}\nDecision: {dc.decision}\nReason: {dc.reason}\nImpact: {impact}\nTags: {tags}"


def _render_cluster(pairs: list[tuple[str, Decision]]) -> str:
    """Numbered cluster with opaque ``id=`` values the model must cite."""
    blocks: list[str] = []
    for i, (entry_id, dc) in enumerate(pairs, 1):
        lines = _render_decision(dc).split("\n")
        header = f"{i}. id={entry_id}"
        rest = "\n".join(f"   {line}" for line in lines)
        blocks.append(f"{header}\n{rest}")
    return "\n\n".join(blocks)


async def _call_llm_for_principles(
    llm: LLMClient, rendered: str, *, temperature: float | None = None
) -> list[dict[str, Any]]:
    """Call LLM and return the validated ``principles`` list."""
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)], temperature=temperature)
    text = response.content
    json_str = _extract_json_object(text)
    data: dict[str, Any] = json.loads(json_str)
    if "principles" not in data:
        raise ValueError(f"principles key missing from LLM response: {data!r}")
    items = data["principles"]
    if not isinstance(items, list):
        raise ValueError(f"principles must be a list, got {type(items).__name__}: {items!r}")  # noqa: TRY004
    return cast("list[dict[str, Any]]", items)


def _extract_json_object(text: str) -> str:
    """First balanced {{...}} block in text (brace-balanced parser for nested JSON)."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in principle LLM response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Unbalanced JSON in principle LLM response: {text[:200]!r}")


def _filter_source_ids(raw: Any, *, allowed: set[str]) -> list[str]:
    """Keep first-seen ids that appear in ``allowed``; drop the rest."""
    if not isinstance(raw, list):
        return []
    typed = cast("list[Any]", raw)
    seen: set[str] = set()
    out: list[str] = []
    for item in typed:
        eid = str(item).strip()
        if eid in allowed and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def _build_principles_from_items(
    items: list[dict[str, Any]],
    *,
    pairs: list[tuple[str, Decision]],
    owner_id: str,
) -> list[Principle]:
    """Build Principle DTOs; drop items with empty text or no remaining source ids."""
    by_id = dict(pairs)
    allowed = set(by_id)
    out: list[Principle] = []
    for item in items:
        if not isinstance(item, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            continue
        title = str(item.get("title") or "").strip()
        statement = str(item.get("statement") or "").strip()
        if not title or not statement:
            continue
        source_ids = _filter_source_ids(item.get("source_entry_ids"), allowed=allowed)
        if not source_ids:
            continue
        timestamp = max(by_id[eid].timestamp for eid in source_ids)
        out.append(
            Principle(
                owner_id=owner_id,
                title=title,
                statement=statement,
                source_entry_ids=source_ids,
                timestamp=timestamp,
            )
        )
    return out
