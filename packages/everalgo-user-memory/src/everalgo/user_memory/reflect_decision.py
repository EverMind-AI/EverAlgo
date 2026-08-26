"""Merge N chronologically-ordered decisions into one current Decision."""

from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync
from pydantic import BaseModel, Field

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import Decision
from everalgo.user_memory._language import (
    EXISTING_DECISION_LANGUAGE_RULE,
    MERGED_DECISIONS_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
from everalgo.user_memory.prompts.en.reflect_decision import REFLECT_DECISION_PROMPT, REFLECT_DECISION_UPDATE_PROMPT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.llm.protocols import LLMClient


def _validate_inputs(decisions: list[Decision], *, min_count: int) -> None:
    """Fail fast: >= *min_count* decisions, sorted by timestamp ascending."""
    if len(decisions) < min_count:
        msg = f"areflect requires at least {min_count} decision(s), got {len(decisions)}"
        raise ValueError(msg)
    for i in range(1, len(decisions)):
        if decisions[i].timestamp < decisions[i - 1].timestamp:
            msg = (
                f"decisions must be sorted by timestamp ascending, "
                f"but decision[{i - 1}].timestamp={decisions[i - 1].timestamp} "
                f"> decision[{i}].timestamp={decisions[i].timestamp}"
            )
            raise ValueError(msg)


def _render_decision(dc: Decision) -> str:
    """Render one Decision's fields for the prompt; omit owner_id and timestamp."""
    impact = dc.impact if dc.impact else "(none)"
    tags = ", ".join(dc.tags) if dc.tags else "(none)"
    return f"Title: {dc.title}\nDecision: {dc.decision}\nReason: {dc.reason}\nImpact: {impact}\nTags: {tags}"


def _render_timeline(decisions: list[Decision]) -> str:
    """Render a numbered chronological list of decisions."""
    blocks: list[str] = []
    for i, dc in enumerate(decisions, 1):
        lines = _render_decision(dc).split("\n")
        numbered = f"{i}. {lines[0]}"
        rest = "\n".join(f"   {line}" for line in lines[1:])
        blocks.append(f"{numbered}\n{rest}")
    return "\n\n".join(blocks)


class _DecisionReflectOutput(BaseModel):
    """Structured Output schema for LLM response.

    ``title`` is declared after ``decision`` / ``reason`` deliberately: Structured Output generates
    fields in schema order, so a title declared first would be written before the trade-off it names.
    """

    decision: str = Field(description="The currently chosen option")
    reason: str = Field(description="Why that option is the current choice")
    title: str = Field(description="Short name for this trade-off")
    impact: str | None = Field(default=None, description="What this constrains later, or null")
    tags: list[str] = Field(default_factory=list, description="Short lowercase labels")


class DecisionReflector:
    """Merge N chronologically-ordered decisions into one current Decision.

    Two modes, mirroring EpisodeReflector:
      - INIT  (old_decision=None): full merge from all decisions.
      - UPDATE (old_decision given): update an existing Decision with new decisions.

    The result is still a Decision DTO, not a Principle.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def areflect(
        self,
        decisions: Sequence[Decision],
        *,
        old_decision: Decision | None = None,
        prompt: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Decision:
        """Merge decisions into one current Decision.

        Args:
            decisions: Source decisions sorted by timestamp ascending.
                INIT mode: must contain >= 2 items.
                UPDATE mode: must contain >= 1 item (new decisions only).
            old_decision: Existing merged Decision. None -> INIT mode. Decision -> UPDATE mode.
            prompt: Prompt override; None uses bundled default for the selected mode.
            output_language: Language to write the merged Decision in, as an :class:`OutputLanguage`
                member or equivalent string in any casing. ``None`` makes each mode inherit the language
                of its input — the decisions being merged, or the Decision being updated. Merging
                decisions written in different languages leaves the model to pick one, and an update
                inherits whatever the existing Decision already says. Name a language when the sources
                may disagree, or to correct a Decision already in the wrong one.

        Returns:
            Merged Decision. owner_id=None, timestamp=decisions[-1].timestamp.

        Raises:
            ValueError: Too few decisions, unsorted, empty required fields, unparseable LLM output, or
                ``output_language`` names no supported language.
            LLMError: Network or provider failure.
        """
        if old_decision is None:
            return await self._init_merge(decisions, prompt=prompt, output_language=output_language)
        return await self._update_merge(old_decision, decisions, prompt=prompt, output_language=output_language)

    reflect = async_to_sync(areflect)

    async def _init_merge(
        self,
        decisions: Sequence[Decision],
        *,
        prompt: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Decision:
        materialized = list(decisions)
        _validate_inputs(materialized, min_count=2)
        timeline = _render_timeline(materialized)
        rendered = render_prompt(
            REFLECT_DECISION_PROMPT,
            prompt,
            timeline=timeline,
            language_rule=build_language_rule(output_language, fallback=MERGED_DECISIONS_LANGUAGE_RULE),
        )
        return await self._call_llm(rendered, materialized)

    async def _update_merge(
        self,
        old_decision: Decision,
        decisions: Sequence[Decision],
        *,
        prompt: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Decision:
        materialized = list(decisions)
        _validate_inputs(materialized, min_count=1)
        timeline = _render_timeline(materialized)
        rendered = render_prompt(
            REFLECT_DECISION_UPDATE_PROMPT,
            prompt,
            old_decision=_render_decision(old_decision),
            new_decisions=timeline,
            language_rule=build_language_rule(output_language, fallback=EXISTING_DECISION_LANGUAGE_RULE),
        )
        return await self._call_llm(rendered, materialized)

    async def _call_llm(self, rendered: str, decisions: list[Decision]) -> Decision:
        response = await self._llm.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format=_DecisionReflectOutput,
        )
        if response.parsed is None:
            raise ValueError("LLM returned no parsed structured output")
        output: _DecisionReflectOutput = response.parsed  # type: ignore[assignment]
        title = output.title.strip()
        decision = output.decision.strip()
        reason = output.reason.strip()
        if not title or not decision or not reason:
            raise ValueError("Decision reflector LLM response has empty title, decision, or reason")
        impact = output.impact.strip() if output.impact else None
        return Decision(
            owner_id=None,
            title=title,
            decision=decision,
            reason=reason,
            impact=impact or None,
            tags=list(output.tags),
            timestamp=decisions[-1].timestamp,
        )
