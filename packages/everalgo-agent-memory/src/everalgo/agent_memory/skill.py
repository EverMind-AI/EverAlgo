"""AgentSkillExtractor — incremental skill maintenance for one cluster.

Operation atoms (``_apply_add`` / ``_apply_update`` / ``_evaluate_maturity`` / formatters) live in
:mod:`everalgo.agent_memory.skill_ops`; this module holds the public :class:`AgentSkillExtractor`.

Return contract: ``list[AgentSkill]`` — caller decodes add / update / retire via
``id ∈ existing_relevant_skills`` + ``confidence < retire_confidence``.
The caller pre-filters ``existing_relevant_skills`` to the relevant subset before passing in.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

from everalgo.agent_memory.prompts.skill_failure import AGENT_SKILL_FAILURE_EXTRACT_PROMPT
from everalgo.agent_memory.prompts.skill_success import AGENT_SKILL_SUCCESS_EXTRACT_PROMPT
from everalgo.agent_memory.reasons import OpOutcome, SkillExtractionResult, SkillSkipReason
from everalgo.agent_memory.skill_ops import (
    _apply_add,
    _apply_update,
    _format_cases,
    _format_existing_skills,
)
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.llm.protocols import LLMClient
    from everalgo.types import AgentCase, AgentSkill

logger = logging.getLogger(__name__)


__all__ = [
    # Re-exported prompt constants — monkey-patch at startup to override the LLM prompts
    "AGENT_SKILL_FAILURE_EXTRACT_PROMPT",
    "AGENT_SKILL_SUCCESS_EXTRACT_PROMPT",
    "AgentSkillExtractor",
]


@dataclass(frozen=True)
class _SkillCfg:
    """Internal threshold bag — packed from aextract kwargs; not part of the public API."""

    # Cases below this quality score short-circuit to [] without calling the LLM.
    skip_quality_threshold: float = 0.2
    # Maturity scoring is implemented but default-skipped: current retrieval doesn't consult
    # maturity_score. Set False to enable; useful for future retrieval paths that filter on it.
    skip_maturity_scoring: bool = True
    maturity_threshold: float = 0.6  # only consulted when skip_maturity_scoring=False
    retire_confidence: float = 0.1
    failure_quality_threshold: float = 0.5
    max_description_tokens: int = 400
    max_content_tokens: int = 5000
    # Below: change-ratio bands for maturity re-eval; inert when skip_maturity_scoring=True.
    maturity_trivial_change_ratio: float = 0.2
    maturity_reeval_change_ratio: float = 0.4


class AgentSkillExtractor:
    """Aggregate one new AgentCase into incremental skill operations for a cluster.

    Returns ``list[AgentSkill]`` with add / update / retire signals (§5.2 encoding); caller persists.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        case: AgentCase,
        *,
        existing_relevant_skills: Sequence[AgentSkill],
        supporting_cases: Sequence[AgentCase],
        prompt_success: str | None = None,
        prompt_failure: str | None = None,
        prompt_maturity: str | None = None,
        skip_quality_threshold: float = 0.2,
        skip_maturity_scoring: bool = True,
        maturity_threshold: float = 0.6,
        retire_confidence: float = 0.1,
        failure_quality_threshold: float = 0.5,
        max_description_tokens: int = 400,
        max_content_tokens: int = 5000,
        maturity_trivial_change_ratio: float = 0.2,
        maturity_reeval_change_ratio: float = 0.4,
    ) -> list[AgentSkill]:
        """Integrate ``case`` into the skill set; return add / update / retire entries.

        Thin wrapper over :meth:`aextract_with_reason` — use that one when an empty (or
        smaller-than-expected) result needs to be explained rather than merely observed.

        See :meth:`aextract_with_reason` for the full argument reference.

        Raises:
            LLMError: Propagated from the underlying LLM client call.
            json.JSONDecodeError: If the LLM response is not valid JSON.
            TypeError: If the LLM response is not a dict, or ``operations`` is not a list.
        """
        result = await self.aextract_with_reason(
            case,
            existing_relevant_skills=existing_relevant_skills,
            supporting_cases=supporting_cases,
            prompt_success=prompt_success,
            prompt_failure=prompt_failure,
            prompt_maturity=prompt_maturity,
            skip_quality_threshold=skip_quality_threshold,
            skip_maturity_scoring=skip_maturity_scoring,
            maturity_threshold=maturity_threshold,
            retire_confidence=retire_confidence,
            failure_quality_threshold=failure_quality_threshold,
            max_description_tokens=max_description_tokens,
            max_content_tokens=max_content_tokens,
            maturity_trivial_change_ratio=maturity_trivial_change_ratio,
            maturity_reeval_change_ratio=maturity_reeval_change_ratio,
        )
        return result.skills

    extract = async_to_sync(aextract)

    async def aextract_with_reason(
        self,
        case: AgentCase,
        *,
        existing_relevant_skills: Sequence[AgentSkill],
        supporting_cases: Sequence[AgentCase],
        prompt_success: str | None = None,
        prompt_failure: str | None = None,
        prompt_maturity: str | None = None,
        skip_quality_threshold: float = 0.2,
        skip_maturity_scoring: bool = True,
        maturity_threshold: float = 0.6,
        retire_confidence: float = 0.1,
        failure_quality_threshold: float = 0.5,
        max_description_tokens: int = 400,
        max_content_tokens: int = 5000,
        maturity_trivial_change_ratio: float = 0.2,
        maturity_reeval_change_ratio: float = 0.4,
    ) -> SkillExtractionResult:
        """Integrate ``case`` into the skill set, reporting what happened to every proposed operation.

        Same algorithm as :meth:`aextract`. The difference is the return shape: one
        :class:`OpOutcome` per LLM-proposed operation (so ``len(outcomes) == len(operations)``),
        each holding either the resulting skill or the reason it was dropped, plus a separate
        ``pre_reason`` / ``pre_detail`` pair for the quality short-circuit that fires before any
        operation exists.

        Args:
            case: The newly extracted ``AgentCase`` to integrate.
            existing_relevant_skills: Pre-filtered by the caller (typically top-K cosine). Pass ``[]`` when none.
            supporting_cases: AgentCase records referenced by ``existing_relevant_skills[*].source_case_ids``;
                the algorithm builds an ``id → AgentCase`` dict and renders matched cases under each skill in
                the prompt. Pass ``[]`` when no supporting cases are available.
            prompt_success: Prompt override for successful-case path.
            prompt_failure: Prompt override for failed-case path.
            prompt_maturity: Prompt override for maturity-evaluation step.
            skip_quality_threshold: Cases with ``quality_score < this`` short-circuit to ``[]`` without LLM call. Default 0.2.
            skip_maturity_scoring: Skip LLM maturity scoring (default True); set False to enable.
            maturity_threshold: Minimum score to mark a skill as mature (consulted when not skipping).
            retire_confidence: Confidence below which a skill is marked for retirement.
            failure_quality_threshold: Quality score below which the failure prompt branch is used.
            max_description_tokens: Token budget for skill description in the prompt.
            max_content_tokens: Token budget for skill content in the prompt.
            maturity_trivial_change_ratio: Change ratio below which maturity re-eval is skipped.
            maturity_reeval_change_ratio: Change ratio above which maturity is reset and re-evaluated.

        Raises:
            LLMError: Propagated from the underlying LLM client call.
            json.JSONDecodeError: If the LLM response is not valid JSON.
            TypeError: If the LLM response is not a dict, or ``operations`` is not a list.
        """
        cfg = _SkillCfg(
            skip_quality_threshold=skip_quality_threshold,
            skip_maturity_scoring=skip_maturity_scoring,
            maturity_threshold=maturity_threshold,
            retire_confidence=retire_confidence,
            failure_quality_threshold=failure_quality_threshold,
            max_description_tokens=max_description_tokens,
            max_content_tokens=max_content_tokens,
            maturity_trivial_change_ratio=maturity_trivial_change_ratio,
            maturity_reeval_change_ratio=maturity_reeval_change_ratio,
        )
        if case.quality_score < cfg.skip_quality_threshold:
            logger.debug(
                "skipping skill extraction: case.quality_score=%.2f < skip_quality_threshold=%.2f",
                case.quality_score,
                cfg.skip_quality_threshold,
            )
            return SkillExtractionResult(
                SkillSkipReason.CASE_QUALITY_BELOW_THRESHOLD,
                {"quality": case.quality_score, "threshold": cfg.skip_quality_threshold},
                [],
            )

        client = self._llm

        existing_list = list(existing_relevant_skills)

        # Format inputs for the LLM prompt
        new_case_json = _format_cases([case])
        existing_skills_json = _format_existing_skills(
            existing_list,
            supporting_cases=supporting_cases,
            max_description_tokens=cfg.max_description_tokens,
            max_content_tokens=cfg.max_content_tokens,
        )

        # Success-vs-failure prompt selection based on case.quality_score
        max_quality = case.quality_score
        if max_quality < cfg.failure_quality_threshold:
            template = AGENT_SKILL_FAILURE_EXTRACT_PROMPT
            override = prompt_failure
            logger.debug("using failure prompt (max_quality=%.2f < %.2f)", max_quality, cfg.failure_quality_threshold)
        else:
            template = AGENT_SKILL_SUCCESS_EXTRACT_PROMPT
            override = prompt_success
            logger.debug("using success prompt (max_quality=%.2f >= %.2f)", max_quality, cfg.failure_quality_threshold)

        # Single LLM call (no retry per ADR 012)
        rendered = render_prompt(
            template,
            override,
            new_case_json=new_case_json,
            existing_skills_json=existing_skills_json,
        )
        logger.debug(
            "incremental extraction: case=%s, new_cases=1, existing_relevant_skills=%d",
            case.id,
            len(existing_list),
        )
        llm_data = await _call_llm_for_skill_extract(client, rendered)
        operations = llm_data["operations"]
        note = llm_data.get("update_note", "")
        if note:
            logger.debug("LLM update_note: %s", note)

        # Apply ops in-memory — one OpOutcome per proposed op, so len(outcomes) == len(operations)
        outcomes: list[OpOutcome] = []
        processed_indices: set[int] = set()
        source_case_ids = [case.id] if case.id else []

        for op_index, op in enumerate(cast("list[dict[str, Any]]", operations)):
            if not isinstance(op, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
                logger.warning("skill op is not a dict, skipping: %r", op)
                outcomes.append(OpOutcome(op_index, None, SkillSkipReason.OP_NOT_DICT, {}))
                continue
            action: str | None = op.get("action")

            if action == "add":
                outcomes.append(
                    await _apply_add(
                        op,
                        source_case_ids,
                        op_index=op_index,
                        client=client,
                        cfg=cfg,
                        prompt_maturity=prompt_maturity,
                    )
                )
            elif action == "update":
                outcomes.append(
                    await _apply_update(
                        op,
                        existing_list,
                        source_case_ids,
                        source_quality=max_quality,
                        op_index=op_index,
                        client=client,
                        prompt_maturity=prompt_maturity,
                        cfg=cfg,
                        processed_indices=processed_indices,
                    )
                )
            elif action == "none":
                outcomes.append(OpOutcome(op_index, None, SkillSkipReason.OP_ACTION_NONE, {}))
            else:
                logger.warning("unknown action %r, skipping", action)
                outcomes.append(OpOutcome(op_index, None, SkillSkipReason.OP_UNKNOWN_ACTION, {"action": action}))

        applied = sum(1 for o in outcomes if o.skill is not None)
        logger.info(
            "case=%s ops applied: total=%d of %d proposed (adds + updates + retires per §5.2 encoding)",
            case.id,
            applied,
            len(outcomes),
        )
        return SkillExtractionResult(None, {}, outcomes)

    extract_with_reason = async_to_sync(aextract_with_reason)


# ---------------------------------------------------------------------------
# LLM callsite — brace-balanced JSON extraction + 5-retry.
# ---------------------------------------------------------------------------


async def _call_llm_for_skill_extract(llm: LLMClient, rendered: str) -> dict[str, Any]:
    """Call LLM for skill extraction and return validated dict with operations list.

    Uses brace-balanced extraction because the response is nested:
    ``{"operations": [{"action": "...", "data": {...}}, ...], "update_note": "..."}``.

    Raises:
        ValueError: If no JSON found or ``operations`` key missing / not a list.
    """
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    json_str = _extract_json_object(text)
    data: dict[str, Any] = json.loads(json_str)
    if "operations" not in data:
        raise ValueError(f"Skill extract response missing 'operations' key: {list(data.keys())!r}")
    if not isinstance(data["operations"], list):
        raise ValueError(f"operations must be a list, got {type(data['operations']).__name__}: {data!r}")  # noqa: TRY004
    return data


def _extract_json_object(text: str) -> str:
    """First balanced {{...}} block in text (brace-balanced parser for nested JSON)."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in skill LLM response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Unbalanced JSON in skill LLM response: {text[:200]!r}")
