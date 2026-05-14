"""AgentSkillExtractor — incremental skill maintenance for one cluster.

Port of opensource ``memory_layer/memory_extractor/agent_skill_extractor.py`` 10-step pipeline (see DESIGN.md
§1.2 / §5). All operation atoms (``_apply_add`` / ``_apply_update`` / ``_evaluate_maturity`` / prompt
formatters) live in :mod:`everalgo.agent_memory.skill_ops` per the O-7 split decision; this module holds
the public :class:`AgentSkillExtractor` and the :class:`SkillConfig` dataclass.

Return contract: ``list[AgentSkill]`` (O-2 decision). Caller decodes add / update / retire via two object
fields — see DESIGN.md §5.2 (``id ∈ existing_relevant_skills`` + ``confidence < retire_confidence``).

Caller-side filtering: external code pre-filters the cluster's skill set to the relevant subset (typically
via vector cosine top-K against the new case's embedding) before passing in — the algorithm no longer takes
``query_vector`` or runs internal top-K. Caller also owns embedding lifecycle (vector / vector_model are
not on the EverAlgo schema).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.agent_memory.prompts.skill_failure import AGENT_SKILL_FAILURE_EXTRACT_PROMPT
from everalgo.agent_memory.prompts.skill_success import AGENT_SKILL_SUCCESS_EXTRACT_PROMPT
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
    "SkillConfig",
]


@dataclass(frozen=True)
class SkillConfig:
    """Policy thresholds for :class:`AgentSkillExtractor`.

    Caller MUST hold an instance (or use the defaults) and reads ``retire_confidence`` to identify retired
    skills in the returned list — see DESIGN.md §5.2 list encoding contract.
    """

    maturity_threshold: float = 0.6
    """Maturity score at/above which a skill is considered "mature" — gates re-eval decisions."""

    retire_confidence: float = 0.1
    """Confidence below which the caller MUST soft-retire (remove from search engines, keep in DB)."""

    failure_quality_threshold: float = 0.5
    """Below this threshold the failure-extract prompt is used; at/above, the success-extract prompt."""

    skip_maturity_scoring: bool = True
    """Skip the per-op maturity LLM call entirely; when ``True`` (default) ``_evaluate_maturity`` returns
    ``1.0`` without making any LLM call. Set ``False`` to opt into 4-dimension LLM scoring."""

    max_case_history: int = 9
    max_description_tokens: int = 400
    max_content_tokens: int = 5000

    maturity_trivial_change_ratio: float = 0.2
    """Content change ratio below which maturity re-eval is always skipped."""

    maturity_reeval_change_ratio: float = 0.4
    """Content change ratio at/above which maturity is always re-evaluated."""


class AgentSkillExtractor:
    """Aggregate one new :class:`AgentCase` into incremental skill operations for its cluster.

    Stateless callable class — no ``__init__``, no instance state. **NO DB writes**: returns
    ``list[AgentSkill]`` carrying add / update / retire signals via the §5.2 encoding, caller persists.
    """

    async def aextract(  # noqa: C901  — branches mirror opensource extract_and_save :790-941
        self,
        case: AgentCase,
        *,
        cluster_id: str,
        existing_relevant_skills: Sequence[AgentSkill],
        case_history: Sequence[AgentCase],
        llm: LLMClient | None = None,
        prompt_success: str | None = None,
        prompt_failure: str | None = None,
        prompt_maturity: str | None = None,
        config: SkillConfig | None = None,
    ) -> list[AgentSkill]:
        """Async main implementation — runs the simplified 5-step pipeline (DESIGN.md §1.2).

        Parameters
        ----------
        case : AgentCase
            The freshly extracted case to integrate.
        cluster_id : str
            Cluster these skills belong to (caller-managed; from :func:`everalgo.clustering.cluster_by_llm`).
        existing_relevant_skills : Sequence[AgentSkill]
            Skills already stored under ``cluster_id``, **pre-filtered by the caller** to the relevant
            subset (typically via vector cosine top-K against ``case.task_intent``). Pass ``[]`` when none.
        case_history : Sequence[AgentCase]
            Historical cases referenced by ``existing_relevant_skills[i].source_case_ids`` — used to attach
            ``supporting_cases`` summaries inside the prompt. Pass ``[]`` when none.
        llm : LLMClient or None, optional
            Per-call LLM override; falls back through the 3-layer chain.
        prompt_success, prompt_failure, prompt_maturity : str or None, optional
            Per-call prompt template overrides. ``None`` falls back to the built-in
            ``AGENT_SKILL_SUCCESS_EXTRACT_PROMPT`` / ``AGENT_SKILL_FAILURE_EXTRACT_PROMPT`` /
            ``AGENT_SKILL_MATURITY_SCORE_PROMPT`` constants.
        config : SkillConfig or None, optional
            Policy thresholds; defaults to ``SkillConfig()`` when ``None``.

        Returns
        -------
        list[AgentSkill]
            Add / update / retire entries under the §5.2 encoding. Caller distinguishes via
            ``id ∈ existing_relevant_skills`` + ``confidence < cfg.retire_confidence``.

        Raises
        ------
        LLMNotConfiguredError
            No LLM resolvable through the 3-layer chain.
        LLMError
            Any provider-side failure on the main extract call (no internal retry, ADR 012).
        """
        cfg = config or SkillConfig()
        client = everalgo.llm.resolve(llm)

        existing_list = list(existing_relevant_skills)

        # Format inputs for the LLM prompt
        new_case_json = _format_cases([case])
        existing_skills_json = _format_existing_skills(
            existing_list,
            case_history=case_history,
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
            "incremental extraction: cluster=%s, new_cases=1, existing_relevant_skills=%d",
            cluster_id,
            len(existing_list),
        )
        response = await client.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
        )
        try:
            llm_result: Any = json.loads(response.content)
        except json.JSONDecodeError as exc:
            logger.warning("skill LLM JSON parse failed for cluster=%s: %s", cluster_id, exc)
            return []

        if not isinstance(llm_result, dict):
            logger.warning("skill LLM returned non-dict for cluster=%s", cluster_id)
            return []
        llm_dict = cast("dict[str, Any]", llm_result)

        operations_raw = llm_dict.get("operations", [])
        if not isinstance(operations_raw, list):
            logger.warning("skill LLM returned non-list operations for cluster=%s", cluster_id)
            return []
        # cast is for pyright strict mode (mypy sees it as redundant — silenced below)
        operations = cast("list[Any]", operations_raw)  # type: ignore[redundant-cast]

        note = str(llm_dict.get("update_note") or "")
        if note:
            logger.debug("LLM update_note: %s", note)

        # Apply ops in-memory — emit list[AgentSkill] per §5.2 encoding
        result: list[AgentSkill] = []
        processed_indices: set[int] = set()
        source_case_ids = [case.id] if case.id else []

        for raw_op in operations:
            if not isinstance(raw_op, dict):
                logger.warning("skipping non-dict op: %r", raw_op)
                continue
            op = cast("dict[str, Any]", raw_op)
            action = str(op.get("action") or "none")

            if action == "add":
                added = await _apply_add(
                    op,
                    cluster_id,
                    source_case_ids,
                    client=client,
                    cfg=cfg,
                    prompt_maturity=prompt_maturity,
                )
                if added is not None:
                    result.append(added)
            elif action == "update":
                updated = await _apply_update(
                    op,
                    existing_list,
                    source_case_ids,
                    source_quality=max_quality,
                    client=client,
                    prompt_maturity=prompt_maturity,
                    cfg=cfg,
                    processed_indices=processed_indices,
                )
                if updated is not None:
                    result.append(updated)
            elif action == "none":
                continue
            else:
                logger.warning("unknown action %r, skipping", action)

        logger.info(
            "cluster=%s ops applied: total=%d (adds + updates + retires per §5.2 encoding)",
            cluster_id,
            len(result),
        )
        return result

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""
