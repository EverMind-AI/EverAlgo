"""Skill ranker facade — thin wrapper over ``rerank._basic_arank`` plus a skill-only post-rerank LLM verify stage."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

from asgiref.sync import async_to_sync
from pydantic import BaseModel, Field

from everalgo.llm.types import ChatMessage
from everalgo.rank.prompts.en.skill_verify import SKILL_VERIFY_PROMPT_EN
from everalgo.rank.rerank import DEFAULT_RANK_CONFIG, RankConfig, _basic_arank
from everalgo.types import RankOutput, ScoredItem

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.llm.protocols import LLMClient
    from everalgo.types import RankInput

__all__ = ["SkillRanker", "arank", "averify", "rank", "verify"]

logger = logging.getLogger(__name__)


class VerifiedItem(BaseModel):
    """One skill's LLM-assigned relevance verdict — mirrors the enterprise JSON schema."""

    index: int
    score: float
    reason: str = ""


class _VerifyResponse(BaseModel):
    """LLM verify output schema (``{"results": [...]}``)."""

    results: list[VerifiedItem] = Field(default_factory=list[VerifiedItem])


async def _verify_relevance(
    items: Sequence[ScoredItem],
    *,
    query: str,
    llm: LLMClient,
    threshold: float,
    prompt: str,
) -> list[ScoredItem]:
    """Inner verify implementation: LLM-score → hard-filter → sort.

    Pulls ``name`` / ``description`` / ``content`` out of each item's metadata to build
    the prompt payload (the same three fields the enterprise pipeline serialises).
    On any LLM exception the input list is returned unchanged.
    """
    if not items:
        return list(items)

    skills_for_prompt = [
        {
            "index": i,
            "name": item.metadata.get("name", ""),
            "description": item.metadata.get("description", ""),
            "content": item.metadata.get("content", ""),
        }
        for i, item in enumerate(items)
    ]
    rendered = prompt.format(
        query=query,
        skills_json=json.dumps(skills_for_prompt, ensure_ascii=False, default=str),
    )

    try:
        response = await llm.chat(
            messages=[ChatMessage(role="user", content=rendered)],
            temperature=0.0,
            response_format=_VerifyResponse,
        )
        parsed = cast("_VerifyResponse | None", response.parsed)
    except Exception:
        logger.warning("Skill verify failed, returning all results", exc_info=True)
        return list(items)
    if parsed is None:
        logger.warning("Skill verify returned no parsed structured output, returning all results")
        return list(items)

    score_map = {r.index: r.score for r in parsed.results}
    survivors: list[ScoredItem] = []
    for i, item in enumerate(items):
        relevance = score_map.get(i, 0.0)
        if relevance < threshold:
            continue
        survivors.append(
            item.model_copy(
                update={
                    "score": relevance,
                    "metadata": {
                        **item.metadata,
                        "pre_verify_score": item.score,
                    },
                }
            )
        )

    survivors.sort(key=lambda s: s.score, reverse=True)
    logger.info("skill verify: %d/%d passed (threshold=%.2f)", len(survivors), len(items), threshold)
    return survivors


async def averify(
    rank_output: RankOutput,
    *,
    query: str,
    llm: LLMClient,
    threshold: float = 0.4,
    prompt: str = SKILL_VERIFY_PROMPT_EN,
) -> RankOutput:
    """Skill-only post-rerank LLM relevance verification.

    Args:
        rank_output: Output of an upstream ``skill.arank`` call (or any ``RankOutput`` whose
            items carry ``name``/``description``/``content`` metadata).
        query: The original user query (passed through verbatim into the prompt).
        llm: LLM client used for the verify call.
        threshold: Minimum LLM relevance score for an item to survive. Default ``0.4``
            matches the enterprise ``_verify_skill_relevance`` cut-off.
        prompt: Verify prompt template; must contain ``{query}`` and ``{skills_json}``
            placeholders. Defaults to ``SKILL_VERIFY_PROMPT_EN``.

    Returns:
        A new ``RankOutput`` containing only items whose LLM-assigned relevance ≥ ``threshold``,
        with each surviving item's score replaced by the LLM score and the original fusion /
        rerank score preserved in ``metadata["pre_verify_score"]``. On LLM failure the input
        is returned unchanged (graceful degradation, matches enterprise behaviour).
    """
    verified = await _verify_relevance(rank_output.items, query=query, llm=llm, threshold=threshold, prompt=prompt)
    return RankOutput(
        items=verified,
        metadata={
            **rank_output.metadata,
            "verified": True,
            "verify_threshold": threshold,
            "verify_dropped": len(rank_output.items) - len(verified),
        },
    )


verify = async_to_sync(averify)
"""Sync bridge for non-event-loop contexts. Per ADR 010."""


class SkillRanker:
    """Class-style facade for the skill ranker.

    The LLM client is bound to the instance at construction time.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def arank(
        self,
        rank_input: RankInput,
        *,
        config: RankConfig = DEFAULT_RANK_CONFIG,
        prompt: str | None = None,
        enable_rerank: bool = False,
        rerank_top_k: int | None = None,
        enable_verify: bool = False,
        verify_threshold: float = 0.4,
        verify_prompt: str = SKILL_VERIFY_PROMPT_EN,
    ) -> RankOutput:
        """Skill ranker — see ``rerank._basic_arank`` for the pipeline body.

        Args:
            rank_input: Query + candidate sets for skill memory ranking.
            config: Fusion mode and hyperparameters.
            prompt: Per-call rerank prompt override; ``None`` uses the built-in default.
            enable_rerank: When ``True``, run the LLM rerank stage after fusion.
            rerank_top_k: When set, Phase-5 LLM rerank truncates to this count instead of
                ``rank_input.top_k`` — lets fusion produce a wider candidate pool that the LLM
                then narrows.
            enable_verify: When ``True``, run the skill-only post-rerank LLM verify stage
                (``averify``) after fusion/rerank. Skill is the only facade exposing this flag.
            verify_threshold: Minimum LLM relevance score for the verify stage (default 0.4).
            verify_prompt: Verify-stage prompt override; defaults to ``SKILL_VERIFY_PROMPT_EN``.

        Returns:
            Ranked, optionally LLM-reranked, and optionally LLM-verified skill items.
        """
        result = await _basic_arank(
            rank_input,
            config=config,
            llm=self._llm,
            prompt=prompt,
            enable_rerank=enable_rerank,
            rerank_top_k=rerank_top_k,
        )
        if enable_verify and result.items:
            result = await averify(
                result,
                query=rank_input.query,
                llm=self._llm,
                threshold=verify_threshold,
                prompt=verify_prompt,
            )
        return result

    rank = async_to_sync(arank)
    """Sync bridge — only callable from non-event-loop contexts."""


async def arank(
    rank_input: RankInput,
    *,
    config: RankConfig = DEFAULT_RANK_CONFIG,
    llm: LLMClient | None = None,
    prompt: str | None = None,
    enable_rerank: bool = False,
    rerank_top_k: int | None = None,
    enable_verify: bool = False,
    verify_threshold: float = 0.4,
    verify_prompt: str = SKILL_VERIFY_PROMPT_EN,
) -> RankOutput:
    """Skill module-level ranker — delegates to ``rerank._basic_arank`` plus optional verify."""
    result = await _basic_arank(
        rank_input,
        config=config,
        llm=llm,
        prompt=prompt,
        enable_rerank=enable_rerank,
        rerank_top_k=rerank_top_k,
    )
    if enable_verify and result.items:
        if llm is None:
            raise ValueError("enable_verify=True requires llm to be provided")
        result = await averify(
            result,
            query=rank_input.query,
            llm=llm,
            threshold=verify_threshold,
            prompt=verify_prompt,
        )
    return result


rank = async_to_sync(arank)
