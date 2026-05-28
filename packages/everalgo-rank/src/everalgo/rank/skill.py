"""Skill ranker facade — thin wrapper over ``rerank._basic_arank`` plus a skill-only relevance gate.

Skill relevance grading (the 0.0-1.0 quality bands once handled by a separate post-rerank verify
stage) is now folded into the single rerank prompt ``SKILL_RERANK_PROMPT_{EN,ZH}``. A skill-only
hard threshold (``min_rerank_score``, default 0.4) then drops candidates the LLM scored as
irrelevant — better to inject nothing than an off-target skill. The gate only fires when the rerank
stage ran, because raw fusion scores (e.g. RRF ~1/k) are not on a 0-1 relevance scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.rank.rerank import DEFAULT_RANK_CONFIG, RankConfig, _basic_arank
from everalgo.types import RankOutput

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import RankInput

__all__ = ["SkillRanker", "arank", "rank"]


def _apply_relevance_gate(result: RankOutput, min_rerank_score: float) -> RankOutput:
    """Skill-only post-rerank hard threshold — drop LLM-scored items below ``min_rerank_score``.

    Only meaningful after the LLM rerank stage, where ``item.score`` is a 0-1 relevance score.
    No-op when rerank did not run (fusion scores live on a different scale) or the gate is disabled.
    """
    if not result.metadata.get("reranked") or min_rerank_score <= 0.0:
        return result

    survivors = [it for it in result.items if it.score >= min_rerank_score]
    return RankOutput(
        items=survivors,
        metadata={
            **result.metadata,
            "rerank_min_score": min_rerank_score,
            "rerank_dropped": len(result.items) - len(survivors),
        },
    )


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
        min_rerank_score: float = 0.4,
    ) -> RankOutput:
        """Skill ranker — see ``rerank._basic_arank`` for the pipeline body.

        Args:
            rank_input: Query + candidate sets for skill memory ranking.
            config: Fusion mode and hyperparameters.
            prompt: Per-call rerank prompt override; ``None`` uses the built-in default.
            enable_rerank: When ``True``, run the LLM rerank stage after fusion. The rerank prompt
                both reorders and quality-grades candidates (see ``SKILL_RERANK_PROMPT_EN``).
            rerank_top_k: When set, Phase-5 LLM rerank truncates to this count instead of
                ``rank_input.top_k`` — lets fusion produce a wider candidate pool that the LLM
                then narrows.
            min_rerank_score: Skill-only relevance gate. After rerank, drop items whose LLM score is
                below this threshold (default 0.4). Set to ``0.0`` to disable. No effect unless
                ``enable_rerank`` ran, since fusion scores are not on a 0-1 scale.

        Returns:
            Ranked, optionally LLM-reranked, and (when reranked) relevance-gated skill items.
        """
        result = await _basic_arank(
            rank_input,
            config=config,
            llm=self._llm,
            prompt=prompt,
            enable_rerank=enable_rerank,
            rerank_top_k=rerank_top_k,
        )
        return _apply_relevance_gate(result, min_rerank_score)

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
    min_rerank_score: float = 0.4,
) -> RankOutput:
    """Skill module-level ranker — delegates to ``rerank._basic_arank`` plus a skill-only gate."""
    result = await _basic_arank(
        rank_input,
        config=config,
        llm=llm,
        prompt=prompt,
        enable_rerank=enable_rerank,
        rerank_top_k=rerank_top_k,
    )
    return _apply_relevance_gate(result, min_rerank_score)


rank = async_to_sync(arank)
