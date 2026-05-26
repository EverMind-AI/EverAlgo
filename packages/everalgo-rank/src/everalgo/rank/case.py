"""Case ranker facade — thin wrapper over ``rerank._basic_arank``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.rank.rerank import DEFAULT_RANK_CONFIG, RankConfig, _basic_arank

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import RankInput, RankOutput

__all__ = ["CaseRanker"]


class CaseRanker:
    """Class-style facade for the case ranker.

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
    ) -> RankOutput:
        """Case ranker — see ``rerank._basic_arank`` for the pipeline body.

        Args:
            rank_input: Query + candidate sets for agent-case memory ranking.
            config: Fusion mode and hyperparameters.
            prompt: Per-call rerank prompt override; ``None`` uses the built-in default.
            enable_rerank: When ``True``, run the LLM rerank stage after fusion.
            rerank_top_k: When set, Phase-5 LLM rerank truncates to this count instead of
                ``rank_input.top_k`` — lets fusion produce a wider candidate pool that the LLM
                then narrows.

        Returns:
            Ranked and optionally LLM-reranked case items.
        """
        return await _basic_arank(
            rank_input,
            config=config,
            llm=self._llm,
            prompt=prompt,
            enable_rerank=enable_rerank,
            rerank_top_k=rerank_top_k,
        )

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
) -> RankOutput:
    """Case module-level ranker — delegates to ``rerank._basic_arank``."""
    return await _basic_arank(
        rank_input,
        config=config,
        llm=llm,
        prompt=prompt,
        enable_rerank=enable_rerank,
        rerank_top_k=rerank_top_k,
    )


rank = async_to_sync(arank)
