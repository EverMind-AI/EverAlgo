"""Case ranker facade — thin wrapper over ``rerank._basic_arank``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.rank.rerank import DEFAULT_RANK_CONFIG, RankConfig, _basic_arank

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.rank.fusion import RerankFn, RetrieveFn
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
        retrieve_fn: RetrieveFn | None = None,
        rerank_fn: RerankFn | None = None,
    ) -> RankOutput:
        """Case ranker — see ``rerank._basic_arank`` for the pipeline body.

        Args:
            rank_input: Query + candidate sets for agent-case memory ranking.
            config: Fusion mode and hyperparameters.
            prompt: Per-call rerank prompt override; ``None`` uses the built-in default.
            enable_rerank: When ``True``, run the LLM rerank stage after fusion.
            retrieve_fn: Optional retrieval callback for the ``'agentic'`` fusion mode Round 2.
            rerank_fn: Required for ``fusion_mode='agentic'``; cross-encoder callback for Round 1 rerank.

        Returns:
            Ranked and optionally LLM-reranked case items.
        """
        return await _basic_arank(
            rank_input,
            config=config,
            llm=self._llm,
            prompt=prompt,
            enable_rerank=enable_rerank,
            retrieve_fn=retrieve_fn,
            rerank_fn=rerank_fn,
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
    retrieve_fn: RetrieveFn | None = None,
    rerank_fn: RerankFn | None = None,
) -> RankOutput:
    """Case module-level ranker — delegates to ``rerank._basic_arank``."""
    return await _basic_arank(
        rank_input,
        config=config,
        llm=llm,
        prompt=prompt,
        enable_rerank=enable_rerank,
        retrieve_fn=retrieve_fn,
        rerank_fn=rerank_fn,
    )


rank = async_to_sync(arank)
