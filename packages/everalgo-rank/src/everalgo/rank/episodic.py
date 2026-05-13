"""Episodic ranker facade — thin wrapper over ``rerank._basic_arank``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.rank.rerank import DEFAULT_RANK_CONFIG, RankConfig, _basic_arank

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.rank.fusion import RerankFn, RetrieveFn
    from everalgo.types import RankInput, RankOutput

__all__ = ["arank", "rank"]


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
    """Episodic ranker facade — see ``rerank._basic_arank`` for the pipeline body."""
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
