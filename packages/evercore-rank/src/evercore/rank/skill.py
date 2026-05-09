"""Skill ranker facade — fusion → maturity + confidence weighted → rerank. Stub."""

from __future__ import annotations

from evercore.types import RankInput, RankOutput

__all__ = ["arank", "rank"]


async def arank(rank_input: RankInput) -> RankOutput:
    """Stub: returns placeholder."""
    raise NotImplementedError("stub")


def rank(rank_input: RankInput) -> RankOutput:
    """Stub: returns placeholder."""
    raise NotImplementedError("stub")
