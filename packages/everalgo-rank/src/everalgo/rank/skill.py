"""Skill ranker facade — fusion → maturity + confidence weighted → rerank. Stub."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everalgo.types import RankInput, RankOutput

__all__ = ["arank", "rank"]


async def arank(rank_input: RankInput) -> RankOutput:
    """Stub: returns placeholder."""
    raise NotImplementedError("stub")


def rank(rank_input: RankInput) -> RankOutput:
    """Stub: returns placeholder."""
    raise NotImplementedError("stub")
