"""Profile ranker facade — cosine + threshold + dedup (sync only). Stub.

profile facade does not provide async interface (pure compute, no LLM).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everalgo.types import RankInput, RankOutput

__all__ = ["rank"]


def rank(rank_input: RankInput) -> RankOutput:
    """Stub: returns placeholder."""
    raise NotImplementedError("stub")
