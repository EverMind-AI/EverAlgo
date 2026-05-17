"""Profile ranker facade — cosine + threshold + dedup (sync only)."""

from __future__ import annotations

from everalgo.types import RankInput, RankOutput, ScoredItem

__all__ = ["rank"]


def rank(
    rank_input: RankInput,
    *,
    threshold: float = 0.0,
) -> RankOutput:
    """Sort dense candidates, drop below threshold, dedup by id, truncate to top_k.

    Only ``dense_candidates`` is consulted; ``sparse_candidates`` and ``episode_to_facts`` are ignored.
    """
    candidates = sorted(rank_input.dense_candidates, key=lambda c: c.score, reverse=True)

    seen: set[str] = set()
    items: list[ScoredItem] = []
    for c in candidates:
        if c.score < threshold:
            continue
        if c.id in seen:
            continue
        seen.add(c.id)
        items.append(
            ScoredItem(
                id=c.id,
                score=c.score,
                item_type="profile",
                metadata=dict(c.metadata),
            )
        )
        if len(items) >= rank_input.top_k:
            break

    return RankOutput(items=items, metadata={"stage": "profile"})
