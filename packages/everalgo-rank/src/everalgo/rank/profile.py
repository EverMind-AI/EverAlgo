"""Profile ranker facade — cosine + threshold + dedup (sync only)."""

from __future__ import annotations

from everalgo.types import RankInput, RankOutput, ScoredItem

__all__ = ["rank"]


def rank(
    rank_input: RankInput,
    *,
    threshold: float = 0.0,
) -> RankOutput:
    """Profile ranker facade.

    Steps:
    1. Read ``rank_input.dense_candidates`` (already sorted by cosine descending
       per the caller's contract; we re-sort defensively).
    2. Drop candidates with ``score < threshold``.
    3. Deduplicate by ``id`` keeping the first (highest-score) occurrence.
    4. Truncate to ``rank_input.top_k``.

    Args:
        rank_input: ``dense_candidates`` is the only source consulted.
            ``sparse_candidates`` and ``episode_to_facts`` are ignored.
        threshold: Minimum score to keep. Default ``0.0`` (no filter); callers
            often pass ``0.65`` or similar based on their embedder.

    Returns
    -------
        ``RankOutput`` with ``item_type='profile'``.
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
