"""Unit tests for ``everalgo.rank.profile``."""

from __future__ import annotations

from everalgo.rank import profile
from everalgo.types import Candidate, RankInput


def _mk(dense: list[Candidate], sparse: list[Candidate] | None = None, top_k: int = 5) -> RankInput:
    return RankInput(
        query="user pizza preference",
        memory_type="profile",
        sparse_candidates=sparse or [],
        dense_candidates=dense,
        top_k=top_k,
    )


def test_threshold_filters_low_score_candidates() -> None:
    dense = [
        Candidate(id="p1", score=0.95, source="vector"),
        Candidate(id="p2", score=0.40, source="vector"),
        Candidate(id="p3", score=0.10, source="vector"),
    ]

    out = profile.rank(_mk(dense), threshold=0.5)

    assert [it.id for it in out.items] == ["p1"]


def test_duplicate_ids_collapsed_keeping_highest_score() -> None:
    dense = [
        Candidate(id="p1", score=0.6, source="vector"),
        Candidate(id="p1", score=0.9, source="vector"),
        Candidate(id="p2", score=0.5, source="vector"),
    ]

    out = profile.rank(_mk(dense))

    ids = [it.id for it in out.items]
    assert ids == ["p1", "p2"]
    # p1 retained its highest 0.9 score (sorted first)
    assert out.items[0].score == 0.9


def test_top_k_truncates_result() -> None:
    dense = [Candidate(id=f"p{i}", score=0.9 - 0.01 * i, source="vector") for i in range(10)]

    out = profile.rank(_mk(dense, top_k=3))

    assert len(out.items) == 3
    assert [it.id for it in out.items] == ["p0", "p1", "p2"]


def test_sparse_candidates_are_ignored() -> None:
    dense = [Candidate(id="p1", score=0.8, source="vector")]
    sparse = [Candidate(id="ignore_me", score=999.0, source="keyword")]

    out = profile.rank(_mk(dense, sparse=sparse))

    assert [it.id for it in out.items] == ["p1"]


def test_item_type_is_profile_for_every_item() -> None:
    dense = [Candidate(id="p1", score=0.8), Candidate(id="p2", score=0.7)]

    out = profile.rank(_mk(dense))

    assert all(it.item_type == "profile" for it in out.items)
