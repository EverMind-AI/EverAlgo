"""Tests for amaxsim_retrieve — child-first MaxSim aggregation retrieval."""

from __future__ import annotations

import numpy as np

from everalgo.rank import amaxsim_retrieve
from everalgo.types import Candidate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_child(child_id: str, score: float, parent_id: object) -> Candidate:
    """Build a child Candidate with a given parent_id in metadata."""
    meta: dict[str, object] = {"parent_id": parent_id} if parent_id is not None else {}
    return Candidate(id=child_id, score=score, source="vector", metadata=meta)


def _make_parent(parent_id: str, text: str = "") -> Candidate:
    return Candidate(id=parent_id, score=0.0, source="other", metadata={"text": text})


# ---------------------------------------------------------------------------
# U1 — bm25-shape bit-for-bit
# ---------------------------------------------------------------------------


async def test_bm25_shape_bit_for_bit() -> None:
    """Max-pool result matches benchmark search_with_bm25_index path bit-for-bit.

    fact_scores=[0.5, 0.9, 0.3, 0.7], fact_to_doc_idx=["d0","d1","d0","d1"]
    => d1 max=0.9, d0 max=0.5 (strict >).
    """
    fact_scores = [0.5, 0.9, 0.3, 0.7]
    fact_to_doc_idx = ["d0", "d1", "d0", "d1"]

    async def child_retrieve(q: str, k: int) -> list[Candidate]:
        return [_make_child(f"fact_{i}", float(s), fact_to_doc_idx[i]) for i, s in enumerate(fact_scores)]

    async def parent_fetch(ids: list[str]) -> list[Candidate]:
        return [_make_parent(pid, text=f"doc {pid}") for pid in ids]

    results = await amaxsim_retrieve(
        "test query",
        child_retrieve=child_retrieve,
        parent_fetch=parent_fetch,
        top_n=2,
    )

    ids = [c.id for c in results]
    scores = [c.score for c in results]

    assert ids == ["d1", "d0"]
    assert scores == [0.9, 0.5]  # bit-exact float comparison


# ---------------------------------------------------------------------------
# U2 — emb-shape max-pool equivalent to np.max
# ---------------------------------------------------------------------------


async def test_emb_shape_maxpool_equals_numpy_max() -> None:
    """Max-pool score per parent equals float(np.max(sims)) for that parent — bit-exact."""
    rng = np.random.default_rng(42)

    query_vec = rng.random(8).astype(np.float64)
    query_vec /= np.linalg.norm(query_vec)

    # 3 parents, 4 facts each (12 facts total)
    parent_ids_for_facts = ["p0"] * 4 + ["p1"] * 4 + ["p2"] * 4
    fact_vecs = rng.random((12, 8)).astype(np.float64)
    fact_vecs /= np.linalg.norm(fact_vecs, axis=1, keepdims=True)

    cosine_sims = (fact_vecs @ query_vec).tolist()

    async def child_retrieve(q: str, k: int) -> list[Candidate]:
        return [_make_child(f"fact_{i}", float(cosine_sims[i]), parent_ids_for_facts[i]) for i in range(12)]

    async def parent_fetch(ids: list[str]) -> list[Candidate]:
        return [_make_parent(pid) for pid in ids]

    results = await amaxsim_retrieve(
        "emb query",
        child_retrieve=child_retrieve,
        parent_fetch=parent_fetch,
        top_n=10,
    )

    # Compute expected max per parent using numpy
    for parent_label, slice_start in [("p0", 0), ("p1", 4), ("p2", 8)]:
        expected_max = float(np.max(cosine_sims[slice_start : slice_start + 4]))
        result_score = next((c.score for c in results if c.id == parent_label), None)
        assert result_score is not None, f"Parent {parent_label} missing from results"
        assert result_score == expected_max, f"Score mismatch for {parent_label}: {result_score} != {expected_max}"


# ---------------------------------------------------------------------------
# U3 — boundary cases
# ---------------------------------------------------------------------------


async def test_empty_children_returns_empty() -> None:
    """child_retrieve returning [] causes immediate [] return — no parent_fetch call."""
    fetch_called = False

    async def child_retrieve(q: str, k: int) -> list[Candidate]:
        return []

    async def parent_fetch(ids: list[str]) -> list[Candidate]:
        nonlocal fetch_called
        fetch_called = True
        return []

    result = await amaxsim_retrieve("q", child_retrieve=child_retrieve, parent_fetch=parent_fetch)

    assert result == []
    assert not fetch_called, "parent_fetch must not be called when children is empty"


async def test_missing_parent_id_skipped() -> None:
    """Children with no parent_id or a non-str parent_id are silently skipped."""

    async def child_retrieve(q: str, k: int) -> list[Candidate]:
        return [
            Candidate(id="f0", score=0.8, source="vector", metadata={"parent_id": "p0"}),  # valid
            Candidate(id="f1", score=0.9, source="vector", metadata={}),  # missing key
            Candidate(id="f2", score=0.7, source="vector", metadata={"parent_id": None}),  # None
            Candidate(id="f3", score=1.0, source="vector", metadata={"parent_id": 42}),  # int
        ]

    async def parent_fetch(ids: list[str]) -> list[Candidate]:
        return [_make_parent(pid) for pid in ids]

    results = await amaxsim_retrieve("q", child_retrieve=child_retrieve, parent_fetch=parent_fetch)

    assert len(results) == 1
    assert results[0].id == "p0"
    assert results[0].score == 0.8


async def test_all_children_missing_parent_id_returns_empty() -> None:
    """When every child lacks a valid string parent_id, parent_max is empty and [] is returned immediately."""
    fetch_called = False

    async def child_retrieve(q: str, k: int) -> list[Candidate]:
        return [
            Candidate(id="f0", score=0.9, source="vector", metadata={}),  # no parent_id
            Candidate(id="f1", score=0.8, source="vector", metadata={"parent_id": None}),
            Candidate(id="f2", score=0.7, source="vector", metadata={"parent_id": 99}),
        ]

    async def parent_fetch(ids: list[str]) -> list[Candidate]:
        nonlocal fetch_called
        fetch_called = True
        return []

    result = await amaxsim_retrieve("q", child_retrieve=child_retrieve, parent_fetch=parent_fetch)

    assert result == []
    assert not fetch_called, "parent_fetch must not be called when parent_max is empty"


async def test_parent_fetch_returns_subset() -> None:
    """parent_fetch may return fewer parents than requested; only returned ones appear in result."""
    fetched_ids: list[list[str]] = []

    async def child_retrieve(q: str, k: int) -> list[Candidate]:
        return [
            _make_child("f0", 0.9, "d0"),
            _make_child("f1", 0.8, "d1"),
            _make_child("f2", 0.7, "d2"),
        ]

    async def parent_fetch(ids: list[str]) -> list[Candidate]:
        fetched_ids.append(list(ids))
        # Intentionally omit d1
        return [_make_parent(pid) for pid in ids if pid != "d1"]

    results = await amaxsim_retrieve("q", child_retrieve=child_retrieve, parent_fetch=parent_fetch)

    result_ids = [c.id for c in results]
    assert "d1" not in result_ids
    assert set(result_ids) == {"d0", "d2"}
    # Scores must come from score_by_id (max-pool), not parent_fetch's score=0.0
    score_by_result = {c.id: c.score for c in results}
    assert score_by_result["d0"] == 0.9
    assert score_by_result["d2"] == 0.7


async def test_min_score_filters_parents() -> None:
    """Parents with max-pool score < min_score are excluded from the final result."""

    async def child_retrieve(q: str, k: int) -> list[Candidate]:
        return [
            _make_child("f0", 0.9, "high"),
            _make_child("f1", 0.3, "low"),
        ]

    async def parent_fetch(ids: list[str]) -> list[Candidate]:
        return [_make_parent(pid) for pid in ids]

    results = await amaxsim_retrieve(
        "q",
        child_retrieve=child_retrieve,
        parent_fetch=parent_fetch,
        min_score=0.5,
    )

    assert len(results) == 1
    assert results[0].id == "high"
    assert results[0].score == 0.9


# ---------------------------------------------------------------------------
# U4 — tie-breaking: first-seen (insertion order) preserved by stable Timsort
# ---------------------------------------------------------------------------


async def test_tie_breaking_preserves_first_seen_order() -> None:
    """When two parents have identical max-pool scores, the first-inserted parent stays first.

    dict insertion order is stable in Python 3.7+; sorted() (Timsort) is stable, so
    equal-score items retain their relative iteration order — which equals insertion order.
    We control child order so d0 populates parent_max before d1.
    """

    async def child_retrieve(q: str, k: int) -> list[Candidate]:
        # d0 appears first in the list → inserted first into parent_max
        return [
            _make_child("f_d0", 0.7, "d0"),
            _make_child("f_d1", 0.7, "d1"),
        ]

    async def parent_fetch(ids: list[str]) -> list[Candidate]:
        return [_make_parent(pid) for pid in ids]

    results = await amaxsim_retrieve("q", child_retrieve=child_retrieve, parent_fetch=parent_fetch)

    assert len(results) == 2
    # Both have score 0.7; stable sort preserves d0-first insertion order
    assert results[0].id == "d0"
    assert results[1].id == "d1"
    assert results[0].score == 0.7
    assert results[1].score == 0.7


# ---------------------------------------------------------------------------
# U5 — child_candidates pass-through
# ---------------------------------------------------------------------------


async def test_child_candidates_passthrough_default() -> None:
    """child_retrieve receives k == child_candidates (default 200)."""
    received_k: list[int] = []

    async def child_retrieve(q: str, k: int) -> list[Candidate]:
        received_k.append(k)
        return [_make_child("f0", 0.5, "p0")]

    async def parent_fetch(ids: list[str]) -> list[Candidate]:
        return [_make_parent(pid) for pid in ids]

    await amaxsim_retrieve("q", child_retrieve=child_retrieve, parent_fetch=parent_fetch)

    assert received_k == [200]


async def test_child_candidates_passthrough_custom() -> None:
    """child_retrieve receives k == child_candidates when explicitly set to 300."""
    received_k: list[int] = []

    async def child_retrieve(q: str, k: int) -> list[Candidate]:
        received_k.append(k)
        return [_make_child("f0", 0.5, "p0")]

    async def parent_fetch(ids: list[str]) -> list[Candidate]:
        return [_make_parent(pid) for pid in ids]

    await amaxsim_retrieve("q", child_retrieve=child_retrieve, parent_fetch=parent_fetch, child_candidates=300)

    assert received_k == [300]
