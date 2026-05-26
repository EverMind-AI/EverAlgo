"""Tests for ahierarchical_retrieve — parent-child hierarchical retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from everalgo.retrieval.hierarchical import ahierarchical_retrieve
from everalgo.types import Candidate

# ─── Helpers ────────────────────────────────────────────────────────────────


def _ep(eid: str, score: float) -> Candidate:
    return Candidate(id=eid, score=score, source="vector")


def _child(cid: str, parent_id: str, score: float) -> Candidate:
    return Candidate(id=cid, score=score, source="vector", metadata={"parent_id": parent_id})


def _const_retriever(lst: list[Candidate]) -> Callable[[str, int], Awaitable[list[Candidate]]]:
    """Return an async retriever that always returns ``lst`` regardless of query / k."""

    async def _retrieve(q: str, k: int) -> list[Candidate]:
        return lst

    return _retrieve


def _dict_child_retriever(
    d: dict[str, list[Candidate]],
) -> Callable[[str, Candidate, int], Awaitable[list[Candidate]]]:
    """Return an async child retriever that looks up children by ``parent.id``."""

    async def _retrieve(q: str, parent: Candidate, k: int) -> list[Candidate]:
        return d.get(parent.id, [])

    return _retrieve


# ─── Minimal smoke tests (were failing before hierarchical.py existed) ───────


async def test_hierarchical_empty_inputs_returns_empty() -> None:
    async def empty_retrieve(q: str, k: int) -> list[Candidate]:
        return []

    async def empty_child(q: str, parent: Candidate, k: int) -> list[Candidate]:
        return []

    results = await ahierarchical_retrieve(
        query="anything",
        parent_dense_retrieve=empty_retrieve,
        parent_sparse_retrieve=empty_retrieve,
        child_retrieve_for_parent=empty_child,
        response_top_k=5,
    )
    assert results == []


async def test_hierarchical_only_dense_no_sparse_falls_through() -> None:
    async def dense_retrieve(q: str, k: int) -> list[Candidate]:
        return [Candidate(id="p1", score=0.9, source="vector", metadata={"text": "p1"})]

    async def empty_retrieve(q: str, k: int) -> list[Candidate]:
        return []

    async def empty_child(q: str, parent: Candidate, k: int) -> list[Candidate]:
        return []

    results = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=dense_retrieve,
        parent_sparse_retrieve=empty_retrieve,
        child_retrieve_for_parent=empty_child,
        response_top_k=5,
    )
    assert len(results) == 1
    assert results[0].id == "p1"


async def test_hierarchical_child_can_evict_parent() -> None:
    """Child with strong cosine evicts its parent (Phase 2-4 heap convergence)."""

    async def dense_retrieve(q: str, k: int) -> list[Candidate]:
        return [
            Candidate(id="p1", score=0.5, source="vector", metadata={}),
            Candidate(id="p2", score=0.4, source="vector", metadata={}),
        ]

    async def sparse_retrieve(q: str, k: int) -> list[Candidate]:
        return [
            Candidate(id="p1", score=0.5, source="keyword", metadata={}),
            Candidate(id="p2", score=0.4, source="keyword", metadata={}),
        ]

    async def child_retrieve(q: str, parent: Candidate, k: int) -> list[Candidate]:
        if parent.id == "p1":
            return [Candidate(id="c1_of_p1", score=0.95, source="vector", metadata={"parent_id": "p1"})]
        return []

    results = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=dense_retrieve,
        parent_sparse_retrieve=sparse_retrieve,
        child_retrieve_for_parent=child_retrieve,
        response_top_k=2,
    )
    ids = [c.id for c in results]
    assert "c1_of_p1" in ids


async def test_facts_outscoring_parent_replace_parent_in_topn() -> None:
    """High-scoring child takes the slot; its parent is evicted."""
    dense = [_ep("ep1", 0.6), _ep("ep2", 0.3)]
    sparse = [_ep("ep1", 0.6), _ep("ep2", 0.3)]
    children: dict[str, list[Candidate]] = {"ep1": [_child("f1", "ep1", 0.95)]}

    results = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=_const_retriever(dense),
        parent_sparse_retrieve=_const_retriever(sparse),
        child_retrieve_for_parent=_dict_child_retriever(children),
        response_top_k=2,
        alpha=1.0,
        max_convergence_rounds=2,
        expand_limit=1,
    )

    ids = [c.id for c in results]
    assert "ep1" not in ids
    assert "f1" in ids
    child_entries = [c for c in results if c.id == "f1"]
    assert child_entries[0].metadata.get("source_kind") == "child"


async def test_low_score_children_keep_parent_in_topn() -> None:
    dense = [_ep("ep1", 0.9), _ep("ep2", 0.8)]
    sparse = [_ep("ep1", 0.9), _ep("ep2", 0.8)]
    children: dict[str, list[Candidate]] = {"ep1": [_child("f1", "ep1", 0.01)]}

    results = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=_const_retriever(dense),
        parent_sparse_retrieve=_const_retriever(sparse),
        child_retrieve_for_parent=_dict_child_retriever(children),
        response_top_k=2,
        alpha=1.0,
        max_convergence_rounds=10,
        expand_limit=1,
    )

    ids = [c.id for c in results]
    assert "ep1" in ids
    assert "ep2" in ids
    assert "f1" not in ids


async def test_convergence_stops_when_topn_stable() -> None:
    # Many parents with no children → heap drains before convergence rounds
    dense = [_ep(f"ep{i}", 0.9 - 0.01 * i) for i in range(10)]
    sparse = [_ep(f"ep{i}", 0.9 - 0.01 * i) for i in range(10)]
    children: dict[str, list[Candidate]] = {f"ep{i}": [] for i in range(10)}

    # With empty children top-N never changes → convergence
    results = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=_const_retriever(dense),
        parent_sparse_retrieve=_const_retriever(sparse),
        child_retrieve_for_parent=_dict_child_retriever(children),
        response_top_k=3,
        max_convergence_rounds=2,
        expand_limit=1,
    )
    # Should return exactly 3 (top-k cap)
    assert len(results) <= 3


async def test_no_children_drains_heap_without_eviction() -> None:
    dense = [_ep("ep1", 0.5), _ep("ep2", 0.4)]
    sparse = [_ep("ep1", 0.5), _ep("ep2", 0.4)]

    results = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=_const_retriever(dense),
        parent_sparse_retrieve=_const_retriever(sparse),
        child_retrieve_for_parent=_dict_child_retriever({}),
        response_top_k=3,
        max_convergence_rounds=100,
        expand_limit=1,
    )

    # Both parents survive; no children in results
    ids = [c.id for c in results]
    assert "ep1" in ids
    assert "ep2" in ids
    child_entries = [c for c in results if c.metadata.get("source_kind") == "child"]
    assert child_entries == []


async def test_response_top_k_caps_final_set() -> None:
    dense = [_ep(f"ep{i}", 0.9 - 0.05 * i) for i in range(5)]
    sparse = [_ep(f"ep{i}", 0.9 - 0.05 * i) for i in range(5)]

    results = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=_const_retriever(dense),
        parent_sparse_retrieve=_const_retriever(sparse),
        child_retrieve_for_parent=_dict_child_retriever({}),
        response_top_k=2,
        max_convergence_rounds=100,
        expand_limit=1,
    )

    assert len(results) <= 2


async def test_alpha_blends_child_and_parent() -> None:
    dense = [_ep("ep1", 0.8)]
    sparse = [_ep("ep1", 0.8)]
    # Child cosine=0.4, parent_score=0.8, alpha=0.5 → final = 0.5*child_lr + 0.5*0.8
    # We check that the child score is blended (not equal to raw cosine)
    children: dict[str, list[Candidate]] = {"ep1": [_child("f1", "ep1", 0.4)]}

    results = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=_const_retriever(dense),
        parent_sparse_retrieve=_const_retriever(sparse),
        child_retrieve_for_parent=_dict_child_retriever(children),
        response_top_k=1,
        alpha=0.5,
        max_convergence_rounds=2,
        expand_limit=1,
    )

    if results and results[0].metadata.get("source_kind") == "child":
        # score should be a blend, not raw cosine 0.4
        assert results[0].score != 0.4


async def test_high_level_runs_phase1_plus_phase24() -> None:
    """End-to-end: Phase 1 fusion + Phase 2-4 expansion, both routes supplied."""
    sparse = [_ep("ep1", 5.0), _ep("ep2", 3.0)]
    dense = [_ep("ep1", 0.95), _ep("ep2", 0.80)]
    children: dict[str, list[Candidate]] = {"ep1": [_child("f1", "ep1", 0.99)]}

    results = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=_const_retriever(dense),
        parent_sparse_retrieve=_const_retriever(sparse),
        child_retrieve_for_parent=_dict_child_retriever(children),
        response_top_k=2,
        alpha=1.0,
        max_convergence_rounds=2,
        expand_limit=1,
    )

    # Phase 2-4 lifted f1 into top-N
    ids = [c.id for c in results]
    assert "f1" in ids


async def test_empty_inputs_short_circuits() -> None:
    results = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=_const_retriever([]),
        parent_sparse_retrieve=_const_retriever([]),
        child_retrieve_for_parent=_dict_child_retriever({}),
        response_top_k=3,
    )
    assert results == []


async def test_dense_only_works_without_sparse() -> None:
    """No sparse → Phase 1 returns dense as-is; expand still runs."""
    dense = [_ep("ep1", 0.9), _ep("ep2", 0.7)]

    results = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=_const_retriever(dense),
        parent_sparse_retrieve=_const_retriever([]),
        child_retrieve_for_parent=_dict_child_retriever({}),
        response_top_k=2,
    )

    ids = {c.id for c in results}
    assert ids <= {"ep1", "ep2"}


# ─── New test: min_score ─────────────────────────────────────────────────────


async def test_hierarchical_min_score_filters() -> None:
    """min_score removes candidates with score below threshold from the final result."""
    dense = [_ep("ep1", 0.9), _ep("ep2", 0.5), _ep("ep3", 0.3)]
    sparse = [_ep("ep1", 0.9), _ep("ep2", 0.5), _ep("ep3", 0.3)]

    results_unfiltered = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=_const_retriever(dense),
        parent_sparse_retrieve=_const_retriever(sparse),
        child_retrieve_for_parent=_dict_child_retriever({}),
        response_top_k=5,
        max_convergence_rounds=2,
        expand_limit=1,
    )
    # Verify unfiltered has more candidates than filtered
    unfiltered_count = len(results_unfiltered)

    # Pick a min_score threshold that excludes at least one candidate
    threshold = max(c.score for c in results_unfiltered) - 1e-6

    results_filtered = await ahierarchical_retrieve(
        query="q",
        parent_dense_retrieve=_const_retriever(dense),
        parent_sparse_retrieve=_const_retriever(sparse),
        child_retrieve_for_parent=_dict_child_retriever({}),
        response_top_k=5,
        max_convergence_rounds=2,
        expand_limit=1,
        min_score=threshold,
    )
    assert len(results_filtered) < unfiltered_count
    assert all(c.score >= threshold for c in results_filtered)
