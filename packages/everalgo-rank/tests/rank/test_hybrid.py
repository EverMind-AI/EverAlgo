"""Tests for ahybrid_retrieve — dual-route RRF / LR fusion."""

from __future__ import annotations

from everalgo.rank import ahybrid_retrieve
from everalgo.rank.fusion import lr
from everalgo.rank.weight import LRCoefs
from everalgo.types import Candidate


async def test_hybrid_dual_route_rrf_fusion() -> None:
    """Candidates overlapping both routes get highest fused RRF score."""

    async def dense_retrieve(q: str, k: int) -> list[Candidate]:
        return [
            Candidate(id="d1", score=0.9, source="vector", metadata={}),
            Candidate(id="d2", score=0.8, source="vector", metadata={}),
            Candidate(id="d3", score=0.7, source="vector", metadata={}),
        ]

    async def sparse_retrieve(q: str, k: int) -> list[Candidate]:
        return [
            Candidate(id="s1", score=10.0, source="keyword", metadata={}),
            Candidate(id="d2", score=8.0, source="keyword", metadata={}),  # overlaps dense
            Candidate(id="s2", score=6.0, source="keyword", metadata={}),
        ]

    results = await ahybrid_retrieve(
        query="q",
        dense_retrieve=dense_retrieve,
        sparse_retrieve=sparse_retrieve,
        top_n=5,
    )
    ids = [c.id for c in results]
    # d2 appears in both → highest fused RRF score
    assert ids[0] == "d2"
    assert len(results) <= 5


async def test_hybrid_empty_inputs_returns_empty() -> None:
    async def empty(q: str, k: int) -> list[Candidate]:
        return []

    results = await ahybrid_retrieve(
        query="q",
        dense_retrieve=empty,
        sparse_retrieve=empty,
        top_n=5,
    )
    assert results == []


async def test_hybrid_one_route_empty_returns_other() -> None:
    async def dense_retrieve(q: str, k: int) -> list[Candidate]:
        return [Candidate(id="d1", score=0.9, source="vector", metadata={})]

    async def empty(q: str, k: int) -> list[Candidate]:
        return []

    results = await ahybrid_retrieve(
        query="q",
        dense_retrieve=dense_retrieve,
        sparse_retrieve=empty,
        top_n=5,
    )
    assert [c.id for c in results] == ["d1"]


async def test_hybrid_dense_empty_returns_sparse() -> None:
    async def empty(q: str, k: int) -> list[Candidate]:
        return []

    async def sparse_retrieve(q: str, k: int) -> list[Candidate]:
        return [Candidate(id="s1", score=10.0, source="keyword", metadata={})]

    results = await ahybrid_retrieve(
        query="q",
        dense_retrieve=empty,
        sparse_retrieve=sparse_retrieve,
        top_n=5,
    )
    assert [c.id for c in results] == ["s1"]


# ─── New tests: LR fusion + min_score ───────────────────────────────────────


async def test_hybrid_lr_fusion() -> None:
    """fusion="lr" result equals manual fusion.lr(dense, sparse) with same inputs."""
    dense_list = [
        Candidate(id="d1", score=0.9, source="vector", metadata={}),
        Candidate(id="d2", score=0.8, source="vector", metadata={}),
    ]
    sparse_list = [
        Candidate(id="d2", score=8.0, source="keyword", metadata={}),
        Candidate(id="s1", score=6.0, source="keyword", metadata={}),
    ]

    async def dense_retrieve(q: str, k: int) -> list[Candidate]:
        return dense_list

    async def sparse_retrieve(q: str, k: int) -> list[Candidate]:
        return sparse_list

    results = await ahybrid_retrieve(
        query="q",
        dense_retrieve=dense_retrieve,
        sparse_retrieve=sparse_retrieve,
        fusion="lr",
        top_n=10,
    )
    expected = lr(dense_list, sparse_list)[:10]
    assert len(results) == len(expected)
    for got, exp in zip(results, expected, strict=True):
        assert got.id == exp.id
        assert abs(got.score - exp.score) < 1e-9


async def test_hybrid_lr_fusion_with_coefs() -> None:
    """Custom LRCoefs are forwarded to fusion.lr; different coefs produce different scores."""
    dense_list = [
        Candidate(id="a", score=0.7, source="vector", metadata={}),
        Candidate(id="b", score=0.5, source="vector", metadata={}),
    ]
    sparse_list = [
        Candidate(id="a", score=3.0, source="keyword", metadata={}),
        Candidate(id="b", score=9.0, source="keyword", metadata={}),
    ]

    async def dense_retrieve(q: str, k: int) -> list[Candidate]:
        return dense_list

    async def sparse_retrieve(q: str, k: int) -> list[Candidate]:
        return sparse_list

    custom_coefs = LRCoefs(emb_coef=1.0, bm25_coef=5.0, intercept=-3.0)
    default_results = await ahybrid_retrieve(
        query="q",
        dense_retrieve=dense_retrieve,
        sparse_retrieve=sparse_retrieve,
        fusion="lr",
        top_n=10,
    )
    custom_results = await ahybrid_retrieve(
        query="q",
        dense_retrieve=dense_retrieve,
        sparse_retrieve=sparse_retrieve,
        fusion="lr",
        lr_coefs=custom_coefs,
        top_n=10,
    )
    # Custom coefs weight bm25 heavily; b (bm25=9.0) should beat a (bm25=3.0) under custom_coefs
    assert custom_results[0].id == "b"
    # Scores differ between default and custom coefs
    default_score_a = next(c.score for c in default_results if c.id == "a")
    custom_score_a = next(c.score for c in custom_results if c.id == "a")
    assert abs(default_score_a - custom_score_a) > 1e-6


async def test_hybrid_min_score_filters() -> None:
    """min_score removes candidates with score strictly below the threshold after top_n truncation."""
    dense_list = [
        Candidate(id="high", score=0.9, source="vector", metadata={}),
        Candidate(id="mid", score=0.5, source="vector", metadata={}),
        Candidate(id="low", score=0.1, source="vector", metadata={}),
    ]

    async def dense_retrieve(q: str, k: int) -> list[Candidate]:
        return dense_list

    async def empty(q: str, k: int) -> list[Candidate]:
        return []

    results = await ahybrid_retrieve(
        query="q",
        dense_retrieve=dense_retrieve,
        sparse_retrieve=empty,
        top_n=10,
        min_score=0.5,
    )
    scores = [c.score for c in results]
    assert all(s >= 0.5 for s in scores)
    ids = [c.id for c in results]
    assert "low" not in ids
    assert "high" in ids
    assert "mid" in ids


async def test_hybrid_default_still_rrf() -> None:
    """Without fusion/lr_coefs/min_score, result equals rrf(dense, sparse)[:top_n] exactly."""
    from everalgo.rank.fusion import rrf

    dense_list = [
        Candidate(id="d1", score=0.9, source="vector", metadata={}),
        Candidate(id="d2", score=0.7, source="vector", metadata={}),
    ]
    sparse_list = [
        Candidate(id="s1", score=5.0, source="keyword", metadata={}),
        Candidate(id="d2", score=4.0, source="keyword", metadata={}),
    ]

    async def dense_retrieve(q: str, k: int) -> list[Candidate]:
        return dense_list

    async def sparse_retrieve(q: str, k: int) -> list[Candidate]:
        return sparse_list

    results = await ahybrid_retrieve(
        query="q",
        dense_retrieve=dense_retrieve,
        sparse_retrieve=sparse_retrieve,
        top_n=5,
    )
    expected = rrf(dense_list, sparse_list)[:5]
    assert len(results) == len(expected)
    for got, exp in zip(results, expected, strict=True):
        assert got.id == exp.id
        assert abs(got.score - exp.score) < 1e-9
