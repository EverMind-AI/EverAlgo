"""Unit tests for ``everalgo.rank.fusion``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from everalgo.rank import fusion, weight
from everalgo.types import Candidate

if TYPE_CHECKING:
    import pytest

# ─── RRF ────────────────────────────────────────────────────────────────────


def test_rrf_two_lists_same_order_preserves_ranking(
    dense_candidates: list[Candidate],
) -> None:
    out = fusion.rrf(dense_candidates, dense_candidates, k=60)

    # Same doc appears in both lists → score doubled relative to others
    assert out[0].id == "d1"
    # Sum of two 1/(60+1) contributions
    assert math.isclose(out[0].score, 2 / 61)


def test_rrf_one_empty_returns_other(dense_candidates: list[Candidate]) -> None:
    out = fusion.rrf((), dense_candidates, k=60)

    assert [c.id for c in out] == ["d1", "d2", "d3", "d4", "d5"]


def test_rrf_three_way_fusion(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    extra = [Candidate(id="d1", score=99.0), Candidate(id="d8", score=1.0)]

    out = fusion.rrf(dense_candidates, sparse_candidates, extra, k=60)

    ids = [c.id for c in out]
    assert ids[0] == "d1"  # appears in all 3 lists at rank 1
    assert "d8" in ids


def test_rrf_k_parameter_affects_smoothing() -> None:
    a = [Candidate(id="x", score=1.0), Candidate(id="y", score=0.5)]
    b = [Candidate(id="y", score=1.0), Candidate(id="x", score=0.5)]

    small_k = fusion.rrf(a, b, k=1)
    large_k = fusion.rrf(a, b, k=100)

    # With both k values both docs end up tied because they swap rank 1/2; check just magnitude
    assert small_k[0].score > large_k[0].score


def test_rrf_empty_returns_empty() -> None:
    assert fusion.rrf() == []
    assert fusion.rrf((), ()) == []


def test_rrf_drops_doc_with_empty_id() -> None:
    out = fusion.rrf([Candidate(id="", score=0.5), Candidate(id="ok", score=0.4)])

    assert [c.id for c in out] == ["ok"]


# ─── LR ─────────────────────────────────────────────────────────────────────


def test_lr_formula_matches_sigmoid_of_weighted_logit() -> None:
    coefs = weight.LRCoefs(emb_coef=2.0, bm25_coef=3.0, intercept=-1.0)
    emb = [Candidate(id="a", score=0.5)]
    bm25 = [Candidate(id="a", score=0.4)]

    out = fusion.lr(emb, bm25, coefs=coefs)

    logit = 0.5 * 2.0 + 0.4 * 3.0 + (-1.0)
    expected = 1.0 / (1.0 + math.exp(-logit))
    assert math.isclose(out[0].score, expected)


def test_lr_missing_source_treats_score_as_zero() -> None:
    coefs = weight.LRCoefs(emb_coef=1.0, bm25_coef=1.0, intercept=0.0)
    emb = [Candidate(id="emb_only", score=2.0)]
    bm25 = [Candidate(id="bm25_only", score=3.0)]

    out = fusion.lr(emb, bm25, coefs=coefs)

    out_map = {c.id: c.score for c in out}
    assert math.isclose(out_map["emb_only"], 1.0 / (1.0 + math.exp(-2.0)))
    assert math.isclose(out_map["bm25_only"], 1.0 / (1.0 + math.exp(-3.0)))


def test_lr_default_coefs_resolved_from_weight_module() -> None:
    emb = [Candidate(id="x", score=0.5)]
    bm25 = [Candidate(id="x", score=2.0)]

    out_default = fusion.lr(emb, bm25)
    out_explicit = fusion.lr(emb, bm25, coefs=weight.default_lr_coefs())

    assert math.isclose(out_default[0].score, out_explicit[0].score)


def test_lr_monkeypatched_default_coefs_affects_fusion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        weight,
        "default_lr_coefs",
        lambda: weight.LRCoefs(emb_coef=0.0, bm25_coef=0.0, intercept=0.0),
    )

    out = fusion.lr([Candidate(id="x", score=0.9)], [Candidate(id="x", score=0.9)])

    # All-zero coefs → logit = 0 → sigmoid(0) = 0.5
    assert math.isclose(out[0].score, 0.5)


# ─── cosine_to_lr_score ─────────────────────────────────────────────────────


def test_cosine_to_lr_score_default_coefs_match_enterprise() -> None:
    coefs = weight.LRCoefs()
    logit = 0.8 * coefs.emb_coef + 0.0 + coefs.intercept
    expected = 1.0 / (1.0 + math.exp(-logit))

    assert math.isclose(fusion.cosine_to_lr_score(0.8), expected)


def test_cosine_to_lr_score_parent_bm25_increases_probability() -> None:
    a = fusion.cosine_to_lr_score(0.5, parent_bm25=0.0)
    b = fusion.cosine_to_lr_score(0.5, parent_bm25=10.0)

    assert b > a


def test_cosine_to_lr_score_accepts_custom_coefs() -> None:
    custom = weight.LRCoefs(emb_coef=0.0, bm25_coef=0.0, intercept=0.0)

    assert math.isclose(fusion.cosine_to_lr_score(0.99, coefs=custom), 0.5)


# ─── score_propagation ─────────────────────────────────────────────────────


def test_score_propagation_alpha_one_uses_child_only() -> None:
    parents = [Candidate(id="p1", score=0.2)]
    children = [Candidate(id="c1", score=0.7, metadata={"parent_id": "p1"})]

    out = fusion.score_propagation(parents, children, alpha=1.0)

    assert math.isclose(out[0].score, 0.7)


def test_score_propagation_alpha_zero_uses_parent_only() -> None:
    parents = [Candidate(id="p1", score=0.2)]
    children = [Candidate(id="c1", score=0.7, metadata={"parent_id": "p1"})]

    out = fusion.score_propagation(parents, children, alpha=0.0)

    assert math.isclose(out[0].score, 0.2)
