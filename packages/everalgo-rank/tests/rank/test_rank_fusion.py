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


# ─── vector_anchored ────────────────────────────────────────────────────────


def test_vector_anchored_overlap_uses_actual_scores() -> None:
    """For a doc present in both sources, score = alpha·cosine + (1-alpha)·(raw/(raw+k))."""
    dense = [Candidate(id="a", score=0.9)]
    sparse = [Candidate(id="a", score=10.0)]

    out = fusion.vector_anchored(dense, sparse, saturation_k=5.0, alpha=0.7)

    expected = 0.7 * 0.9 + 0.3 * (10.0 / (10.0 + 5.0))
    assert len(out) == 1
    assert out[0].id == "a"
    assert math.isclose(out[0].score, expected)


def test_vector_anchored_alpha_one_uses_vec_only() -> None:
    dense = [Candidate(id="a", score=0.9)]
    sparse = [Candidate(id="a", score=10.0)]

    out = fusion.vector_anchored(dense, sparse, alpha=1.0)

    assert math.isclose(out[0].score, 0.9)


def test_vector_anchored_alpha_zero_uses_sat_bm25_only() -> None:
    dense = [Candidate(id="a", score=0.9)]
    sparse = [Candidate(id="a", score=10.0)]

    out = fusion.vector_anchored(dense, sparse, saturation_k=5.0, alpha=0.0)

    assert math.isclose(out[0].score, 10.0 / 15.0)


def test_vector_anchored_saturation_compresses_bm25_into_unit_interval() -> None:
    """Saturation maps any positive BM25 raw into ``[0, 1)``; raw=k maps to ~0.5."""
    dense: list[Candidate] = []
    sparse = [
        Candidate(id="lo", score=1.0),  # 1/6 ≈ 0.167
        Candidate(id="mid", score=5.0),  # 5/10 = 0.5
        Candidate(id="hi", score=100.0),  # 100/105 ≈ 0.952
    ]

    out = fusion.vector_anchored(dense, sparse, saturation_k=5.0, alpha=0.0)

    by_id = {c.id: c.score for c in out}
    assert math.isclose(by_id["lo"], 1.0 / 6.0)
    assert math.isclose(by_id["mid"], 0.5)
    assert math.isclose(by_id["hi"], 100.0 / 105.0)
    assert all(0.0 < s < 1.0 for s in by_id.values())


def test_vector_anchored_missing_dense_uses_vec_floor() -> None:
    """Doc only in sparse → its imputed cosine = min of dense scores ("not recalled" != "not relevant")."""
    dense = [Candidate(id="a", score=0.9), Candidate(id="b", score=0.5)]  # vec_floor = 0.5
    sparse = [Candidate(id="c", score=10.0)]

    # alpha=1 isolates the imputation effect: c's score should equal vec_floor exactly.
    out = fusion.vector_anchored(dense, sparse, saturation_k=5.0, alpha=1.0)

    by_id = {c.id: c.score for c in out}
    assert math.isclose(by_id["c"], 0.5)


def test_vector_anchored_missing_sparse_uses_kw_floor() -> None:
    """Doc only in dense → its imputed sat_bm25 = min of sat-mapped sparse scores."""
    dense = [Candidate(id="a", score=0.9)]
    sparse = [Candidate(id="b", score=10.0), Candidate(id="c", score=2.0)]
    # kw_floor = min(10/15, 2/7) = 2/7
    expected_kw_floor = 2.0 / 7.0

    # alpha=0 isolates the imputation effect: a's score should equal kw_floor exactly.
    out = fusion.vector_anchored(dense, sparse, saturation_k=5.0, alpha=0.0)

    by_id = {c.id: c.score for c in out}
    assert math.isclose(by_id["a"], expected_kw_floor)


def test_vector_anchored_sorted_descending() -> None:
    dense = [Candidate(id="a", score=0.95), Candidate(id="b", score=0.50)]
    sparse = [Candidate(id="a", score=12.0), Candidate(id="c", score=3.0)]

    out = fusion.vector_anchored(dense, sparse, saturation_k=5.0, alpha=0.7)

    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)


def test_vector_anchored_preserves_dense_metadata_over_sparse() -> None:
    """When a doc appears in both, metadata is taken from the dense entry."""
    dense = [Candidate(id="a", score=0.9, metadata={"side": "dense", "extra": 1})]
    sparse = [Candidate(id="a", score=10.0, metadata={"side": "sparse"})]

    out = fusion.vector_anchored(dense, sparse)

    assert out[0].metadata["side"] == "dense"
    assert out[0].metadata["extra"] == 1


def test_vector_anchored_empty_returns_empty() -> None:
    assert fusion.vector_anchored([], []) == []


def test_vector_anchored_drops_doc_with_empty_id() -> None:
    dense = [Candidate(id="", score=0.9), Candidate(id="ok", score=0.5)]
    sparse = [Candidate(id="", score=10.0), Candidate(id="ok", score=4.0)]

    out = fusion.vector_anchored(dense, sparse)

    assert [c.id for c in out] == ["ok"]
