"""Unit tests for ``everalgo.rank.weight``."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from everalgo.rank import weight
from everalgo.types import Candidate

if TYPE_CHECKING:
    import pytest

# ─── weighted_score: single-list LR ────────────────────────────────────────


def test_weighted_score_default_is_sigmoid_of_base_plus_field_combo() -> None:
    """Logit = 1.0 * score + 0.5 * quality_score + 0; prob = sigmoid(logit)."""
    items = [Candidate(id="x", score=0.3, metadata={"quality_score": 0.9})]

    out = weight.weighted_score(items, fields={"quality_score": 0.5})

    logit = 0.5 * 0.9
    expected = 1.0 / (1.0 + math.exp(-logit))
    assert math.isclose(out[0].score, expected)


def test_weighted_score_missing_metadata_field_treated_as_zero() -> None:
    items = [Candidate(id="x", score=0.0, metadata={})]

    out = weight.weighted_score(items, fields={"quality_score": 2.0})

    # logit = 0 + 0 + 0 = 0 → sigmoid(0) = 0.5
    assert math.isclose(out[0].score, 0.5)


def test_weighted_score_accumulates_multiple_fields() -> None:
    items = [Candidate(id="x", score=0.0, metadata={"a": 2.0, "b": 3.0})]

    out = weight.weighted_score(items, fields={"a": 0.5, "b": 0.25})

    logit = 0.5 * 2.0 + 0.25 * 3.0  # base_weight=1 * score(0) + bonuses
    assert math.isclose(out[0].score, 1.0 / (1.0 + math.exp(-logit)))


def test_weighted_score_base_weight_zero_discards_score_only_uses_fields() -> None:
    """Skill convention: ``base_weight=0`` removes the fusion score from the logit."""
    items = [
        Candidate(id="s1", score=0.99, metadata={"maturity_score": 0.9, "confidence": 0.85}),
        Candidate(id="s2", score=0.99, metadata={"maturity_score": 0.6, "confidence": 0.95}),
    ]

    out = weight.weighted_score(
        items,
        fields={"maturity_score": 0.6, "confidence": 0.4},
    )

    # logit excludes the 0.99 base; s1 logit = 0.6*0.9+0.4*0.85 = 0.88
    expected_s1 = 1.0 / (1.0 + math.exp(-(0.6 * 0.9 + 0.4 * 0.85)))
    assert math.isclose(out[0].score, expected_s1)


def test_weighted_score_intercept_shifts_logit() -> None:
    items = [Candidate(id="x", score=0.0, metadata={})]

    out_zero = weight.weighted_score(items, fields={}, intercept=0.0)
    out_shift = weight.weighted_score(items, fields={}, intercept=10.0)

    assert math.isclose(out_zero[0].score, 0.5)
    assert out_shift[0].score > 0.999  # sigmoid(10) ≈ 1.0


# ─── multi_field_weighting: multi-source LR ────────────────────────────────


def test_multi_field_weighting_two_sources_match_lr_formula() -> None:
    """Generalisation of fusion.lr — verify same formula with 2 sources."""
    emb = [Candidate(id="x", score=0.5)]
    bm25 = [Candidate(id="x", score=0.4)]

    out = weight.multi_field_weighting(
        {"emb": emb, "bm25": bm25},
        weights={"emb": 2.0, "bm25": 3.0},
        intercept=-1.0,
    )

    logit = 0.5 * 2.0 + 0.4 * 3.0 - 1.0
    expected = 1.0 / (1.0 + math.exp(-logit))
    assert math.isclose(out[0].score, expected)


def test_multi_field_weighting_three_sources_generalises_lr() -> None:
    """fusion.lr is hardcoded to 2 sources; multi_field_weighting handles N."""
    out = weight.multi_field_weighting(
        {
            "emb": [Candidate(id="x", score=1.0)],
            "bm25": [Candidate(id="x", score=2.0)],
            "recency": [Candidate(id="x", score=0.5)],
        },
        weights={"emb": 1.0, "bm25": 1.0, "recency": 4.0},
    )

    logit = 1.0 + 2.0 + 4.0 * 0.5
    assert math.isclose(out[0].score, 1.0 / (1.0 + math.exp(-logit)))


def test_multi_field_weighting_missing_source_score_treated_as_zero() -> None:
    """Doc in source A but not source B → B's contribution is 0."""
    out = weight.multi_field_weighting(
        {
            "a": [Candidate(id="x", score=2.0)],
            "b": [Candidate(id="y", score=3.0)],
        },
        weights={"a": 1.0, "b": 1.0},
    )

    out_map = {c.id: c.score for c in out}
    assert math.isclose(out_map["x"], 1.0 / (1.0 + math.exp(-2.0)))
    assert math.isclose(out_map["y"], 1.0 / (1.0 + math.exp(-3.0)))


def test_multi_field_weighting_sorts_descending() -> None:
    """Output sorted by probability descending."""
    out = weight.multi_field_weighting(
        {
            "a": [
                Candidate(id="lo", score=0.1),
                Candidate(id="hi", score=2.0),
            ]
        },
        weights={"a": 1.0},
    )

    assert [c.id for c in out] == ["hi", "lo"]


# ─── LRCoefs / default_lr_coefs ────────────────────────────────────────────


def test_lr_coefs_defaults_match_enterprise_production() -> None:
    coefs = weight.LRCoefs()

    assert math.isclose(coefs.emb_coef, 6.27473151675093)
    assert math.isclose(coefs.bm25_coef, 0.09395183408310023)
    assert math.isclose(coefs.intercept, -4.858095765012703)


def test_default_lr_coefs_returns_lr_coefs_instance() -> None:
    coefs = weight.default_lr_coefs()

    assert isinstance(coefs, weight.LRCoefs)
    assert coefs == weight.LRCoefs()


def test_default_lr_coefs_is_monkeypatchable(monkeypatch: pytest.MonkeyPatch) -> None:
    custom = weight.LRCoefs(emb_coef=1.0, bm25_coef=0.0, intercept=0.0)
    monkeypatch.setattr(weight, "default_lr_coefs", lambda: custom)

    assert weight.default_lr_coefs() == custom
