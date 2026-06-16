"""Tests for ``rank/category.py`` — primitives and ``acategory_retrieve`` facade."""

from __future__ import annotations

from everalgo.rank import (
    acategory_retrieve,
    apply_category_boost,
    category_retrieve,
    rollup_category_mass,
)
from everalgo.types import Candidate

# ── rollup_category_mass ─────────────────────────────────────────────


def _cand(id_: str, score: float, cat: str | None = None) -> Candidate:
    meta: dict[str, object] = {}
    if cat is not None:
        meta["category_id"] = cat
    return Candidate(id=id_, score=score, source="vector", metadata=meta)


def test_rollup_empty_returns_empty_and_zero_conf() -> None:
    p, conf = rollup_category_mass([])
    assert p == {}
    assert conf == 0.0


def test_rollup_unclassified_hits_do_not_form_a_bucket() -> None:
    hits = [_cand("a", 1.0), _cand("b", 0.5)]  # neither has category
    p, conf = rollup_category_mass(hits)
    assert p == {}
    assert conf == 0.0


def test_rollup_single_category_yields_conf_one() -> None:
    hits = [_cand("a", 1.0, "how-to"), _cand("b", 0.5, "how-to")]
    p, conf = rollup_category_mass(hits)
    assert p == {"how-to": 1.0}
    assert conf == 1.0


def test_rollup_is_score_weighted_not_count_weighted() -> None:
    # Three weak "news" hits vs one strong "how-to" hit.
    hits = [
        _cand("h", 10.0, "how-to"),
        _cand("n1", 0.5, "news"),
        _cand("n2", 0.5, "news"),
        _cand("n3", 0.5, "news"),
    ]
    p, conf = rollup_category_mass(hits)
    total = 10.0 + 1.5
    assert p["how-to"] == 10.0 / total
    assert p["news"] == 1.5 / total
    assert conf == p["how-to"] - p["news"]


def test_rollup_respects_top_m_window() -> None:
    # The top-2 are "how-to" with score 1; if top_m=2, "news" is excluded entirely.
    hits = [
        _cand("h1", 1.0, "how-to"),
        _cand("h2", 1.0, "how-to"),
        _cand("n1", 1.0, "news"),
        _cand("n2", 1.0, "news"),
    ]
    p, conf = rollup_category_mass(hits, top_m=2)
    assert p == {"how-to": 1.0}
    assert conf == 1.0


def test_rollup_tied_top_two_collapses_confidence() -> None:
    hits = [
        _cand("a1", 1.0, "how-to"),
        _cand("b1", 1.0, "news"),
    ]
    p, conf = rollup_category_mass(hits)
    assert p == {"how-to": 0.5, "news": 0.5}
    assert conf == 0.0


def test_rollup_custom_category_key() -> None:
    c = Candidate(id="x", score=1.0, source="vector", metadata={"domain": "law"})
    p, conf = rollup_category_mass([c], category_key="domain")
    assert p == {"law": 1.0}
    assert conf == 1.0


# ── apply_category_boost ─────────────────────────────────────────────


def test_apply_boost_zero_conf_preserves_relevance_ordering() -> None:
    reranked = [
        _cand("a", 0.9, "how-to"),
        _cand("b", 0.5, "news"),
        _cand("c", 0.1, "how-to"),
    ]
    p = {"how-to": 1.0}
    out = apply_category_boost(reranked, p, conf=0.0)
    assert [c.id for c in out] == ["a", "b", "c"]


def test_apply_boost_high_conf_tiebreaks_via_category() -> None:
    # Two candidates with identical rel — boost should put the preferred category first.
    reranked = [
        _cand("h", 0.5, "how-to"),
        _cand("n", 0.5, "news"),
    ]
    p = {"how-to": 0.9, "news": 0.1}
    out = apply_category_boost(reranked, p, conf=1.0, lam=0.5)
    assert [c.id for c in out] == ["h", "n"]
    # final = 0.5 (flat-rel fallback) + 0.5 * 1 * 0.9 = 0.95
    assert out[0].score > out[1].score


def test_apply_boost_does_not_mutate_input() -> None:
    reranked = [_cand("a", 0.9, "how-to"), _cand("b", 0.1, "news")]
    original_scores = [c.score for c in reranked]
    apply_category_boost(reranked, {"how-to": 1.0}, conf=1.0, lam=0.2)
    assert [c.score for c in reranked] == original_scores


def test_apply_boost_empty_input_returns_empty() -> None:
    assert apply_category_boost([], {}, 0.5) == []


def test_apply_boost_clamps_negative_conf_to_zero() -> None:
    reranked = [_cand("a", 0.9, "how-to"), _cand("b", 0.5, "news")]
    out = apply_category_boost(reranked, {"how-to": 1.0}, conf=-0.5, lam=10.0)
    # Negative conf → effective lam = 0 → ranking driven solely by rel.
    assert [c.id for c in out] == ["a", "b"]


def test_apply_boost_unknown_category_gets_zero_mass() -> None:
    reranked = [_cand("a", 0.5, "unknown"), _cand("b", 0.5, "how-to")]
    p = {"how-to": 1.0}
    out = apply_category_boost(reranked, p, conf=1.0, lam=0.5)
    # b gets the boost; a does not.
    assert out[0].id == "b"


# ── acategory_retrieve (facade) ──────────────────────────────────────


async def test_acategory_retrieve_end_to_end_with_score_capture() -> None:
    """Verify the timing invariant: rollup sees recall scores, boost sees rerank scores."""
    captured: dict[str, list[float]] = {"recall_scores_at_rollup": [], "rerank_input_scores": []}

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        return [
            _cand("a", 10.0, "how-to"),
            _cand("b", 8.0, "how-to"),
            _cand("c", 6.0, "news"),
            _cand("d", 4.0, "news"),
        ]

    async def rerank_fn(q: str, cands: list[Candidate]) -> list[Candidate]:
        captured["rerank_input_scores"] = [c.score for c in cands]
        # Reverse the order via the relevance score — d/c best, a/b worst.
        return [
            Candidate(id=c.id, score=float(i), source=c.source, metadata=dict(c.metadata)) for i, c in enumerate(cands)
        ]

    results = await acategory_retrieve(
        query="anything",
        base_retrieve=base_retrieve,
        rerank_fn=rerank_fn,
        recall_n=4,
        rerank_n=4,
        lam=0.0,  # disable boost so the test just checks plumbing + ordering
        top_n=4,
    )

    # When lam=0, ordering is whatever min-max(rel) gives, which preserves rerank descending.
    assert [c.id for c in results] == ["d", "c", "b", "a"]
    # rerank_fn saw the recall scores (10, 8, 6, 4), not anything mutated by us mid-flight.
    assert captured["rerank_input_scores"] == [10.0, 8.0, 6.0, 4.0]


async def test_acategory_retrieve_high_conf_query_boosts_preferred_category() -> None:
    # Five "how-to" hits dominate recall — rollup should hand a strong how-to mass.
    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        return [
            _cand("h1", 10.0, "how-to"),
            _cand("h2", 9.0, "how-to"),
            _cand("h3", 8.0, "how-to"),
            _cand("h4", 7.0, "how-to"),
            _cand("n1", 6.0, "news"),
        ]

    async def rerank_fn(q: str, cands: list[Candidate]) -> list[Candidate]:
        # Tie: every candidate gets the same relevance score — boost is the only differentiator.
        return [Candidate(id=c.id, score=0.5, source=c.source, metadata=dict(c.metadata)) for c in cands]

    results = await acategory_retrieve(
        query="anything",
        base_retrieve=base_retrieve,
        rerank_fn=rerank_fn,
        recall_n=5,
        rerank_n=5,
        lam=0.5,
        top_n=5,
    )

    # All how-to candidates outrank the news candidate.
    assert results[-1].id == "n1"
    assert {c.id for c in results[:4]} == {"h1", "h2", "h3", "h4"}


async def test_acategory_retrieve_empty_recall_returns_empty() -> None:
    async def empty(q: str, k: int) -> list[Candidate]:
        return []

    async def never(q: str, cands: list[Candidate]) -> list[Candidate]:
        raise AssertionError("rerank_fn must not be called when recall is empty")

    out = await acategory_retrieve(
        query="q",
        base_retrieve=empty,
        rerank_fn=never,
    )
    assert out == []


async def test_acategory_retrieve_respects_rerank_n_window() -> None:
    """rerank_n smaller than recall_n means only the top of the recall pool is reranked."""
    rerank_pool_sizes: list[int] = []

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        return [_cand(f"c{i}", float(10 - i), "how-to") for i in range(10)]

    async def rerank_fn(q: str, cands: list[Candidate]) -> list[Candidate]:
        rerank_pool_sizes.append(len(cands))
        return cands

    await acategory_retrieve(
        query="q",
        base_retrieve=base_retrieve,
        rerank_fn=rerank_fn,
        recall_n=10,
        rerank_n=3,
        top_n=3,
    )
    assert rerank_pool_sizes == [3]


def test_category_retrieve_sync_bridge_works() -> None:
    """Sync bridge round-trips the same as the async facade in a non-event-loop context."""

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        return [_cand("a", 1.0, "how-to"), _cand("b", 0.5, "news")]

    async def rerank_fn(q: str, cands: list[Candidate]) -> list[Candidate]:
        return [Candidate(id=c.id, score=c.score, source=c.source, metadata=dict(c.metadata)) for c in cands]

    out = category_retrieve(
        query="q",
        base_retrieve=base_retrieve,
        rerank_fn=rerank_fn,
        recall_n=2,
        rerank_n=2,
        lam=0.0,
        top_n=2,
    )
    assert [c.id for c in out] == ["a", "b"]
