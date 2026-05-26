"""Unit tests for ``everalgo.rank.fusion``."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from everalgo.rank import fusion, weight
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from everalgo.llm.types import ChatResponse

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


# ─── aagentic_rank ────────────────────────────────────────────────────────


def _make_retrieve(
    *, per_query: dict[str, list[Candidate]] | None = None
) -> tuple[Callable[[str, int], Awaitable[list[Candidate]]], list[tuple[str, int]]]:
    """Build a Round-2 retrieve callback that records ``(query, top_n)`` and returns per-query lists."""
    calls: list[tuple[str, int]] = []
    table = per_query or {}

    async def _retrieve(query: str, top_n: int) -> list[Candidate]:
        calls.append((query, top_n))
        return table.get(query, [])[:top_n]

    return _retrieve, calls


def _make_rerank(
    *,
    orderings: list[list[Candidate]],
) -> tuple[Callable[[str, Sequence[Candidate], int], Awaitable[list[Candidate]]], list[int]]:
    """Build a rerank callback that returns ``orderings[i]`` truncated to ``top_n`` on call ``i``."""
    state = {"i": 0}
    truncations: list[int] = []

    async def _rerank(_q: str, _cands: Sequence[Candidate], top_n: int) -> list[Candidate]:
        ordering = orderings[state["i"]]
        state["i"] += 1
        truncations.append(top_n)
        return list(ordering[:top_n])

    return _rerank, truncations


async def test_aagentic_rank_round1_concat_dedup_no_rrf(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    """Round 1 is concat(sparse, dense) + dedup by id."""
    captured_round1: list[Candidate] = []

    async def _rerank(_q: str, cands: Sequence[Candidate], top_n: int) -> list[Candidate]:
        captured_round1.extend(cands)
        return list(cands)[:top_n]

    fake = FakeLLMClient(responses=[json.dumps({"is_sufficient": True, "reasoning": "ok", "missing_information": []})])

    await fusion.aagentic_rank(
        "q",
        sparse_candidates,
        dense_candidates,
        rerank=_rerank,
        top_k=3,
        llm=fake,
        config=fusion.AgenticConfig(round1_rerank_top_n=3),
    )

    # sparse has {d1, d6, d3, d7}; dense has {d1, d2, d3, d4, d5}; union by id should be 7 unique.
    ids = [c.id for c in captured_round1]
    assert ids == ["d1", "d6", "d3", "d7", "d2", "d4", "d5"]  # sparse first, dense appended unique only


async def test_aagentic_rank_sufficient_skips_round2(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    """Sufficiency=True → no Round 2; result is Round-1 rerank truncated to top_k."""
    retrieve, retrieve_calls = _make_retrieve()  # would record any unwanted calls
    rerank, truncations = _make_rerank(orderings=[dense_candidates])

    fake = FakeLLMClient(responses=[json.dumps({"is_sufficient": True, "reasoning": "ok", "missing_information": []})])

    out = await fusion.aagentic_rank(
        "q",
        sparse_candidates,
        dense_candidates,
        rerank=rerank,
        retrieve=retrieve,
        top_k=2,
        llm=fake,
        config=fusion.AgenticConfig(round1_rerank_top_n=3),
    )

    assert [c.id for c in out] == ["d1", "d2"]
    assert retrieve_calls == []  # Round 2 skipped
    assert fake.call_count == 1  # only sufficiency check
    assert truncations == [3]  # only Round 1 rerank


async def test_aagentic_rank_insufficient_triggers_round2(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    """Sufficiency=False + retrieve_fn → multi-query → parallel retrieve → dedup-merge → final rerank."""
    round2_extra = [
        Candidate(id="d8", score=0.9, source="vector"),
        Candidate(id="d9", score=0.8, source="vector"),
    ]
    retrieve, retrieve_calls = _make_retrieve(
        per_query={"follow-up alpha": round2_extra, "follow-up beta": round2_extra},
    )
    final_order = [
        Candidate(id="d8", score=0.99, source="vector"),
        Candidate(id="d1", score=0.95, source="vector"),
        Candidate(id="d9", score=0.85, source="vector"),
    ]
    rerank, truncations = _make_rerank(orderings=[dense_candidates, final_order])

    fake = FakeLLMClient(
        responses=[
            json.dumps({"is_sufficient": False, "reasoning": "missing X", "missing_information": ["X"]}),
            json.dumps({"queries": ["follow-up alpha", "follow-up beta"], "reasoning": "diff angles"}),
        ]
    )

    out = await fusion.aagentic_rank(
        "q",
        sparse_candidates,
        dense_candidates,
        rerank=rerank,
        retrieve=retrieve,
        top_k=2,
        llm=fake,
        config=fusion.AgenticConfig(
            round1_rerank_top_n=3,
            round2_per_query_top_n=2,
            combined_total=10,
            num_queries=2,
        ),
    )

    assert [c.id for c in out] == ["d8", "d1"]
    assert len(retrieve_calls) == 2  # 2 follow-up queries in parallel
    assert {call[0] for call in retrieve_calls} == {"follow-up alpha", "follow-up beta"}
    assert fake.call_count == 2  # sufficiency + multi-query
    # Round 1 rerank top_n=max(3,2)=3; final rerank top_n=max(combined_total=10, top_k=2)=10
    assert truncations == [3, 10]


async def test_aagentic_rank_insufficient_without_retrieve_fn_returns_round1(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    """Sufficiency=False but no retrieve_fn → return Round-1 rerank truncated to top_k (no Round 2)."""
    rerank, truncations = _make_rerank(orderings=[dense_candidates])

    fake = FakeLLMClient(
        responses=[json.dumps({"is_sufficient": False, "reasoning": "x", "missing_information": ["x"]})]
    )

    out = await fusion.aagentic_rank(
        "q",
        sparse_candidates,
        dense_candidates,
        rerank=rerank,
        retrieve=None,
        top_k=2,
        llm=fake,
        config=fusion.AgenticConfig(round1_rerank_top_n=3),
    )

    assert [c.id for c in out] == ["d1", "d2"]
    assert fake.call_count == 1  # only sufficiency
    assert truncations == [3]


async def test_aagentic_rank_empty_sparse_and_dense_returns_empty() -> None:
    """No initial candidates → return [] without calling rerank or LLM."""
    rerank, truncations = _make_rerank(orderings=[])
    fake = FakeLLMClient(responses=[])  # would error if popped

    out = await fusion.aagentic_rank("q", [], [], rerank=rerank, top_k=5, llm=fake)

    assert out == []
    assert truncations == []
    assert fake.call_count == 0


async def test_aagentic_rank_unlimited_top_k(
    dense_candidates: list[Candidate],
) -> None:
    """``top_k=-1`` → rerank top_n stays at round1_rerank_top_n, no truncation at end."""
    rerank, truncations = _make_rerank(orderings=[dense_candidates])
    fake = FakeLLMClient(responses=[json.dumps({"is_sufficient": True, "reasoning": "ok", "missing_information": []})])

    out = await fusion.aagentic_rank(
        "q",
        [],
        dense_candidates,
        rerank=rerank,
        top_k=-1,
        llm=fake,
        config=fusion.AgenticConfig(round1_rerank_top_n=3),
    )

    assert len(out) == 3
    assert truncations == [3]


async def test_aagentic_rank_propagates_rerank_errors(
    dense_candidates: list[Candidate],
) -> None:
    """Exceptions from the rerank callback propagate — no swallow."""

    async def _broken_rerank(_q: str, _c: Sequence[Candidate], _n: int) -> list[Candidate]:
        raise RuntimeError("boom")

    fake = FakeLLMClient(responses=[])

    with pytest.raises(RuntimeError, match="boom"):
        await fusion.aagentic_rank("q", [], dense_candidates, rerank=_broken_rerank, top_k=5, llm=fake)


# ─── Private helpers — parse + format + LLM-error fallback ────────────────


def test_format_candidates_for_llm_empty_input_returns_placeholder() -> None:
    assert fusion._format_candidates_for_llm([], max_docs=5) == "No retrieval results"


def test_format_candidates_for_llm_renders_known_fields() -> None:
    """Pulls ``episode`` / ``summary`` / ``subject`` from metadata, in that order."""
    cands = [
        Candidate(id="a", score=0.9, metadata={"episode": "had pizza for lunch", "timestamp": "2026-05-12"}),
        Candidate(id="b", score=0.8, metadata={"summary": "lunch chat"}),
    ]
    out = fusion._format_candidates_for_llm(cands, max_docs=5)
    assert "[Memory 1]" in out
    assert "had pizza for lunch" in out
    assert "2026-05-12" in out
    assert "[Memory 2]" in out
    assert "lunch chat" in out


async def test_acheck_sufficiency_propagates_llm_error() -> None:
    """LLM raises → error propagates (no swallow)."""
    from everalgo.rank.prompts.en.agentic import AGENTIC_SUFFICIENCY_CHECK_PROMPT_EN

    def _boom(*_a: object, **_kw: object) -> ChatResponse:
        raise RuntimeError("llm down")

    fake = FakeLLMClient(handler=_boom)
    with pytest.raises(RuntimeError, match="llm down"):
        await fusion._acheck_sufficiency(
            query="q",
            candidates=[],
            llm=fake,
            prompt=AGENTIC_SUFFICIENCY_CHECK_PROMPT_EN,
            max_docs=5,
            max_tokens=100,
            temperature=0.0,
        )


async def test_agen_multi_queries_propagates_llm_error() -> None:
    """LLM raises → error propagates (no fallback to original query)."""
    from everalgo.rank.prompts.en.agentic import AGENTIC_MULTI_QUERY_PROMPT_EN

    def _boom(*_a: object, **_kw: object) -> ChatResponse:
        raise RuntimeError("llm down")

    fake = FakeLLMClient(handler=_boom)
    with pytest.raises(RuntimeError, match="llm down"):
        await fusion._agen_multi_queries(
            original_query="orig query text",
            candidates=[],
            missing_info=["X"],
            llm=fake,
            prompt=AGENTIC_MULTI_QUERY_PROMPT_EN,
            max_docs=5,
            num_queries=3,
            max_tokens=100,
            temperature=0.4,
        )
