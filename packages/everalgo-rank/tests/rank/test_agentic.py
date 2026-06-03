"""Tests for aagentic_retrieve — agentic wrapper over caller-supplied base_retrieve."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from everalgo.rank import AgenticDecision, aagentic_retrieve
from everalgo.rank.agentic import MultiQueryResponse, RefinedQueryResponse, SufficiencyCheckResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate

if TYPE_CHECKING:
    from everalgo.llm.types import ChatResponse


def _ep_cand(cid: str, score: float = 1.0) -> Candidate:
    """Build a Candidate with a valid episode dict (required by _format_docs fail-loud)."""
    return Candidate(id=cid, score=score, metadata={"episode": {"subject": cid, "content": f"content of {cid}"}})


# ─── Smoke tests (3 new) ────────────────────────────────────────────────────


async def test_agentic_sufficient_round1_returns_base_no_round2() -> None:
    """When LLM judges Round 1 sufficient, return reranked[:top_n] without triggering Round 2."""
    base_calls: list[int] = []

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        base_calls.append(k)
        return [_ep_cand(f"d{i}", 1.0 / (i + 1)) for i in range(k)]

    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "is_sufficient": True,
                    "reasoning": "ok",
                    "key_information_found": [],
                    "missing_information": [],
                }
            ),
        ],
    )

    results, decision = await aagentic_retrieve(
        "q",
        base_retrieve=base_retrieve,
        rerank_fn=None,
        llm=fake,
        top_n=5,
        round1_top_n=10,
        round1_rerank_top_n=5,
    )
    assert len(results) == 5
    assert decision.is_sufficient is True
    assert decision.is_multi_round is False
    assert decision.refined_queries == []
    assert base_calls == [10]  # exactly one base call


async def test_agentic_insufficient_triggers_multi_query_round2() -> None:
    """Sufficiency=False + multi_query strategy generates queries and issues parallel base_retrieve calls."""
    base_calls: list[str] = []

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        base_calls.append(q)
        return [_ep_cand(f"{q}_d{i}", 1.0 / (i + 1)) for i in range(k)]

    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "is_sufficient": False,
                    "reasoning": "needs more",
                    "key_information_found": [],
                    "missing_information": ["date"],
                }
            ),
            json.dumps({"queries": ["alpha query", "beta query"], "reasoning": "two alts"}),
        ],
    )

    _results, decision = await aagentic_retrieve(
        "q",
        base_retrieve=base_retrieve,
        rerank_fn=None,
        llm=fake,
        top_n=5,
        refinement_strategy="multi_query",
        multi_query_count=2,
    )
    assert decision.is_multi_round is True
    assert decision.refined_queries == ["alpha query", "beta query"]
    assert decision.query_strategy == "multi_query"
    assert "q" in base_calls
    assert "alpha query" in base_calls
    assert "beta query" in base_calls


async def test_agentic_decision_returns_correct_type() -> None:
    """Return value is always tuple[list[Candidate], AgenticDecision]."""

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        return []

    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "is_sufficient": True,
                    "reasoning": "",
                    "key_information_found": [],
                    "missing_information": [],
                }
            ),
        ],
    )

    results, decision = await aagentic_retrieve(
        "q",
        base_retrieve=base_retrieve,
        rerank_fn=None,
        llm=fake,
        top_n=5,
    )
    assert isinstance(decision, AgenticDecision)
    assert results == []


# ─── Migrated from test_rank_fusion.py (aagentic_rank section) ──────────────


async def test_agentic_empty_base_returns_empty() -> None:
    """No initial candidates → return ([], AgenticDecision) without calling LLM."""
    fake = FakeLLMClient(responses=[])  # would error if popped

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        return []

    results, decision = await aagentic_retrieve(
        "q",
        base_retrieve=base_retrieve,
        rerank_fn=None,
        llm=fake,
        top_n=5,
    )

    assert results == []
    assert fake.call_count == 0
    assert decision.is_multi_round is False


async def test_agentic_sufficient_returns_reranked_truncated_to_top_n() -> None:
    """Sufficiency=True → return reranked[:top_n] (cross-encoder-ordered, mirrors scene_retrieval.py:262-267)."""

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        return [_ep_cand(f"d{i}", 1.0 / (i + 1)) for i in range(k)]

    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "is_sufficient": True,
                    "reasoning": "ok",
                    "key_information_found": [],
                    "missing_information": [],
                }
            ),
        ],
    )

    results, decision = await aagentic_retrieve(
        "q",
        base_retrieve=base_retrieve,
        rerank_fn=None,
        llm=fake,
        top_n=3,
        round1_top_n=10,
        round1_rerank_top_n=5,
    )

    assert len(results) == 3
    assert [c.id for c in results] == ["d0", "d1", "d2"]
    assert decision.is_sufficient is True
    assert decision.is_multi_round is False
    assert fake.call_count == 1  # only sufficiency check


async def test_agentic_rerank_fn_applied_before_sufficiency_check() -> None:
    """When rerank_fn is provided, the reranked slice is forwarded to the sufficiency LLM."""
    rerank_calls: list[str] = []

    async def rerank_fn(q: str, candidates: list[Candidate]) -> list[Candidate]:
        rerank_calls.append(q)
        return list(reversed(candidates))

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        return [_ep_cand(f"d{i}", 1.0 / (i + 1)) for i in range(k)]

    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "is_sufficient": True,
                    "reasoning": "ok",
                    "key_information_found": [],
                    "missing_information": [],
                }
            ),
        ],
    )

    _results, decision = await aagentic_retrieve(
        "q",
        base_retrieve=base_retrieve,
        rerank_fn=rerank_fn,
        llm=fake,
        top_n=3,
        round1_top_n=5,
        round1_rerank_top_n=3,
    )

    assert rerank_calls == ["q"]
    assert decision.is_sufficient is True


async def test_agentic_insufficient_parallel_round2_gather() -> None:
    """Insufficient + multi_query: Round 2 base_retrieve is called in parallel for each sub-query."""
    call_order: list[str] = []

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        call_order.append(q)
        return [_ep_cand(f"{q}_{i}", 0.5) for i in range(k)]

    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "is_sufficient": False,
                    "reasoning": "missing X",
                    "key_information_found": [],
                    "missing_information": ["X"],
                }
            ),
            json.dumps({"queries": ["q_alpha", "q_beta", "q_gamma"], "reasoning": "three angles"}),
        ],
    )

    _results, decision = await aagentic_retrieve(
        "original",
        base_retrieve=base_retrieve,
        rerank_fn=None,
        llm=fake,
        top_n=10,
        multi_query_count=3,
        refinement_strategy="multi_query",
    )

    assert decision.is_multi_round is True
    assert decision.refined_queries == ["q_alpha", "q_beta", "q_gamma"]
    assert fake.call_count == 2  # sufficiency + multi-query
    assert "original" in call_order  # Round 1
    assert "q_alpha" in call_order  # Round 2 parallel
    assert "q_beta" in call_order
    assert "q_gamma" in call_order


async def test_agentic_insufficient_refined_query_strategy() -> None:
    """refined_query strategy: single LLM call generates one refined query and single Round 2 retrieve."""
    base_calls: list[str] = []

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        base_calls.append(q)
        # Round 1 (original query "q") returns one candidate; Round 2 returns empty.
        if q == "q":
            return [_ep_cand("d1", 0.9)]
        return []

    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "is_sufficient": False,
                    "reasoning": "missing date",
                    "key_information_found": [],
                    "missing_information": ["date"],
                }
            ),
            # Refined-query prompt outputs plain text (evercore 93 format) — no JSON block.
            "Refined Query: When did Alice move to New York?",
        ],
    )

    _results, decision = await aagentic_retrieve(
        "q",
        base_retrieve=base_retrieve,
        rerank_fn=None,
        llm=fake,
        top_n=5,
        refinement_strategy="refined_query",
    )

    assert decision.is_multi_round is True
    assert decision.query_strategy == "refined_query"
    assert decision.refined_queries == ["When did Alice move to New York?"]
    assert "q" in base_calls  # Round 1
    assert "When did Alice move to New York?" in base_calls  # Round 2
    assert fake.call_count == 2


async def test_agentic_round2_dedup_by_id() -> None:
    """Round 2 candidates that share an id with Round 1 are excluded from the merge."""
    round1_cands = [_ep_cand("d1", 0.9)]
    round2_cands = [
        _ep_cand("d1", 0.5),  # duplicate — should be dropped
        _ep_cand("d2", 0.4),
    ]
    call_n = {"n": 0}

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        if call_n["n"] == 0:
            call_n["n"] += 1
            return round1_cands
        return round2_cands

    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "is_sufficient": False,
                    "reasoning": "missing",
                    "key_information_found": [],
                    "missing_information": ["X"],
                }
            ),
            json.dumps({"queries": ["extra query"], "reasoning": "one extra"}),
        ],
    )

    results, _decision = await aagentic_retrieve(
        "q",
        base_retrieve=base_retrieve,
        rerank_fn=None,
        llm=fake,
        top_n=10,
        refinement_strategy="multi_query",
        multi_query_count=1,
    )

    ids = [c.id for c in results]
    assert ids.count("d1") == 1  # dedup
    assert "d2" in ids


async def test_agentic_propagates_llm_error() -> None:
    """LLM raises → error propagates (no swallow)."""

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        return [_ep_cand("d1", 0.9)]

    def _boom(*_a: object, **_kw: object) -> ChatResponse:
        raise RuntimeError("llm down")

    fake = FakeLLMClient(handler=_boom)

    with pytest.raises(RuntimeError, match="llm down"):
        await aagentic_retrieve(
            "q",
            base_retrieve=base_retrieve,
            rerank_fn=None,
            llm=fake,
            top_n=5,
        )


async def test_agentic_round2_applies_final_rerank() -> None:
    """When rerank_fn is provided, Round 2 merged pool is rerank-passed before truncation."""
    base_calls: list[str] = []
    rerank_calls: list[tuple[str, int]] = []

    async def base_retrieve(q: str, k: int) -> list[Candidate]:
        base_calls.append(q)
        return [_ep_cand(f"{q}_d{i}", 1.0 / (i + 1)) for i in range(k)]

    async def rerank_fn(q: str, docs: list[Candidate]) -> list[Candidate]:
        rerank_calls.append((q, len(docs)))
        # Return reversed order to detect that rerank was applied.
        return list(reversed(docs))

    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "is_sufficient": False,
                    "reasoning": "needs more",
                    "key_information_found": [],
                    "missing_information": ["x"],
                }
            ),
            json.dumps({"queries": ["alternative query"], "reasoning": ""}),
        ],
    )

    _results, decision = await aagentic_retrieve(
        "q",
        base_retrieve=base_retrieve,
        rerank_fn=rerank_fn,
        llm=fake,
        top_n=5,
        refinement_strategy="multi_query",
        multi_query_count=1,
    )
    # rerank_fn should be called twice: Round 1 + final (Round 2 merged pool)
    assert len(rerank_calls) == 2, f"Expected 2 rerank calls, got {rerank_calls}"
    # Both calls use the original query
    assert all(call[0] == "q" for call in rerank_calls)
    assert decision.is_multi_round is True


async def test_agentic_schemas_are_importable() -> None:
    """SufficiencyCheckResponse / MultiQueryResponse / RefinedQueryResponse are stable public types."""
    sr = SufficiencyCheckResponse(is_sufficient=True, reasoning="ok", key_information_found=[], missing_information=[])
    mq = MultiQueryResponse(queries=["q1", "q2"], reasoning="r")
    rq = RefinedQueryResponse(refined_query="rq", reasoning="")
    assert sr.is_sufficient is True
    assert mq.queries == ["q1", "q2"]
    assert rq.refined_query == "rq"


# ─── _format_docs Date row tests ────────────────────────────────────────────


from datetime import UTC, datetime  # noqa: E402

from everalgo.rank.agentic import _format_docs  # noqa: E402


def test_format_docs_renders_date_row_iso_z() -> None:
    """Candidate with ms-epoch timestamp renders ``Date: YYYY-MM-DDTHH:MM:SSZ``."""
    ts_ms = int(datetime(2024, 3, 15, 13, 0, 0, tzinfo=UTC).timestamp() * 1000)
    c = Candidate(
        id="mc_1",
        score=0.9,
        metadata={
            "episode": {"subject": "fishing trip", "content": "Alice caught a fish"},
            "timestamp": ts_ms,
        },
    )
    rendered = _format_docs([c])
    assert "Date: 2024-03-15T13:00:00Z" in rendered
    assert "Title: fishing trip" in rendered
    assert "Content: Alice caught a fish" in rendered


def test_format_docs_missing_timestamp_renders_na() -> None:
    """Candidate without a timestamp key renders ``Date: N/A``."""
    c = Candidate(
        id="mc_1",
        score=0.9,
        metadata={"episode": {"subject": "x", "content": "y"}},
    )
    rendered = _format_docs([c])
    assert "Date: N/A" in rendered


def test_format_docs_truncates_long_body_to_500() -> None:
    """Content longer than 500 chars is truncated to 500 chars followed by ``...``."""
    body = "x" * 600
    c = Candidate(
        id="mc_1",
        score=0.9,
        metadata={"episode": {"subject": "long", "content": body}, "timestamp": 0},
    )
    rendered = _format_docs([c])
    assert "x" * 500 + "..." in rendered
    assert "x" * 501 not in rendered


# ─── _call_llm_for_refined_query plain-text parser tests ────────────────────


from everalgo.rank.agentic import _call_llm_for_refined_query, _format_doc_timestamp  # noqa: E402


async def test_call_llm_for_refined_query_strips_prefix() -> None:
    """FakeLLM returns plain text with ``Refined Query:`` prefix; parser strips it."""
    fake = FakeLLMClient(responses=["Refined Query: hello world more text"])
    result = await _call_llm_for_refined_query(fake, "rendered prompt", original_query="what is X?")
    assert result.refined_query == "hello world more text"
    assert result.reasoning == ""


async def test_call_llm_for_refined_query_falls_back_on_too_short() -> None:
    """LLM returns a string shorter than 5 chars; parser falls back to original_query."""
    fake = FakeLLMClient(responses=["hi"])
    original = "what happened yesterday?"
    result = await _call_llm_for_refined_query(fake, "rendered prompt", original_query=original)
    assert result.refined_query == original


async def test_call_llm_for_refined_query_falls_back_on_too_long() -> None:
    """LLM returns a string longer than 300 chars; parser falls back to original_query."""
    fake = FakeLLMClient(responses=["x" * 301])
    original = "what happened yesterday?"
    result = await _call_llm_for_refined_query(fake, "rendered prompt", original_query=original)
    assert result.refined_query == original


async def test_call_llm_for_refined_query_falls_back_when_identical() -> None:
    """LLM echoes the original query (case-mismatched); parser falls back to original_query."""
    original = "What did Alice do last Tuesday?"
    fake = FakeLLMClient(responses=[original.upper()])  # case-insensitive identity check
    result = await _call_llm_for_refined_query(fake, "rendered prompt", original_query=original)
    assert result.refined_query == original


# ─── _format_doc_timestamp edge-case tests ──────────────────────────────────


def test_format_doc_timestamp_returns_na_for_nan_inf_bool() -> None:
    """``NaN`` / ``inf`` / bool / None / string all render as ``N/A``."""
    assert _format_doc_timestamp(float("nan")) == "N/A"
    assert _format_doc_timestamp(float("inf")) == "N/A"
    assert _format_doc_timestamp(True) == "N/A"  # noqa: FBT003
    assert _format_doc_timestamp(False) == "N/A"  # noqa: FBT003
    assert _format_doc_timestamp(None) == "N/A"
    assert _format_doc_timestamp("not a number") == "N/A"  # type: ignore[arg-type]


# ─── round2_retrieve + round2_cap parameter tests ───────────────────────────


def _cand(cid: str, score: float = 1.0) -> Candidate:
    return Candidate(id=cid, score=score, metadata={"episode": {"subject": cid, "content": cid}})


async def test_round2_retrieve_separates_r1_and_r2_bases() -> None:
    """When sufficiency=False, R2 must call round2_retrieve, NOT base_retrieve."""
    r1_calls: list[str] = []
    r2_calls: list[str] = []

    async def base(q: str, k: int) -> list[Candidate]:
        r1_calls.append(q)
        return [_cand("r1_a"), _cand("r1_b")]

    async def round2(q: str, k: int) -> list[Candidate]:
        r2_calls.append(q)
        return [_cand("r2_a"), _cand("r2_b")]

    fake = FakeLLMClient(
        responses=[
            # Sufficiency: False -> triggers R2
            json.dumps(
                {
                    "is_sufficient": False,
                    "reasoning": "missing X",
                    "key_information_found": [],
                    "missing_information": ["X"],
                }
            ),
            # Multi-query generation: 2 sub-queries
            json.dumps({"queries": ["sub query one", "sub query two"], "reasoning": "expand"}),
        ],
    )

    _final, decision = await aagentic_retrieve(
        "q",
        base_retrieve=base,
        round2_retrieve=round2,
        rerank_fn=None,
        llm=fake,
        top_n=10,
        round1_top_n=10,
        round1_rerank_top_n=10,
        multi_query_count=2,
    )

    assert decision.is_multi_round is True
    assert r1_calls == ["q"]
    # R2 used round2 closure, not base
    assert r2_calls == ["sub query one", "sub query two"]


async def test_round2_retrieve_default_falls_back_to_base() -> None:
    """When round2_retrieve is None, R2 must use base_retrieve (hybrid-path behavior)."""
    calls: list[str] = []

    async def base(q: str, k: int) -> list[Candidate]:
        calls.append(q)
        return [_cand("a")]

    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "is_sufficient": False,
                    "reasoning": "x",
                    "key_information_found": [],
                    "missing_information": ["x"],
                }
            ),
            json.dumps({"queries": ["round two query"], "reasoning": "x"}),
        ],
    )

    await aagentic_retrieve(
        "q",
        base_retrieve=base,
        rerank_fn=None,
        llm=fake,
        top_n=5,
        round1_top_n=5,
        round1_rerank_top_n=5,
        multi_query_count=1,
    )
    # base used for both R1 and R2
    assert calls == ["q", "round two query"]


async def test_round2_cap_truncates_merged_to_cap() -> None:
    """``round2_cap=5`` with 3 R1 results means r2_unique trims to ``5 - 3 = 2`` items."""

    async def base(q: str, k: int) -> list[Candidate]:
        return [_cand(f"r1_{i}") for i in range(3)]

    async def round2(q: str, k: int) -> list[Candidate]:
        return [_cand(f"r2_{i}") for i in range(10)]

    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "is_sufficient": False,
                    "reasoning": "x",
                    "key_information_found": [],
                    "missing_information": ["x"],
                }
            ),
            json.dumps({"queries": ["round two query"], "reasoning": "x"}),
        ],
    )

    final, _ = await aagentic_retrieve(
        "q",
        base_retrieve=base,
        round2_retrieve=round2,
        round2_cap=5,
        rerank_fn=None,
        llm=fake,
        top_n=50,
        round1_top_n=10,
        round1_rerank_top_n=3,
        multi_query_count=1,
    )
    # 3 R1 + 2 R2-unique = 5 total (cap honored)
    assert len(final) == 5
    ids = [c.id for c in final]
    assert ids[:3] == ["r1_0", "r1_1", "r1_2"]
    assert ids[3:] == ["r2_0", "r2_1"]


async def test_round2_cap_none_keeps_current_behavior() -> None:
    """``round2_cap=None`` (default) means no truncation: all R1 + all r2_unique."""

    async def base(q: str, k: int) -> list[Candidate]:
        return [_cand(f"r1_{i}") for i in range(3)]

    async def round2(q: str, k: int) -> list[Candidate]:
        return [_cand(f"r2_{i}") for i in range(5)]

    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "is_sufficient": False,
                    "reasoning": "x",
                    "key_information_found": [],
                    "missing_information": ["x"],
                }
            ),
            json.dumps({"queries": ["round two query"], "reasoning": "x"}),
        ],
    )

    final, _ = await aagentic_retrieve(
        "q",
        base_retrieve=base,
        round2_retrieve=round2,
        rerank_fn=None,
        llm=fake,
        top_n=50,
        round1_top_n=10,
        round1_rerank_top_n=3,
        multi_query_count=1,
    )
    assert len(final) == 8  # 3 + 5
