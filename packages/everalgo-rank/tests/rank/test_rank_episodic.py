"""Unit tests for ``everalgo.rank.episodic``."""

from __future__ import annotations

import json

from everalgo.rank import FusionMode, RankConfig, episodic
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate, FactCandidate, RankInput


def _make_input(
    *,
    sparse: list[Candidate] | None = None,
    dense: list[Candidate] | None = None,
    facts: dict[str, list[FactCandidate]] | None = None,
    top_k: int = 5,
) -> RankInput:
    return RankInput(
        query="when did we discuss the project plan",
        memory_type="episodic",
        sparse_candidates=sparse or [],
        dense_candidates=dense or [],
        episode_to_facts=facts or {},
        top_k=top_k,
    )


async def test_arank_mrag_mode_runs_hierarchical_expansion(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
    episode_to_facts: dict[str, list[FactCandidate]],
) -> None:
    """``fusion_mode='mrag'`` triggers Phase 1 + Phase 2-4 hierarchical expand."""
    out = await episodic.arank(
        _make_input(sparse=sparse_candidates, dense=dense_candidates, facts=episode_to_facts, top_k=4),
        config=RankConfig(fusion_mode="mrag", max_convergence_rounds=3, expand_limit=1),
    )

    assert len(out.items) <= 4
    assert all(item.score >= 0 for item in out.items)
    # mix of episode + atomic_fact item types (some facts entered top-N)
    types = {it.item_type for it in out.items}
    assert "atomic_fact" in types or "episode" in types
    assert out.metadata.get("fusion_mode") == "mrag"


async def test_arank_rrf_mode_skips_expansion(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
    episode_to_facts: dict[str, list[FactCandidate]],
) -> None:
    """``fusion_mode='rrf'`` does NOT expand, even when episode_to_facts is provided.

    This is the parity test for ``fusion_mode`` being parallel to ``lr`` / ``mrag``:
    expand is opt-in via ``"mrag"``, not implicit in ``"rrf"``.
    """
    out = await episodic.arank(
        _make_input(sparse=sparse_candidates, dense=dense_candidates, facts=episode_to_facts, top_k=4),
        config=RankConfig(fusion_mode="rrf"),
    )

    # All results stay as episodes; no atomic_fact climbs into top-N.
    assert all(it.item_type == "episode" for it in out.items)
    assert out.metadata.get("fusion_mode") == "rrf"


async def test_arank_no_facts_degrades_gracefully(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    """No ep→fact linkage + mrag → expand has nothing to expand, all episodes."""
    out = await episodic.arank(
        _make_input(sparse=sparse_candidates, dense=dense_candidates, top_k=3),
        config=RankConfig(fusion_mode="mrag"),
    )

    assert all(it.item_type == "episode" for it in out.items)
    assert len(out.items) <= 3


async def test_arank_rerank_disabled_does_not_call_llm(
    dense_candidates: list[Candidate],
) -> None:
    fake = FakeLLMClient(responses=[])  # would error if called

    out = await episodic.arank(
        _make_input(dense=dense_candidates),
        config=RankConfig(fusion_mode="rrf"),
        llm=fake,
        enable_rerank=False,
    )

    assert fake.call_count == 0
    assert out.metadata.get("reranked") is not True


async def test_arank_rerank_enabled_replaces_scores(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    """LLM reranks the top-N fusion result and reorders by LLM score."""
    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "ranked": [
                        {"id": "d3", "score": 0.99},
                        {"id": "d1", "score": 0.50},
                    ]
                }
            )
        ]
    )

    out = await episodic.arank(
        _make_input(sparse=sparse_candidates, dense=dense_candidates, top_k=2),
        config=RankConfig(fusion_mode="rrf"),
        llm=fake,
        enable_rerank=True,
    )

    assert fake.call_count == 1
    assert out.metadata.get("reranked") is True
    # fusion top-2 is {d1, d3}; LLM rerank pushes d3 above d1
    assert [it.id for it in out.items] == ["d3", "d1"]


async def test_arank_empty_input_returns_empty_output() -> None:
    out = await episodic.arank(_make_input())
    assert out.items == []
    assert out.metadata.get("stop_reason") == "no_candidates"


def test_sync_bridge_is_callable(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    """``rank()`` (sync) usable from pytest test body."""
    out = episodic.rank(
        _make_input(sparse=sparse_candidates, dense=dense_candidates, top_k=2),
        config=RankConfig(fusion_mode="rrf"),
    )
    assert len(out.items) <= 2


async def test_three_fusion_modes_all_runnable(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    modes: list[FusionMode] = ["rrf", "lr", "mrag"]
    for mode in modes:
        out = await episodic.arank(
            _make_input(sparse=sparse_candidates, dense=dense_candidates, top_k=2),
            config=RankConfig(fusion_mode=mode),
        )
        assert out.items, f"fusion_mode={mode} produced empty output"


# ── Instance-level llm= binding (EpisodicRanker class) ──────────────────────────────────────────────


async def test_episodic_ranker_uses_instance_llm_when_per_call_omitted(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    """Instance-level llm= is used when EpisodicRanker.arank() is called without per-call llm=."""
    from everalgo.rank.episodic import EpisodicRanker

    rerank_payload = json.dumps({"ranked": [{"id": "d1", "score": 0.9}]})
    instance_fake = FakeLLMClient(responses=[rerank_payload])
    ranker = EpisodicRanker(llm=instance_fake)
    out = await ranker.arank(
        _make_input(sparse=sparse_candidates, dense=dense_candidates, top_k=1),
        config=RankConfig(fusion_mode="rrf"),
        enable_rerank=True,
    )
    assert instance_fake.call_count == 1
    assert len(out.items) <= 1
