"""Unit tests for ``everalgo.rank.skill``."""

from __future__ import annotations

import json

from everalgo.rank import RankConfig, skill
from everalgo.rank.skill import _apply_relevance_gate
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate, RankInput, RankOutput, ScoredItem


def _mk(sparse: list[Candidate], dense: list[Candidate], top_k: int = 5) -> RankInput:
    return RankInput(
        query="implement structured logging",
        memory_type="skill",
        sparse_candidates=sparse,
        dense_candidates=dense,
        top_k=top_k,
    )


async def test_arank_returns_skill_items_sorted_by_fusion(
    skill_candidates: list[Candidate],
) -> None:
    """No business-field weighting — order driven entirely by fusion."""
    out = await skill.arank(
        _mk(sparse=[], dense=skill_candidates, top_k=3),
        config=RankConfig(fusion_mode="rrf"),
    )

    assert all(it.item_type == "skill" for it in out.items)
    scores = [it.score for it in out.items]
    assert scores == sorted(scores, reverse=True)


async def test_arank_passes_business_fields_through_metadata(
    skill_candidates: list[Candidate],
) -> None:
    """maturity_score / confidence stay in metadata for downstream use."""
    out = await skill.arank(_mk(sparse=[], dense=skill_candidates, top_k=3))

    for item in out.items:
        # Each skill_candidates entry has both business fields in metadata.
        assert "maturity_score" in item.metadata
        assert "confidence" in item.metadata


async def test_arank_empty_input_short_circuits() -> None:
    out = await skill.arank(_mk([], []))
    assert out.items == []
    assert out.metadata.get("stop_reason") == "no_candidates"


def test_sync_bridge_is_callable(skill_candidates: list[Candidate]) -> None:
    out = skill.rank(_mk(sparse=[], dense=skill_candidates, top_k=2))
    assert len(out.items) <= 2


# ─── enable_rerank (single LLM pass: reorder + quality-grade) ────────────────


def _rerank_response(scores: dict[str, float]) -> str:
    """Build a JSON-encoded rerank payload (``{"ranked": [{id, score}]}``) for FakeLLMClient."""
    return json.dumps({"ranked": [{"id": item_id, "score": score} for item_id, score in scores.items()]})


async def test_arank_enable_rerank_applies_llm_scores(
    skill_candidates: list[Candidate],
) -> None:
    """``enable_rerank=True`` runs one LLM pass that reorders/grades via SKILL_RERANK_PROMPT."""
    ids = [c.id for c in skill_candidates]
    # Invert the natural order so we can prove the LLM scores drive the output ordering.
    fake = FakeLLMClient(responses=[_rerank_response({ids[0]: 0.2, ids[1]: 0.9, ids[2]: 0.5})])

    out = await skill.arank(
        _mk(sparse=[], dense=skill_candidates, top_k=3),
        config=RankConfig(fusion_mode="rrf"),
        llm=fake,
        enable_rerank=True,
        min_rerank_score=0.0,  # isolate rerank ordering from the relevance gate
    )

    assert fake.call_count == 1
    assert out.metadata.get("reranked") is True
    assert [it.id for it in out.items] == [ids[1], ids[2], ids[0]]


async def test_arank_enable_rerank_without_llm_raises(
    skill_candidates: list[Candidate],
) -> None:
    """Module-level ``arank`` needs an LLM client when ``enable_rerank=True``."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="enable_rerank=True requires llm"):
        await skill.arank(
            _mk(sparse=[], dense=skill_candidates, top_k=3),
            config=RankConfig(fusion_mode="rrf"),
            enable_rerank=True,
        )


async def test_arank_disable_rerank_skips_llm(
    skill_candidates: list[Candidate],
) -> None:
    """``enable_rerank=False`` (default) leaves output untouched by the LLM."""
    fake = FakeLLMClient(responses=[])  # would error if invoked

    out = await skill.arank(
        _mk(sparse=[], dense=skill_candidates, top_k=3),
        config=RankConfig(fusion_mode="rrf"),
        llm=fake,
    )

    assert fake.call_count == 0
    assert "reranked" not in out.metadata


# ─── min_rerank_score (skill-only post-rerank relevance gate) ────────────────


async def test_arank_relevance_gate_drops_below_threshold(
    skill_candidates: list[Candidate],
) -> None:
    """After rerank, items scored below ``min_rerank_score`` (default 0.4) are dropped."""
    ids = [c.id for c in skill_candidates]
    fake = FakeLLMClient(responses=[_rerank_response({ids[0]: 0.9, ids[1]: 0.3, ids[2]: 0.5})])

    out = await skill.arank(
        _mk(sparse=[], dense=skill_candidates, top_k=3),
        config=RankConfig(fusion_mode="rrf"),
        llm=fake,
        enable_rerank=True,
    )

    # ids[1] (0.3 < 0.4) dropped; survivors sorted desc by LLM score.
    assert [it.id for it in out.items] == [ids[0], ids[2]]
    assert out.metadata["rerank_min_score"] == 0.4
    assert out.metadata["rerank_dropped"] == 1


async def test_arank_relevance_gate_disabled_with_zero_threshold(
    skill_candidates: list[Candidate],
) -> None:
    """``min_rerank_score=0.0`` keeps every reranked item."""
    ids = [c.id for c in skill_candidates]
    fake = FakeLLMClient(responses=[_rerank_response({ids[0]: 0.9, ids[1]: 0.1, ids[2]: 0.05})])

    out = await skill.arank(
        _mk(sparse=[], dense=skill_candidates, top_k=3),
        config=RankConfig(fusion_mode="rrf"),
        llm=fake,
        enable_rerank=True,
        min_rerank_score=0.0,
    )

    assert len(out.items) == 3
    assert "rerank_dropped" not in out.metadata


async def test_arank_relevance_gate_inactive_without_rerank(
    skill_candidates: list[Candidate],
) -> None:
    """Gate is a no-op when rerank did not run — fusion scores are not on a 0-1 scale."""
    out = await skill.arank(
        _mk(sparse=[], dense=skill_candidates, top_k=3),
        config=RankConfig(fusion_mode="rrf"),
    )

    # All fused candidates survive even though RRF scores are far below 0.4.
    assert len(out.items) == 3
    assert "rerank_dropped" not in out.metadata


# ─── SkillRanker (class facade) ─────────────────────────────────────────────


async def test_skill_ranker_class_rerank_and_gate(
    skill_candidates: list[Candidate],
) -> None:
    """``SkillRanker.arank`` runs rerank with the bound LLM and applies the relevance gate."""
    ids = [c.id for c in skill_candidates]
    fake = FakeLLMClient(responses=[_rerank_response({ids[0]: 0.9, ids[1]: 0.2, ids[2]: 0.6})])
    ranker = skill.SkillRanker(llm=fake)

    out = await ranker.arank(
        _mk(sparse=[], dense=skill_candidates, top_k=3),
        config=RankConfig(fusion_mode="rrf"),
        enable_rerank=True,
    )

    assert fake.call_count == 1
    # ids[1] (0.2 < 0.4) dropped by the gate; survivors sorted desc.
    assert [it.id for it in out.items] == [ids[0], ids[2]]
    assert out.metadata["rerank_dropped"] == 1


def test_skill_ranker_class_sync_bridge(skill_candidates: list[Candidate]) -> None:
    ranker = skill.SkillRanker(llm=FakeLLMClient(responses=[]))
    out = ranker.rank(_mk(sparse=[], dense=skill_candidates, top_k=2))
    assert len(out.items) <= 2


# ─── _apply_relevance_gate (unit) ───────────────────────────────────────────


def _scored(*items: tuple[str, float], reranked: bool = True) -> RankOutput:
    """Build a skill RankOutput from ``(id, score)`` pairs, defaulting to a reranked result."""
    meta = {"stage": "skill", "reranked": True} if reranked else {"stage": "skill"}
    return RankOutput(
        items=[ScoredItem(id=item_id, score=score, item_type="skill") for item_id, score in items],
        metadata=meta,
    )


def test_gate_drops_items_below_threshold() -> None:
    result = _scored(("s1", 0.9), ("s2", 0.3), ("s3", 0.5))

    gated = _apply_relevance_gate(result, 0.4)

    assert [it.id for it in gated.items] == ["s1", "s3"]
    assert gated.metadata["rerank_min_score"] == 0.4
    assert gated.metadata["rerank_dropped"] == 1


def test_gate_keeps_item_exactly_at_threshold() -> None:
    """Threshold is inclusive (``score >= min``)."""
    result = _scored(("s1", 0.4), ("s2", 0.39))

    gated = _apply_relevance_gate(result, 0.4)

    assert [it.id for it in gated.items] == ["s1"]


def test_gate_noop_when_not_reranked() -> None:
    """Fusion-only output is left untouched — its scores are not on a 0-1 scale."""
    result = _scored(("s1", 0.016), ("s2", 0.008), reranked=False)

    gated = _apply_relevance_gate(result, 0.4)

    assert gated is result
    assert [it.id for it in gated.items] == ["s1", "s2"]
    assert "rerank_dropped" not in gated.metadata


def test_gate_disabled_with_zero_threshold() -> None:
    result = _scored(("s1", 0.9), ("s2", 0.1))

    gated = _apply_relevance_gate(result, 0.0)

    assert gated is result
    assert "rerank_dropped" not in gated.metadata


def test_gate_negative_threshold_is_disabled() -> None:
    result = _scored(("s1", 0.9), ("s2", 0.1))

    gated = _apply_relevance_gate(result, -1.0)

    assert gated is result


def test_gate_preserves_existing_metadata() -> None:
    result = _scored(("s1", 0.9))
    result = result.model_copy(update={"metadata": {**result.metadata, "rerank_top_k": 5}})

    gated = _apply_relevance_gate(result, 0.4)

    assert gated.metadata["stage"] == "skill"
    assert gated.metadata["reranked"] is True
    assert gated.metadata["rerank_top_k"] == 5


def test_gate_empty_items() -> None:
    result = RankOutput(items=[], metadata={"stage": "skill", "reranked": True})

    gated = _apply_relevance_gate(result, 0.4)

    assert gated.items == []
    assert gated.metadata["rerank_dropped"] == 0


def test_gate_drops_all_when_none_pass() -> None:
    result = _scored(("s1", 0.2), ("s2", 0.1))

    gated = _apply_relevance_gate(result, 0.4)

    assert gated.items == []
    assert gated.metadata["rerank_dropped"] == 2
