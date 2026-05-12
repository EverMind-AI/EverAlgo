"""Unit tests for ``everalgo.rank.skill``."""

from __future__ import annotations

from everalgo.rank import RankConfig, skill
from everalgo.types import Candidate, RankInput


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
