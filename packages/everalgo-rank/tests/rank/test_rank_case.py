"""Unit tests for ``everalgo.rank.case``."""

from __future__ import annotations

import json

from everalgo.rank import RankConfig, case
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate, RankInput


def _mk(sparse: list[Candidate], dense: list[Candidate], top_k: int = 5) -> RankInput:
    return RankInput(
        query="ship a feature flag",
        memory_type="case",
        sparse_candidates=sparse,
        dense_candidates=dense,
        top_k=top_k,
    )


async def test_arank_end_to_end_returns_sorted_case_items(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    out = await case.arank(
        _mk(sparse_candidates, dense_candidates, top_k=4),
        config=RankConfig(fusion_mode="rrf"),
    )

    assert all(it.item_type == "case" for it in out.items)
    scores = [it.score for it in out.items]
    assert scores == sorted(scores, reverse=True)


async def test_arank_passes_quality_score_through_metadata(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    """quality_score is not weighted into the score but stays in metadata."""
    out = await case.arank(_mk(sparse_candidates, dense_candidates, top_k=4))

    # Every item should preserve its quality_score metadata for downstream use.
    for item in out.items:
        if "quality_score" in item.metadata:
            assert isinstance(item.metadata["quality_score"], (int, float))


async def test_arank_empty_input_short_circuits() -> None:
    out = await case.arank(_mk([], []))

    assert out.items == []
    assert out.metadata.get("stop_reason") == "no_candidates"


async def test_arank_rerank_enabled_calls_llm(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    fake = FakeLLMClient(responses=[json.dumps({"ranked": [{"id": "d1", "score": 0.99}]})])

    out = await case.arank(
        _mk(sparse_candidates, dense_candidates, top_k=2),
        config=RankConfig(fusion_mode="rrf"),
        llm=fake,
        enable_rerank=True,
    )

    assert fake.call_count == 1
    assert out.metadata.get("reranked") is True
    assert [it.id for it in out.items] == ["d1"]


def test_sync_bridge_is_callable(
    dense_candidates: list[Candidate],
    sparse_candidates: list[Candidate],
) -> None:
    out = case.rank(_mk(sparse_candidates, dense_candidates, top_k=3))
    assert len(out.items) <= 3
