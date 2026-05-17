"""Unit tests for ``everalgo.rank.rerank``."""

from __future__ import annotations

import json
import math

import pytest

from everalgo.rank import rerank as rerank_mod
from everalgo.rank.prompts.en.episodic import EPISODIC_RERANK_PROMPT_EN
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate


def _items() -> list[Candidate]:
    return [
        Candidate(id="a", score=0.3, metadata={"__rerank_query__": "show me history"}),
        Candidate(id="b", score=0.5, metadata={}),
        Candidate(id="c", score=0.7, metadata={}),
    ]


async def test_arerank_replaces_scores_and_sorts_descending() -> None:
    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "ranked": [
                        {"id": "b", "score": 0.95},
                        {"id": "a", "score": 0.80},
                        {"id": "c", "score": 0.40},
                    ]
                }
            )
        ]
    )

    out = await rerank_mod.arerank(_items(), prompt=EPISODIC_RERANK_PROMPT_EN, top_k=5, llm=fake)

    assert [c.id for c in out] == ["b", "a", "c"]
    assert math.isclose(out[0].score, 0.95)
    # fusion_score is preserved for audit
    assert math.isclose(out[0].metadata["fusion_score"], 0.5)


async def test_arerank_respects_top_k() -> None:
    fake = FakeLLMClient(
        responses=[
            json.dumps(
                {
                    "ranked": [
                        {"id": "c", "score": 0.9},
                        {"id": "a", "score": 0.7},
                        {"id": "b", "score": 0.5},
                    ]
                }
            )
        ]
    )

    out = await rerank_mod.arerank(_items(), prompt=EPISODIC_RERANK_PROMPT_EN, top_k=2, llm=fake)

    assert len(out) == 2
    assert [c.id for c in out] == ["c", "a"]


async def test_arerank_drops_hallucinated_and_omitted_ids() -> None:
    fake = FakeLLMClient(responses=[json.dumps({"ranked": [{"id": "ghost", "score": 1.0}, {"id": "a", "score": 0.5}]})])

    out = await rerank_mod.arerank(_items(), prompt=EPISODIC_RERANK_PROMPT_EN, top_k=5, llm=fake)

    assert [c.id for c in out] == ["a"]


async def test_arerank_returns_empty_for_empty_input() -> None:
    fake = FakeLLMClient(responses=[json.dumps({"ranked": []})])

    out = await rerank_mod.arerank([], prompt=EPISODIC_RERANK_PROMPT_EN, top_k=5, llm=fake)

    assert out == []
    assert fake.call_count == 0


async def test_arerank_raises_on_non_json_response() -> None:
    """Non-JSON LLM response → JSONDecodeError propagates (no fusion-order fallback)."""
    fake = FakeLLMClient(responses=["not json at all"])

    with pytest.raises(json.JSONDecodeError):
        await rerank_mod.arerank(_items(), prompt=EPISODIC_RERANK_PROMPT_EN, top_k=5, llm=fake)


def test_arerank_raises_on_prompt_with_unknown_placeholder() -> None:
    """A prompt referencing keys we don't supply must surface as KeyError."""
    fake = FakeLLMClient(responses=[json.dumps({"ranked": []})])

    with pytest.raises(KeyError):
        rerank_mod.rerank(
            _items(),
            prompt="bad template {nonexistent}",
            top_k=2,
            llm=fake,
        )


def test_sync_bridge_callable_from_pytest() -> None:
    """``rerank`` (sync) should work outside an event loop."""
    fake = FakeLLMClient(responses=[json.dumps({"ranked": [{"id": "a", "score": 0.99}]})])

    out = rerank_mod.rerank(_items(), prompt=EPISODIC_RERANK_PROMPT_EN, top_k=1, llm=fake)

    assert [c.id for c in out] == ["a"]
