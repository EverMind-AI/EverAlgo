"""Unit tests for ``everalgo.rank.rerank``."""

from __future__ import annotations

import json
import math

import pytest

from everalgo.llm.types import ChatMessage, ChatResponse
from everalgo.rank import rerank as rerank_mod
from everalgo.rank.prompts.en.episodic import EPISODIC_RERANK_PROMPT_EN
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate


def _items() -> list[Candidate]:
    return [
        Candidate(id="a", score=0.3, metadata={}),
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

    out = await rerank_mod.arerank(
        _items(), query="show me history", prompt=EPISODIC_RERANK_PROMPT_EN, top_k=5, llm=fake
    )

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

    out = await rerank_mod.arerank(
        _items(), query="show me history", prompt=EPISODIC_RERANK_PROMPT_EN, top_k=2, llm=fake
    )

    assert len(out) == 2
    assert [c.id for c in out] == ["c", "a"]


async def test_arerank_drops_hallucinated_and_omitted_ids() -> None:
    fake = FakeLLMClient(responses=[json.dumps({"ranked": [{"id": "ghost", "score": 1.0}, {"id": "a", "score": 0.5}]})])

    out = await rerank_mod.arerank(
        _items(), query="show me history", prompt=EPISODIC_RERANK_PROMPT_EN, top_k=5, llm=fake
    )

    assert [c.id for c in out] == ["a"]


async def test_arerank_returns_empty_for_empty_input() -> None:
    fake = FakeLLMClient(responses=[json.dumps({"ranked": []})])

    out = await rerank_mod.arerank([], query="test query", prompt=EPISODIC_RERANK_PROMPT_EN, top_k=5, llm=fake)

    assert out == []
    assert fake.call_count == 0


async def test_arerank_raises_on_invalid_response() -> None:
    """Invalid LLM response (not matching schema) → ValueError immediately (fail-loud, no retry)."""
    fake = FakeLLMClient(responses=["not json at all"])

    with pytest.raises(ValueError):
        await rerank_mod.arerank(_items(), query="show me history", prompt=EPISODIC_RERANK_PROMPT_EN, top_k=5, llm=fake)


def test_arerank_raises_on_prompt_with_unknown_placeholder() -> None:
    """A prompt referencing keys we don't supply must surface as KeyError."""
    fake = FakeLLMClient(responses=[json.dumps({"ranked": []})])

    with pytest.raises(KeyError):
        rerank_mod.rerank(
            _items(),
            query="test query",
            prompt="bad template {nonexistent}",
            top_k=2,
            llm=fake,
        )


def test_sync_bridge_callable_from_pytest() -> None:
    """``rerank`` (sync) should work outside an event loop."""
    fake = FakeLLMClient(responses=[json.dumps({"ranked": [{"id": "a", "score": 0.99}]})])

    out = rerank_mod.rerank(_items(), query="show me history", prompt=EPISODIC_RERANK_PROMPT_EN, top_k=1, llm=fake)

    assert [c.id for c in out] == ["a"]


async def test_arerank_serializes_non_native_metadata_via_default_str() -> None:
    """``default=str`` in json.dumps must prevent TypeError when metadata contains datetime.

    Regression guard: before the fix, passing a Candidate with a ``datetime`` value
    in ``metadata`` (e.g. a LanceDB ``timestamp`` column forwarded by EverOS's
    ``row_to_candidate``) raised ``TypeError: Object of type datetime is not JSON
    serializable``.  After the fix, the value is serialized via ``str()``, which
    produces the ISO-8601-like representation that is sufficient for LLM prompting.
    """
    from datetime import UTC, datetime

    ts = datetime(2026, 5, 19, 14, 0, 0, tzinfo=UTC)
    ts_str = str(ts)  # "2026-05-19 14:00:00+00:00"

    captured_prompt: list[str] = []

    def _handler(messages: list[ChatMessage], **_kwargs: object) -> ChatResponse:
        content = messages[-1].content
        assert isinstance(content, str)
        captured_prompt.append(content)
        return ChatResponse(
            content=json.dumps({"ranked": [{"id": "dt_item", "score": 0.88}]}),
            model="fake",
            usage=None,
            finish_reason="stop",
            raw=None,
        )

    fake = FakeLLMClient(handler=_handler)

    candidate = Candidate(
        id="dt_item",
        score=0.6,
        metadata={"timestamp": ts},
    )

    # Must not raise TypeError
    out = await rerank_mod.arerank(
        [candidate], query="recent events", prompt=EPISODIC_RERANK_PROMPT_EN, top_k=5, llm=fake
    )

    # The serialized datetime string must appear in the prompt sent to the LLM
    assert len(captured_prompt) == 1
    assert ts_str in captured_prompt[0]

    # Result shape: one item with the LLM-assigned score
    assert len(out) == 1
    assert out[0].id == "dt_item"
    assert out[0].score == pytest.approx(0.88)  # pyright: ignore[reportUnknownMemberType]
