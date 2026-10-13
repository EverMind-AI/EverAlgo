"""Unit tests for ``everalgo.rank.rerank``."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from everalgo.llm import LLMOutputError
from everalgo.rank import rerank as rerank_mod
from everalgo.rank._replies import _RerankReply
from everalgo.rank.prompts.en.episodic import EPISODIC_RERANK_PROMPT_EN
from everalgo.rank.prompts.rerank_user import RERANK_USER_PROMPT
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate
from everalgo.utils import render_prompt

if TYPE_CHECKING:
    from everalgo.llm import LLMMessage


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


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        json.dumps({"items": [{"id": "a", "score": 0.5}]}),
        json.dumps({"ranked": [{"id": "a"}]}),
        json.dumps({"ranked": [{"id": "a", "score": 0.5}], "reasoning": "extra"}),
        json.dumps({"ranked": [{"id": "a", "score": 0.5, "why": "extra"}]}),
        json.dumps({"ranked": [{"id": 7, "score": 0.5}]}),
    ],
    ids=["not_json", "missing_ranked", "missing_score", "extra_field", "extra_item_field", "id_not_str"],
)
async def test_arerank_raises_llm_output_error_on_invalid_reply(reply: str) -> None:
    """A reply off the ``ranked`` schema raises ``LLMOutputError`` immediately (fail-loud, no retry)."""
    fake = FakeLLMClient(responses=[reply])

    with pytest.raises(LLMOutputError, match="rerank reply is not a valid _RerankReply"):
        await rerank_mod.arerank(_items(), query="show me history", prompt=EPISODIC_RERANK_PROMPT_EN, top_k=5, llm=fake)


async def test_arerank_requests_the_rerank_model_with_no_temperature() -> None:
    fake = FakeLLMClient(responses=[json.dumps({"ranked": []})])

    await rerank_mod.arerank(_items(), query="q", prompt=EPISODIC_RERANK_PROMPT_EN, top_k=5, llm=fake)

    record = fake.calls[0]
    assert record.response_format is _RerankReply
    assert (record.temperature, record.max_tokens) == (None, None)


async def test_arerank_sends_the_rules_with_top_k_then_the_query_and_candidates() -> None:
    fake = FakeLLMClient(responses=[json.dumps({"ranked": []})])

    await rerank_mod.arerank(_items(), query="show me history", prompt=EPISODIC_RERANK_PROMPT_EN, top_k=2, llm=fake)

    system, user = fake.calls[0].messages
    payload = json.dumps([{"id": c.id, "score": c.score} for c in _items()])
    assert (system.role, system.content) == ("system", render_prompt(EPISODIC_RERANK_PROMPT_EN, None, top_k=2))
    assert "Return at most 2 items" in str(system.content)
    assert (user.role, user.content) == (
        "user",
        render_prompt(RERANK_USER_PROMPT, None, query="show me history", candidates_json=payload),
    )
    assert user.content == f"<user_query>\nshow me history\n</user_query>\n\n<candidates>\n{payload}\n</candidates>"


async def test_arerank_query_is_never_rendered_as_a_template() -> None:
    fake = FakeLLMClient(responses=[json.dumps({"ranked": []})])

    await rerank_mod.arerank(_items(), query="{{top_k}} {{candidates_json}}", prompt="Rules.", top_k=2, llm=fake)

    assert str(fake.calls[0].messages[1].content).startswith("<user_query>\n{{top_k}} {{candidates_json}}\n")


def test_arerank_raises_on_a_prompt_written_for_the_single_message_call() -> None:
    """An override that still carries the data slots would send them literally; it fails loud instead."""
    fake = FakeLLMClient(responses=[json.dumps({"ranked": []})])

    with pytest.raises(ValueError, match=r"\{\{query\}\} \(no value supplied\)"):
        rerank_mod.rerank(_items(), query="q", prompt="Rank for {{query}} at most {{top_k}}.", top_k=2, llm=fake)

    assert fake.call_count == 0


def test_arerank_raises_on_prompt_with_unknown_placeholder() -> None:
    """A prompt naming a slot we do not supply must fail loudly rather than ship the literal slot."""
    fake = FakeLLMClient(responses=[json.dumps({"ranked": []})])

    with pytest.raises(ValueError, match=r"\{\{nonexistent\}\} \(no value supplied\)"):
        rerank_mod.rerank(
            _items(),
            query="test query",
            prompt="bad template {{nonexistent}}",
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

    def _handler(messages: list[LLMMessage], **_kwargs: object) -> str:
        content = messages[-1].content
        assert isinstance(content, str)
        captured_prompt.append(content)
        return json.dumps({"ranked": [{"id": "dt_item", "score": 0.88}]})

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

    # The serialized datetime string must appear in the user message sent to the LLM
    assert len(captured_prompt) == 1
    assert ts_str in captured_prompt[0]

    # Result shape: one item with the LLM-assigned score
    assert len(out) == 1
    assert out[0].id == "dt_item"
    assert out[0].score == pytest.approx(0.88)  # pyright: ignore[reportUnknownMemberType]
