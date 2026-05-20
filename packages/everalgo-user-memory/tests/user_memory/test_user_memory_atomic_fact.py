"""Tests for everalgo.user_memory.atomic_fact — AtomicFactExtractor.

No internal retry — exceptions propagate directly to the caller.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.user_memory.atomic_fact import (
    AtomicFactExtractor,
    _parse_llm_response,
    _render_input_text,
    _validate_atomic_facts,
)


def _memcell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="Alice scheduled a 3pm meeting with Bob on 2024-03-14.",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            )
        ],
        timestamp=1700000000000,
    )


async def test_aextract_splits_atomic_facts_list_into_entities() -> None:
    """Each string in ``atomic_facts.atomic_fact`` becomes one AtomicFact entity."""
    llm_json = (
        '{"atomic_facts": {'
        '"time": "March 14, 2024(Thursday) at 3:00 PM UTC", '
        '"atomic_fact": ['
        '"Alice scheduled a 3pm meeting with Bob on 2024-03-14.",'
        '"The meeting is on the calendar."'
        "]}}"
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    facts = await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert len(facts) == 2
    assert facts[0].fact.startswith("Alice scheduled")
    assert facts[1].fact == "The meeting is on the calendar."
    assert facts[0].time_label == "March 14, 2024(Thursday) at 3:00 PM UTC"  # type: ignore[attr-defined]


async def test_aextract_owner_id_equals_sender_id() -> None:
    """``AtomicFact.owner_id`` must equal the ``sender_id`` argument."""
    llm_json = '{"atomic_facts": {"time": "T", "atomic_fact": ["f"]}}'
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    facts = await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert facts[0].owner_id == "u_alice"


async def test_aextract_owner_id_equals_custom_sender_id() -> None:
    """``AtomicFact.owner_id`` must equal a non-default ``sender_id``."""
    llm_json = '{"atomic_facts": {"time": "T", "atomic_fact": ["f"]}}'
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    facts = await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_custom")

    assert facts[0].owner_id == "u_custom"


async def test_aextract_raises_when_atomic_facts_missing() -> None:
    """No atomic_facts key → ValueError propagates immediately (no retry)."""
    bad = ChatResponse(content='{"unrelated": []}', model="fake")
    fake = FakeLLMClient(responses=[bad])

    with pytest.raises(ValueError, match="Missing 'atomic_facts'"):
        await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert fake.call_count == 1


@pytest.mark.asyncio
async def test_aextract_accepts_empty_atomic_fact_list() -> None:
    """Regression: empty atomic_fact list must return [] without raising.

    EverCore's original atomic_fact_extractor.py accepts empty arrays for
    MemCells with no extractable facts (e.g. greeting-only conversations).
    EverAlgo's port had added a stricter check that broke benchmark parity —
    removing it restores the EverCore contract.

    Surfaced by LoCoMo benchmark Stage 1 (smoke5: 2 ok, 1 failed).
    """
    response = ChatResponse(content='{"atomic_facts": {"time": "T", "atomic_fact": []}}', model="fake")
    fake = FakeLLMClient(responses=[response])

    facts = await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert facts == []
    assert fake.call_count == 1


async def test_aextract_raises_on_bad_json() -> None:
    """Unparseable JSON → ValueError propagates immediately."""
    bad = ChatResponse(content="not json", model="fake")
    fake = FakeLLMClient(responses=[bad])

    with pytest.raises((json.JSONDecodeError, ValueError)):
        await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert fake.call_count == 1


async def test_aextract_skips_non_string_or_empty_atomic_fact_items() -> None:
    llm_json = '{"atomic_facts": {"time": "T", "atomic_fact": ["good", "", null, 42, "also good"]}}'
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    facts = await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert [f.fact for f in facts] == ["good", "also good"]


async def test_aextract_per_call_prompt_overrides_default_uses_double_brace_replace() -> None:
    """Per-call prompt= goes through .replace() with double-brace placeholders."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(
            content='{"atomic_facts": {"time": "T", "atomic_fact": ["fact-1"]}}',
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)
    custom = "CUSTOM ATOMIC INPUT={{INPUT_TEXT}} TIME={{TIME}}"

    await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice", prompt=custom)

    assert captured["content"].startswith("CUSTOM ATOMIC")
    assert "Alice: Alice scheduled" in captured["content"]
    assert "{{INPUT_TEXT}}" not in captured["content"]
    assert "{{TIME}}" not in captured["content"]


# ==========================================================================
# _render_input_text skips empty content
# ==========================================================================


def test_render_input_text_skips_empty_content() -> None:
    cell = MemCell(
        items=[
            ChatMessage(id="m1", role="user", content="hi", timestamp=1, sender_id="u_alice", sender_name="Alice"),
            ChatMessage(id="m2", role="user", content="", timestamp=2, sender_id="u_bob", sender_name="Bob"),
        ],
        timestamp=2,
    )
    rendered = _render_input_text(cell)
    assert "Alice: hi" in rendered
    assert "Bob" not in rendered


# ==========================================================================
# _parse_llm_response — 4 parse strategies
# ==========================================================================


def test_parse_llm_response_handles_json_fence() -> None:
    """Strategy 1: ```json``` fenced response."""
    raw = '```json\n{"atomic_facts": {"time": "T", "atomic_fact": ["x"]}}\n```'
    parsed = _parse_llm_response(raw)
    assert parsed == {"atomic_facts": {"time": "T", "atomic_fact": ["x"]}}


def test_parse_llm_response_handles_generic_code_fence_with_lang_specifier() -> None:
    """Strategy 2: ```yaml / ```python / ```anything fence with leading lang line."""
    raw = '```python\n{"atomic_facts": {"time": "T", "atomic_fact": ["x"]}}\n```'
    parsed = _parse_llm_response(raw)
    assert parsed == {"atomic_facts": {"time": "T", "atomic_fact": ["x"]}}


def test_parse_llm_response_falls_back_to_regex_embedded_object() -> None:
    """Strategy 3: regex finds embedded ``{atomic_facts{time,atomic_fact}}`` object."""
    raw = 'Some prose {"atomic_facts": {"time": "T", "atomic_fact": ["x"]}} trailing'
    parsed = _parse_llm_response(raw)
    assert parsed == {"atomic_facts": {"time": "T", "atomic_fact": ["x"]}}


def test_parse_llm_response_falls_back_to_direct_load_with_strip() -> None:
    """Strategy 4: direct ``json.loads(raw.strip())``. Whitespace padding shouldn't break it."""
    raw = '   {"other": "shape"}   '
    parsed = _parse_llm_response(raw)
    assert parsed == {"other": "shape"}


def test_parse_llm_response_raises_when_all_strategies_fail() -> None:
    """All strategies fail → ValueError."""
    with pytest.raises(ValueError):
        _parse_llm_response("totally not json at all")


# ==========================================================================
# _validate_atomic_facts schema branches
# ==========================================================================


def test_validate_atomic_facts_raises_on_non_dict_input() -> None:
    """Top-level non-dict → ValueError."""
    with pytest.raises(ValueError, match="not a JSON object"):
        _validate_atomic_facts([1, 2, 3])


def test_validate_atomic_facts_raises_when_time_missing() -> None:
    """Missing ``time`` field → ValueError."""
    with pytest.raises(ValueError, match="Missing time"):
        _validate_atomic_facts({"atomic_facts": {"atomic_fact": ["x"]}})


def test_validate_atomic_facts_raises_when_atomic_fact_key_missing() -> None:
    """Missing ``atomic_fact`` key → ValueError."""
    with pytest.raises(ValueError, match="Missing atomic_fact"):
        _validate_atomic_facts({"atomic_facts": {"time": "T"}})


def test_validate_atomic_facts_raises_when_atomic_fact_not_a_list() -> None:
    """``atomic_fact`` not a list → ValueError."""
    with pytest.raises(ValueError, match="atomic_fact is not a list"):
        _validate_atomic_facts({"atomic_facts": {"time": "T", "atomic_fact": "single string"}})


# ==========================================================================
# Silent-skip contract — agent → user-memory pipeline
# ==========================================================================


async def test_aextract_silently_skips_non_chat_items() -> None:
    """AtomicFactExtractor must silently skip ToolCallRequest / ToolCallResult items.

    Locks the agent → user-memory pipeline contract: a MemCell with mixed items (ChatMessage +
    tool calls) must produce the same AtomicFact list as a chat-only MemCell.
    """
    llm_json = '{"atomic_facts": {"time": "March 14, 2024", "atomic_fact": ["Alice scheduled a meeting."]}}'

    chat_only_cell = MemCell(
        items=[
            ChatMessage(
                id="c1",
                role="user",
                content="Alice scheduled a meeting.",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            ),
        ],
        timestamp=1700000000000,
    )
    mixed_cell = MemCell(
        items=[
            ChatMessage(
                id="c1",
                role="user",
                content="Alice scheduled a meeting.",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ToolCallRequest(
                tool_calls=[ToolCall(id="tc1", function=ToolCallFunction(name="calendar.create", arguments="{}"))],
                timestamp=1700000001000,
                sender_id="assistant",
            ),
            ToolCallResult(
                tool_call_id="tc1",
                content="Done.",
                timestamp=1700000002000,
            ),
        ],
        timestamp=1700000002000,
    )

    fake_chat = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])
    fake_mixed = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    facts_chat = await AtomicFactExtractor(llm=fake_chat).aextract(chat_only_cell, sender_id="u_alice")
    facts_mixed = await AtomicFactExtractor(llm=fake_mixed).aextract(mixed_cell, sender_id="u_alice")

    assert len(facts_chat) == len(facts_mixed) == 1
    assert facts_chat[0].fact == facts_mixed[0].fact
    assert facts_chat[0].owner_id == facts_mixed[0].owner_id


@pytest.mark.asyncio
async def test_extract_generic_when_sender_id_is_none() -> None:
    """sender_id=None → owner_id=None on every emitted AtomicFact; prompt rendering identical to user-tagged path."""
    llm_json = '{"atomic_facts": {"time": "T", "atomic_fact": ["f"]}}'
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])
    facts = await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id=None)
    assert len(facts) >= 1
    assert all(f.owner_id is None for f in facts)
