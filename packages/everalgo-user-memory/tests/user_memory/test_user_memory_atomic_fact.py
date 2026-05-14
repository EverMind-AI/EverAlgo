"""Tests for everalgo.user_memory.atomic_fact — AtomicFactExtractor (new-release schema)."""

from __future__ import annotations

from typing import Any

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import MemCell, Message, MessageRole
from everalgo.user_memory.atomic_fact import (
    AtomicFactExtractor,
    _derive_owner_id,
    _parse_llm_response,
    _render_input_text,
    _validate_atomic_facts,
)


def _memcell() -> MemCell:
    msg = Message(
        role=MessageRole.USER,
        content="Alice scheduled a 3pm meeting with Bob on 2024-03-14.",
        timestamp=1700000000000,
        sender_id="u_alice",
        sender_name="Alice",
    )
    return MemCell(
        event_id="mc_af_001",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1700000000000,
        participants=["u_alice"],
        sender_ids=["u_alice"],
    )


async def test_aextract_splits_atomic_facts_list_into_entities() -> None:
    """Each string in new-release ``atomic_facts.atomic_fact`` becomes one AtomicFact entity."""
    llm_json = (
        '{"atomic_facts": {'
        '"time": "March 14, 2024(Thursday) at 3:00 PM UTC", '
        '"atomic_fact": ['
        '"Alice scheduled a 3pm meeting with Bob on 2024-03-14.",'
        '"The meeting is on the calendar."'
        "]}}"
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    facts = await AtomicFactExtractor().aextract(_memcell(), llm=fake)

    assert len(facts) == 2
    assert facts[0].fact.startswith("Alice scheduled")
    assert facts[1].fact == "The meeting is on the calendar."
    assert facts[0].time_label == "March 14, 2024(Thursday) at 3:00 PM UTC"  # type: ignore[attr-defined]


async def test_aextract_auto_fills_parent_id_and_owner_id() -> None:
    llm_json = '{"atomic_facts": {"time": "T", "atomic_fact": ["f"]}}'
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])
    mc = _memcell()

    facts = await AtomicFactExtractor().aextract(mc, llm=fake)

    assert facts[0].parent_id == mc.event_id
    assert facts[0].parent_type == "memcell"
    assert facts[0].owner_id == "u_alice"


async def test_aextract_raises_runtimeerror_when_atomic_facts_missing_after_5_retries() -> None:
    """No atomic_facts key → ValueError → retry 5 times → RuntimeError."""
    import pytest

    bad = ChatResponse(content='{"unrelated": []}', model="fake")
    fake = FakeLLMClient(responses=[bad, bad, bad, bad, bad])

    with pytest.raises(RuntimeError, match="all 5 retries exhausted"):
        await AtomicFactExtractor().aextract(_memcell(), llm=fake)
    assert fake.call_count == 5


async def test_aextract_raises_runtimeerror_when_atomic_fact_list_empty_after_5_retries() -> None:
    """Empty atomic_fact list → ValueError → retry 5 times → RuntimeError."""
    import pytest

    bad = ChatResponse(content='{"atomic_facts": {"time": "T", "atomic_fact": []}}', model="fake")
    fake = FakeLLMClient(responses=[bad, bad, bad, bad, bad])

    with pytest.raises(RuntimeError, match="all 5 retries exhausted"):
        await AtomicFactExtractor().aextract(_memcell(), llm=fake)
    assert fake.call_count == 5


async def test_aextract_skips_non_string_or_empty_atomic_fact_items() -> None:
    llm_json = '{"atomic_facts": {"time": "T", "atomic_fact": ["good", "", null, 42, "also good"]}}'
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    facts = await AtomicFactExtractor().aextract(_memcell(), llm=fake)

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

    await AtomicFactExtractor().aextract(_memcell(), llm=fake, prompt=custom)

    assert captured["content"].startswith("CUSTOM ATOMIC")
    assert "Alice: Alice scheduled" in captured["content"]
    assert "{{INPUT_TEXT}}" not in captured["content"]
    assert "{{TIME}}" not in captured["content"]


# ==========================================================================
# _render_input_text skips empty content (line 103)
# ==========================================================================


def test_render_input_text_skips_empty_content() -> None:
    real = Message(role=MessageRole.USER, content="hi", timestamp=1, sender_name="Alice")
    empty = Message(role=MessageRole.USER, content="", timestamp=2, sender_name="Bob")
    cell = MemCell(
        event_id="mc_x",
        original_data=[
            {"message": real.model_dump(exclude_none=True)},
            {"message": empty.model_dump(exclude_none=True)},
        ],
        timestamp=2,
    )
    rendered = _render_input_text(cell)
    assert "Alice: hi" in rendered
    assert "Bob" not in rendered


# ==========================================================================
# _parse_llm_response — 4 parse strategies (lines 127-160)
# ==========================================================================


def test_parse_llm_response_handles_json_fence() -> None:
    """Strategy 1: ```json``` fenced response (lines 127-133)."""
    raw = '```json\n{"atomic_facts": {"time": "T", "atomic_fact": ["x"]}}\n```'
    parsed = _parse_llm_response(raw)
    assert parsed == {"atomic_facts": {"time": "T", "atomic_fact": ["x"]}}


def test_parse_llm_response_handles_generic_code_fence_with_lang_specifier() -> None:
    """Strategy 2: ```yaml / ```python / ```anything fence with leading lang line (lines 135-146)."""
    raw = '```python\n{"atomic_facts": {"time": "T", "atomic_fact": ["x"]}}\n```'
    parsed = _parse_llm_response(raw)
    assert parsed == {"atomic_facts": {"time": "T", "atomic_fact": ["x"]}}


def test_parse_llm_response_falls_back_to_regex_embedded_object() -> None:
    """Strategy 3: regex finds embedded ``{atomic_facts{time,atomic_fact}}`` object (lines 155-156)."""
    raw = 'Some prose {"atomic_facts": {"time": "T", "atomic_fact": ["x"]}} trailing'
    parsed = _parse_llm_response(raw)
    assert parsed == {"atomic_facts": {"time": "T", "atomic_fact": ["x"]}}


def test_parse_llm_response_falls_back_to_direct_load_with_strip() -> None:
    """Strategy 4: direct ``json.loads(raw.strip())`` (lines 159-160). Whitespace padding shouldn't break it."""
    raw = '   {"other": "shape"}   '
    parsed = _parse_llm_response(raw)
    assert parsed == {"other": "shape"}


def test_parse_llm_response_raises_when_all_strategies_fail() -> None:
    """All 4 strategies fail → ValueError (line 160 raise)."""
    import pytest

    with pytest.raises(ValueError, match="Unable to parse"):
        _parse_llm_response("totally not json at all")


# ==========================================================================
# _validate_atomic_facts schema branches (lines 170, 177, 179, 182)
# ==========================================================================


def test_validate_atomic_facts_raises_on_non_dict_input() -> None:
    """Top-level non-dict → ValueError (line 170)."""
    import pytest

    with pytest.raises(ValueError, match="not a JSON object"):
        _validate_atomic_facts([1, 2, 3])


def test_validate_atomic_facts_raises_when_time_missing() -> None:
    """Missing ``time`` field → ValueError (line 177)."""
    import pytest

    with pytest.raises(ValueError, match="Missing time"):
        _validate_atomic_facts({"atomic_facts": {"atomic_fact": ["x"]}})


def test_validate_atomic_facts_raises_when_atomic_fact_key_missing() -> None:
    """Missing ``atomic_fact`` key → ValueError (line 179)."""
    import pytest

    with pytest.raises(ValueError, match="Missing atomic_fact"):
        _validate_atomic_facts({"atomic_facts": {"time": "T"}})


def test_validate_atomic_facts_raises_when_atomic_fact_not_a_list() -> None:
    """``atomic_fact`` not a list → ValueError (line 182)."""
    import pytest

    with pytest.raises(ValueError, match="atomic_fact is not a list"):
        _validate_atomic_facts({"atomic_facts": {"time": "T", "atomic_fact": "single string"}})


# ==========================================================================
# _derive_owner_id fallbacks (lines 217-220)
# ==========================================================================


def test_derive_owner_id_falls_back_to_message_sender_id() -> None:
    """No participants → first message with sender_id wins (lines 217-218)."""
    msg = Message(role=MessageRole.USER, content="x", timestamp=1, sender_id="u_from_msg")
    cell = MemCell(
        event_id="mc_x",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1,
    )
    assert _derive_owner_id(cell) == "u_from_msg"


def test_derive_owner_id_returns_u_default_when_nothing_identifies_user() -> None:
    """No participants, no sender_id → ``u_default`` (line 220)."""
    msg = Message(role=MessageRole.USER, content="x", timestamp=1)
    cell = MemCell(
        event_id="mc_x",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1,
    )
    assert _derive_owner_id(cell) == "u_default"
