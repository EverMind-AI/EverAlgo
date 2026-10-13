"""Tests for everalgo.user_memory.atomic_fact — AtomicFactExtractor.aextract, the MemCell path.

One call, no internal retry: an invalid reply raises LLMOutputError, and client exceptions propagate unchanged.
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from everalgo.llm import LLMMessage, LLMOutputError
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.user_memory import OutputLanguage
from everalgo.user_memory._replies import _AtomicFactsReply
from everalgo.user_memory.atomic_fact import (
    AtomicFactExtractor,
    _render_input_text,
)
from everalgo.user_memory.prompts.en.atomic_fact import ATOMIC_FACT_PROMPT


def _reply(*facts: str) -> str:
    return json.dumps({"atomic_facts": {"atomic_fact": list(facts)}})


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
    reply = _reply(
        "Alice scheduled a 3pm meeting with Bob on 2024-03-14.",
        "The meeting is on the calendar.",
    )
    fake = FakeLLMClient(responses=[reply])

    facts = await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert len(facts) == 2
    assert facts[0].content.startswith("Alice scheduled")
    assert facts[1].content == "The meeting is on the calendar."
    assert {fact.timestamp for fact in facts} == {1700000000000}


async def test_aextract_owner_id_equals_sender_id() -> None:
    """``AtomicFact.owner_id`` must equal the ``sender_id`` argument."""
    fake = FakeLLMClient(responses=[_reply("f")])

    facts = await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert facts[0].owner_id == "u_alice"


async def test_aextract_owner_id_equals_custom_sender_id() -> None:
    """``AtomicFact.owner_id`` must equal a non-default ``sender_id``."""
    fake = FakeLLMClient(responses=[_reply("f")])

    facts = await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_custom")

    assert facts[0].owner_id == "u_custom"


async def test_aextract_sends_one_user_message_and_requests_the_reply_model() -> None:
    """The MemCell path keeps its single-message prompt; only the reply contract moved."""
    fake = FakeLLMClient(responses=[_reply("f")])

    await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    (message,) = fake.calls[0].messages
    assert message.role == "user"
    assert fake.calls[0].response_format is _AtomicFactsReply


async def test_aextract_raises_when_atomic_facts_missing() -> None:
    """No atomic_facts key → LLMOutputError on the one call, naming the location, not the reply."""
    fake = FakeLLMClient(responses=['{"unrelated": []}'])

    with pytest.raises(LLMOutputError, match="atomic_facts") as excinfo:
        await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert "unrelated: Extra inputs are not permitted" in str(excinfo.value)
    assert "[]" not in str(excinfo.value)
    assert fake.call_count == 1


async def test_aextract_accepts_empty_atomic_fact_list() -> None:
    """Regression: empty atomic_fact list must return [] without raising.

    the upstream reference's original atomic_fact_extractor.py accepts empty arrays for
    MemCells with no extractable facts (e.g. greeting-only conversations).
    EverAlgo's port had added a stricter check that broke benchmark parity —
    removing it restores the upstream reference contract.

    Surfaced by LoCoMo benchmark Stage 1 (smoke5: 2 ok, 1 failed).
    """
    fake = FakeLLMClient(responses=[_reply()])

    facts = await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert facts == []
    assert fake.call_count == 1


async def test_aextract_raises_on_bad_json() -> None:
    """Unparseable JSON → LLMOutputError on the one call (no internal retry)."""
    fake = FakeLLMClient(responses=["not json"])

    with pytest.raises(LLMOutputError):
        await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert fake.call_count == 1


async def test_aextract_skips_blank_atomic_fact_items() -> None:
    fake = FakeLLMClient(responses=[_reply("good", "", "   ", " also good ")])

    facts = await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert [f.content for f in facts] == ["good", "also good"]


async def test_aextract_rejects_non_string_atomic_fact_items() -> None:
    """The strict model rejects what the brace-matching parser used to drop silently."""
    fake = FakeLLMClient(responses=['{"atomic_facts": {"atomic_fact": ["good", null, 42]}}'])

    with pytest.raises(LLMOutputError, match=r"atomic_facts\.atomic_fact\.1: Input should be a valid string"):
        await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")


def test_aextract_takes_no_prompt_override() -> None:
    """``prompt=`` is gone from both entry points; this path's bundled prompt is the only one."""
    assert list(inspect.signature(AtomicFactExtractor.aextract).parameters) == [
        "self",
        "memcell",
        "sender_id",
        "output_language",
    ]


# ==========================================================================
# _render_input_text
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


def test_render_input_text_prefixes_each_line_with_the_utc_minute() -> None:
    """Seconds are truncated, not rounded: 14:00:59.999 still renders as 14:00."""
    cell = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hi",
                timestamp=1_712_757_659_999,
                sender_id="u_alice",
                sender_name="Alice",
            ),
        ],
        timestamp=1_712_757_659_999,
    )
    assert _render_input_text(cell) == "[2024-04-10 14:00 UTC (Wednesday)] Alice: hi"


# ==========================================================================
# Silent-skip contract — agent → user-memory pipeline
# ==========================================================================


async def test_aextract_silently_skips_non_chat_items() -> None:
    """AtomicFactExtractor must silently skip ToolCallRequest / ToolCallResult items.

    Locks the agent → user-memory pipeline contract: a MemCell with mixed items (ChatMessage +
    tool calls) must produce the same AtomicFact list as a chat-only MemCell.
    """
    reply = _reply("Alice scheduled a meeting.")

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

    fake_chat = FakeLLMClient(responses=[reply])
    fake_mixed = FakeLLMClient(responses=[reply])

    facts_chat = await AtomicFactExtractor(llm=fake_chat).aextract(chat_only_cell, sender_id="u_alice")
    facts_mixed = await AtomicFactExtractor(llm=fake_mixed).aextract(mixed_cell, sender_id="u_alice")

    assert len(facts_chat) == len(facts_mixed) == 1
    assert facts_chat[0].content == facts_mixed[0].content
    assert facts_chat[0].owner_id == facts_mixed[0].owner_id


async def test_extract_generic_when_sender_id_is_none() -> None:
    """sender_id=None → owner_id=None on every emitted AtomicFact; prompt rendering identical to user-tagged path."""
    fake = FakeLLMClient(responses=[_reply("f")])
    facts = await AtomicFactExtractor(llm=fake).aextract(_memcell(), sender_id=None)
    assert len(facts) >= 1
    assert all(f.owner_id is None for f in facts)


# ==========================================================================
# Language rules — mixed-input clauses (mirrors test_user_memory_foresight.py)
# ==========================================================================


async def test_time_is_the_first_message_utc_minute_with_its_weekday() -> None:
    """24-hour and zone-labelled: the 12-hour label this replaced read ``10:13 PM`` here."""
    rendered = await _render_atomic_fact_prompt()

    assert "\nConversation start time: 2023-11-14 22:13 UTC (Tuesday)\n" in rendered


def test_prompt_asks_for_no_time_echo() -> None:
    """The reply is ``{"atomic_facts": {"atomic_fact": [...]}}``; ``TIME`` is an input only."""
    assert '"time"' not in ATOMIC_FACT_PROMPT
    assert "* The `TIME` input represents the conversation start time." in ATOMIC_FACT_PROMPT


def test_prompt_carries_the_language_placeholder_at_both_ends() -> None:
    """Long prompts lose middle instructions, so the rule is spliced at head and tail."""
    assert ATOMIC_FACT_PROMPT.count("{{language_rule}}") == 2


async def test_default_prompt_renders_with_no_placeholder_left_behind() -> None:
    """Every slot the shipped template names is filled by the ``aextract`` call site."""
    rendered = await _render_atomic_fact_prompt()

    assert re.search(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}", rendered) is None, rendered


async def test_rendering_injects_the_participant_rule_when_no_language_is_named() -> None:
    rendered = await _render_atomic_fact_prompt()

    assert rendered.count("CRITICAL LANGUAGE RULE") == 2
    assert "the language the participants use" in rendered
    assert "{{language_rule}}" not in rendered


async def test_rendering_injects_the_named_language() -> None:
    rendered = await _render_atomic_fact_prompt(output_language=OutputLanguage.GERMAN)

    assert (
        rendered.count(
            "Write every output field in German, whatever language the input is in, including anything already stored."
        )
        == 2
    )
    assert "German, whatever language the input is in." not in rendered
    assert "CRITICAL LANGUAGE RULE" not in rendered
    assert "the language the participants use" not in rendered


async def _render_atomic_fact_prompt(output_language: OutputLanguage | None = None) -> str:
    """Capture what the extractor hands the LLM; the rule only exists after rendering."""
    captured: list[LLMMessage] = []

    def handler(messages: list[LLMMessage], **_: object) -> str:
        captured.extend(messages)
        return _reply()

    await AtomicFactExtractor(llm=FakeLLMClient(handler=handler)).aextract(
        _memcell(), sender_id="u_alice", output_language=output_language
    )
    (message,) = captured
    return str(message.content)
