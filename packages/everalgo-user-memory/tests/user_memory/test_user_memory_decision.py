"""Tests for everalgo.user_memory.decision — DecisionExtractor.

No internal retry — exceptions propagate directly. Empty list is a valid successful return.
There is no sender_id: owner_id is always None.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.user_memory import OutputLanguage
from everalgo.user_memory.decision import DecisionExtractor, _render_conversation


def _memcell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="Core Agent Runtime stays Python; device runtime stays Rust.",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            )
        ],
        timestamp=1700000000000,
    )


def _decision_payload(**overrides: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "title": "Agent Runtime language choice",
        "decision": "Use Python for the core Agent runtime and Rust on device.",
        "reason": "Python fits fast iteration; Rust fits stable device operation.",
        "impact": "Device capabilities connect through APIs.",
        "tags": ["architecture", "runtime"],
    }
    item.update(overrides)
    return item


# ==========================================================================
# Language rules — mixed-input clauses (mirrors test_user_memory_foresight.py)
# ==========================================================================


def test_prompt_carries_the_language_placeholder_at_both_ends() -> None:
    """Long prompts lose middle instructions, so the rule is spliced at head and tail."""
    from everalgo.user_memory.prompts.en.decision import DECISION_GENERATION_PROMPT

    assert DECISION_GENERATION_PROMPT.count("{language_rule}") == 2


def test_default_prompt_has_no_per_sender_placeholders() -> None:
    """Whole-memcell extract: no USER_ID / USER_NAME slots for a Foresight-style fan-out."""
    from everalgo.user_memory.prompts.en.decision import DECISION_GENERATION_PROMPT

    assert "{USER_ID}" not in DECISION_GENERATION_PROMPT
    assert "{USER_NAME}" not in DECISION_GENERATION_PROMPT


async def test_rendering_injects_the_participant_rule_when_no_language_is_named() -> None:
    rendered = await _render_decision_prompt()

    assert rendered.count("CRITICAL LANGUAGE RULE") == 2
    assert "the language the participants use" in rendered
    assert "{language_rule}" not in rendered


async def test_rendering_injects_the_named_language() -> None:
    rendered = await _render_decision_prompt(output_language=OutputLanguage.GERMAN)

    assert rendered.count("CRITICAL LANGUAGE RULE") == 2
    assert "Write ALL output fields in German." in rendered
    assert "the language the participants use" not in rendered


async def _render_decision_prompt(**kwargs: object) -> str:
    """Capture what the extractor hands the LLM; the rule only exists after rendering."""
    captured: list[str] = []

    class Capture:
        async def chat(self, messages: list[LLMChatMessage], **_: object) -> ChatResponse:
            assert isinstance(messages[0].content, str)  # narrow for test
            captured.append(messages[0].content)
            raise _PromptCapturedError

    with pytest.raises(_PromptCapturedError):
        await DecisionExtractor(llm=Capture()).aextract(_memcell(), **kwargs)  # type: ignore[arg-type]
    return captured[0]


class _PromptCapturedError(Exception):
    """Ends the call once the prompt has been captured — no LLM response is needed."""


# ==========================================================================
# Parsing
# ==========================================================================


async def test_aextract_parses_wrapped_decision_payload() -> None:
    """Prose-wrapped JSON object with decisions array → list[Decision]."""
    inner = json.dumps({"decisions": [_decision_payload()]})
    llm_json = f"Here you go:\n```json\n{inner}\n```"
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    decisions = await DecisionExtractor(llm=fake).aextract(_memcell())

    assert len(decisions) == 1
    dc = decisions[0]
    assert dc.title == "Agent Runtime language choice"
    assert dc.decision == "Use Python for the core Agent runtime and Rust on device."
    assert dc.reason == "Python fits fast iteration; Rust fits stable device operation."
    assert dc.impact == "Device capabilities connect through APIs."
    assert dc.tags == ["architecture", "runtime"]
    assert dc.owner_id is None
    assert dc.timestamp == 1700000000000


async def test_aextract_owner_id_is_always_none() -> None:
    """LLM-emitted owner_id must not bind the DTO — EverOS fans out later."""
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content=json.dumps({"decisions": [_decision_payload(owner_id="u_alice")]}), model="fake")
        ]
    )

    decisions = await DecisionExtractor(llm=fake).aextract(_memcell())

    assert decisions[0].owner_id is None


async def test_aextract_returns_empty_list_when_llm_returns_empty_decisions() -> None:
    """Empty decisions array from LLM → return [] immediately (no retry)."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"decisions": []}', model="fake")])

    result = await DecisionExtractor(llm=fake).aextract(_memcell())

    assert result == []
    assert fake.call_count == 1


async def test_aextract_raises_on_bad_json() -> None:
    """Unparseable JSON → ValueError on first attempt (no internal retry)."""
    fake = FakeLLMClient(responses=[ChatResponse(content="not json", model="fake")])

    with pytest.raises(ValueError):
        await DecisionExtractor(llm=fake).aextract(_memcell())

    assert fake.call_count == 1


async def test_aextract_raises_when_decisions_key_missing() -> None:
    fake = FakeLLMClient(responses=[ChatResponse(content='{"foresights": []}', model="fake")])

    with pytest.raises(ValueError, match="decisions key missing"):
        await DecisionExtractor(llm=fake).aextract(_memcell())


async def test_aextract_raises_when_decisions_is_not_a_list() -> None:
    fake = FakeLLMClient(responses=[ChatResponse(content='{"decisions": "nope"}', model="fake")])

    with pytest.raises(ValueError, match="decisions must be a list"):
        await DecisionExtractor(llm=fake).aextract(_memcell())


async def test_aextract_skips_invalid_items() -> None:
    """Items missing title / decision / reason are filtered; a valid sibling is kept."""
    payload = {
        "decisions": [
            _decision_payload(),
            _decision_payload(title=""),
            _decision_payload(decision=""),
            _decision_payload(reason=""),
            "not-an-object",
        ]
    }
    fake = FakeLLMClient(responses=[ChatResponse(content=json.dumps(payload), model="fake")])

    decisions = await DecisionExtractor(llm=fake).aextract(_memcell())
    assert len(decisions) == 1
    assert decisions[0].title == "Agent Runtime language choice"


async def test_aextract_blank_impact_becomes_none_and_non_list_tags_become_empty() -> None:
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content=json.dumps({"decisions": [_decision_payload(impact="  ", tags="architecture")]}),
                model="fake",
            )
        ]
    )
    decisions = await DecisionExtractor(llm=fake).aextract(_memcell())
    assert decisions[0].impact is None
    assert decisions[0].tags == []


async def test_aextract_per_call_prompt_overrides_default() -> None:
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"decisions": []}', model="fake")

    fake = FakeLLMClient(handler=handler)
    custom = "CUSTOM DECISION conv={CONVERSATION_TEXT}"

    await DecisionExtractor(llm=fake).aextract(_memcell(), prompt=custom)

    assert captured["content"].startswith("CUSTOM DECISION")
    assert "Core Agent Runtime stays Python" in captured["content"]
    assert "{CONVERSATION_TEXT}" not in captured["content"]


# ==========================================================================
# Truncation at _DECISION_MAX_COUNT
# ==========================================================================


async def test_aextract_truncates_when_more_than_10_decisions() -> None:
    """LLM returns 12 decisions → truncated to 10."""
    items = [_decision_payload(title=f"d-{i}") for i in range(12)]
    fake = FakeLLMClient(responses=[ChatResponse(content=json.dumps({"decisions": items}), model="fake")])

    decisions = await DecisionExtractor(llm=fake).aextract(_memcell())
    assert len(decisions) == 10


# ==========================================================================
# _render_conversation skips empty content
# ==========================================================================


def test_render_conversation_skips_empty_content() -> None:
    """Messages with empty content are silently dropped."""
    cell = MemCell(
        items=[
            ChatMessage(id="m1", role="user", content="hi", timestamp=1, sender_id="u_alice", sender_name="Alice"),
            ChatMessage(id="m2", role="user", content="", timestamp=2, sender_id="u_bob", sender_name="Bob"),
        ],
        timestamp=2,
    )
    rendered = _render_conversation(cell)
    assert "Alice: hi" in rendered
    assert "Bob" not in rendered


# ==========================================================================
# Silent-skip contract — agent → user-memory pipeline
# ==========================================================================


async def test_aextract_silently_skips_non_chat_items() -> None:
    """DecisionExtractor must silently skip ToolCallRequest / ToolCallResult items.

    Locks the agent → user-memory pipeline contract: a MemCell with mixed items (ChatMessage +
    tool calls) must produce the same Decision list as a chat-only MemCell with the same ChatMessages.
    """
    llm_json = json.dumps({"decisions": [_decision_payload()]})

    chat_only_cell = MemCell(
        items=[
            ChatMessage(
                id="c1",
                role="user",
                content="Core Agent Runtime stays Python; device runtime stays Rust.",
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
                content="Core Agent Runtime stays Python; device runtime stays Rust.",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ToolCallRequest(
                tool_calls=[
                    ToolCall(id="tc1", function=ToolCallFunction(name="docs.search", arguments='{"q": "runtime"}'))
                ],
                timestamp=1700000001000,
                sender_id="assistant",
            ),
            ToolCallResult(
                tool_call_id="tc1",
                content="No extra context.",
                timestamp=1700000002000,
            ),
        ],
        timestamp=1700000002000,
    )

    fake_chat = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])
    fake_mixed = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    dc_chat = await DecisionExtractor(llm=fake_chat).aextract(chat_only_cell)
    dc_mixed = await DecisionExtractor(llm=fake_mixed).aextract(mixed_cell)

    assert len(dc_chat) == len(dc_mixed) == 1
    assert dc_chat[0].decision == dc_mixed[0].decision
    assert dc_chat[0].owner_id is None
    assert dc_mixed[0].owner_id is None
    assert dc_mixed[0].timestamp == mixed_cell.timestamp
