"""Tests for everalgo.user_memory.atomic_fact — AtomicFactExtractor."""

from __future__ import annotations

from typing import Any

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import MemCell, Message, MessageRole
from everalgo.user_memory.atomic_fact import AtomicFactExtractor


def _memcell() -> MemCell:
    """Helper: build a minimal MemCell with verifiable assertions."""
    return MemCell(
        id="mc_test_001",
        messages=[
            Message(
                role=MessageRole.USER,
                content="Alice scheduled a 3pm meeting with Bob on 2024-03-14.",
                timestamp=1700000000000,
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="Confirmed. Meeting is on the calendar.",
                timestamp=1700000001000,
            ),
        ],
        timestamp=1700000001000,
    )


async def test_aextract_returns_atomic_fact_list_from_llm_json() -> None:
    """Valid LLM JSON yields a list[AtomicFact] with all fields populated."""
    llm_json = (
        '{"atomic_facts": ['
        '{"id": "af_001", "owner_id": "u_alice", '
        '"fact": "Alice scheduled a 3pm meeting with Bob on 2024-03-14.", '
        '"timestamp": 1700000000000},'
        '{"id": "af_002", "owner_id": "u_alice", '
        '"fact": "The meeting was placed on the calendar.", '
        '"timestamp": 1700000001000}'
        "]}"
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    facts = await AtomicFactExtractor().aextract(_memcell(), llm=fake)

    assert len(facts) == 2
    af = facts[0]
    assert af.id == "af_001"
    assert af.owner_id == "u_alice"
    assert "Alice" in af.fact
    assert af.timestamp == 1700000000000


async def test_aextract_auto_fills_parent_id_from_memcell() -> None:
    """LLM-emitted JSON without parent_id gets parent_id from the source MemCell."""
    llm_json = '{"atomic_facts": [{"id": "af_x", "owner_id": "u_x", "fact": "x", "timestamp": 1700000000000}]}'
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])
    mc = _memcell()

    facts = await AtomicFactExtractor().aextract(mc, llm=fake)

    assert facts[0].parent_id == mc.id
    assert facts[0].parent_type == "memcell"


async def test_aextract_returns_empty_list_when_no_facts() -> None:
    """LLM JSON ``{"atomic_facts": []}`` yields an empty list, not an error."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"atomic_facts": []}', model="fake")])

    facts = await AtomicFactExtractor().aextract(_memcell(), llm=fake)

    assert facts == []


async def test_aextract_per_call_prompt_overrides_default() -> None:
    """Per-call prompt= argument is the rendered prompt, not the default."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"atomic_facts": []}', model="fake")

    fake = FakeLLMClient(handler=handler)
    custom_prompt = "CUSTOM ATOMIC PROMPT conv={memcell_text} ts={timestamp}"

    await AtomicFactExtractor().aextract(_memcell(), llm=fake, prompt=custom_prompt)

    assert captured["content"].startswith("CUSTOM ATOMIC PROMPT")
    assert "[user] Alice scheduled a 3pm meeting with Bob on 2024-03-14." in captured["content"]


async def test_aextract_per_call_llm_overrides_default() -> None:
    """Per-call llm= argument is the one used by the extractor."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["called"] = True
        return ChatResponse(
            content=('{"atomic_facts": [{"id": "af_y", "owner_id": "u_y", "fact": "y", "timestamp": 1700000000000}]}'),
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)

    await AtomicFactExtractor().aextract(_memcell(), llm=fake)

    assert captured["called"] is True
