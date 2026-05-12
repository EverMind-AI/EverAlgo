"""Tests for everalgo.user_memory.foresight — ForesightExtractor."""

from __future__ import annotations

from typing import Any

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import MemCell, Message, MessageRole
from everalgo.user_memory.foresight import ForesightExtractor


def _memcell() -> MemCell:
    """Helper: build a minimal MemCell with forward-looking dialogue."""
    return MemCell(
        id="mc_test_001",
        messages=[
            Message(
                role=MessageRole.USER,
                content="I'll send Alice the draft by Friday.",
                timestamp=1700000000000,
            ),
            Message(
                role=MessageRole.ASSISTANT,
                content="Got it. I'll follow up next week to confirm review.",
                timestamp=1700000001000,
            ),
        ],
        timestamp=1700000001000,
    )


async def test_aextract_returns_foresight_list_from_llm_json() -> None:
    """Valid LLM JSON yields a list[Foresight] with all fields populated."""
    llm_json = (
        '{"foresights": ['
        '{"id": "fs_001", "owner_id": "u_alice", '
        '"foresight": "User will send Alice the draft by Friday.", '
        '"evidence": "\\"I\'ll send Alice the draft by Friday.\\"", '
        '"timestamp": 1700000000000},'
        '{"id": "fs_002", "owner_id": "u_alice", '
        '"foresight": "Assistant will follow up next week.", '
        '"evidence": "\\"I\'ll follow up next week to confirm review.\\"", '
        '"timestamp": 1700000001000}'
        "]}"
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    foresights = await ForesightExtractor().aextract(_memcell(), llm=fake)

    assert len(foresights) == 2
    fs = foresights[0]
    assert fs.id == "fs_001"
    assert fs.owner_id == "u_alice"
    assert "Friday" in fs.foresight
    assert "draft" in fs.evidence
    assert fs.timestamp == 1700000000000


async def test_aextract_auto_fills_parent_id_from_memcell() -> None:
    """LLM-emitted JSON without parent_id gets parent_id from the source MemCell."""
    llm_json = (
        '{"foresights": [{"id": "fs_x", "owner_id": "u_x", '
        '"foresight": "x", "evidence": "x", "timestamp": 1700000000000}]}'
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])
    mc = _memcell()

    foresights = await ForesightExtractor().aextract(mc, llm=fake)

    assert foresights[0].parent_id == mc.id
    assert foresights[0].parent_type == "memcell"


async def test_aextract_returns_empty_list_when_no_foresights() -> None:
    """LLM JSON ``{"foresights": []}`` yields an empty list, not an error."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"foresights": []}', model="fake")])

    foresights = await ForesightExtractor().aextract(_memcell(), llm=fake)

    assert foresights == []


async def test_aextract_per_call_prompt_overrides_default() -> None:
    """Per-call prompt= argument is the rendered prompt, not the default."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"foresights": []}', model="fake")

    fake = FakeLLMClient(handler=handler)
    custom_prompt = "CUSTOM FORESIGHT PROMPT conv={memcell_text} ts={timestamp}"

    await ForesightExtractor().aextract(_memcell(), llm=fake, prompt=custom_prompt)

    assert captured["content"].startswith("CUSTOM FORESIGHT PROMPT")
    assert "[user] I'll send Alice the draft by Friday." in captured["content"]


async def test_aextract_per_call_llm_overrides_default() -> None:
    """Per-call llm= argument is the one used by the extractor."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["called"] = True
        return ChatResponse(
            content=(
                '{"foresights": [{"id": "fs_y", "owner_id": "u_y", '
                '"foresight": "y", "evidence": "y", "timestamp": 1700000000000}]}'
            ),
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)

    await ForesightExtractor().aextract(_memcell(), llm=fake)

    assert captured["called"] is True
