"""Tests for evercore.boundary.chat — ChatMemCellExtractor."""

from __future__ import annotations

from typing import Any

from evercore.boundary.chat import ChatMemCellExtractor
from evercore.llm.types import ChatMessage as LLMChatMessage
from evercore.llm.types import ChatResponse
from evercore.testing.fake_llm import FakeLLMClient
from evercore.types import Message, MessageRole


def _user(content: str, ts: int = 1700000000000) -> Message:
    """Helper: build a user-role Message."""
    return Message(role=MessageRole.USER, content=content, timestamp=ts)


def _assistant(content: str, ts: int = 1700000001000) -> Message:
    """Helper: build an assistant-role Message."""
    return Message(role=MessageRole.ASSISTANT, content=content, timestamp=ts)


async def test_adetect_returns_single_memcell_when_llm_returns_no_split() -> None:
    """split_at=null in LLM response yields a single coherent MemCell."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"split_at": null}', model="fake")])
    msgs = [_user("hello"), _assistant("hi there")]

    memcells = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert len(memcells) == 1
    assert memcells[0].messages == msgs


async def test_adetect_returns_two_memcells_when_llm_returns_split_index() -> None:
    """split_at=2 yields two MemCells: messages[:2] and messages[2:]."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"split_at": 2}', model="fake")])
    msgs = [
        _user("topic A part 1"),
        _assistant("topic A part 2"),
        _user("now topic B"),
        _assistant("topic B reply"),
    ]

    memcells = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert len(memcells) == 2
    assert memcells[0].messages == msgs[:2]
    assert memcells[1].messages == msgs[2:]


async def test_adetect_per_call_prompt_overrides_default() -> None:
    """Per-call prompt= argument is the rendered prompt, not the default."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"split_at": null}', model="fake")

    fake = FakeLLMClient(handler=handler)
    custom_prompt = "CUSTOM PROMPT messages={messages} tokens={token_count}"

    await ChatMemCellExtractor().adetect([_user("hi")], llm=fake, prompt=custom_prompt)

    assert captured["content"].startswith("CUSTOM PROMPT")
    assert "[user] hi" in captured["content"]
