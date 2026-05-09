"""Tests for evercore.testing.fake_llm — CallRecord."""

from typing import Any

import pytest

from evercore.llm.types import ChatMessage, ChatResponse
from evercore.testing.fake_llm import CallRecord, FakeLLMClient


def test_call_record_minimum_fields() -> None:
    """Messages is the only required field."""
    record = CallRecord(messages=[ChatMessage(role="user", content="hi")])
    assert record.messages == [ChatMessage(role="user", content="hi")]
    assert record.model is None
    assert record.temperature is None
    assert record.max_tokens is None
    assert record.response_format is None
    assert record.extra == {}


def test_call_record_all_fields_populated() -> None:
    """All optional fields can be set explicitly."""
    record = CallRecord(
        messages=[ChatMessage(role="user", content="hi")],
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=128,
        response_format={"type": "json_object"},
        extra={"seed": 42},
    )
    assert record.model == "gpt-4o-mini"
    assert record.temperature == 0.7
    assert record.max_tokens == 128
    assert record.response_format == {"type": "json_object"}
    assert record.extra == {"seed": 42}


def test_call_record_extra_field_is_independent_per_instance() -> None:
    """Each instance must own its own extra dict (not shared across instances)."""
    a = CallRecord(messages=[ChatMessage(role="user", content="a")])
    b = CallRecord(messages=[ChatMessage(role="user", content="b")])
    a.extra["key"] = "value"
    assert b.extra == {}


# ---- FakeLLMClient construction (Task 2) ----------------------------------


def test_fake_llm_client_constructed_with_responses_only() -> None:
    """Scripted list mode is one valid construction path."""
    client = FakeLLMClient(responses=["hello"])
    assert client.call_count == 0


def test_fake_llm_client_constructed_with_handler_only() -> None:
    """Callable handler mode is the other valid construction path."""

    def handler(messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(content="ok", model="fake")

    client = FakeLLMClient(handler=handler)
    assert client.call_count == 0


def test_fake_llm_client_both_responses_and_handler_raises() -> None:
    """Mutual exclusion: passing both is a ValueError."""

    def handler(messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(content="ok", model="fake")

    with pytest.raises(ValueError, match="exactly one of"):
        FakeLLMClient(responses=["hi"], handler=handler)


def test_fake_llm_client_neither_responses_nor_handler_raises() -> None:
    """Mutual exclusion: passing neither is a ValueError."""
    with pytest.raises(ValueError, match="exactly one of"):
        FakeLLMClient()


def test_fake_llm_client_responses_invalid_element_type_raises() -> None:
    """Each responses element must be str or ChatResponse."""
    with pytest.raises(TypeError, match="must contain str or ChatResponse"):
        FakeLLMClient(responses=[123])  # type: ignore[list-item]


# ---- FakeLLMClient.chat scripted-list mode (Task 3) -----------------------


async def test_chat_str_element_wrapped_to_default_chat_response() -> None:
    """Str element gets auto-wrapped: model='fake', finish_reason='stop'."""
    client = FakeLLMClient(responses=["hello"])
    response = await client.chat(messages=[ChatMessage(role="user", content="hi")])
    assert response.content == "hello"
    assert response.model == "fake"
    assert response.usage is None
    assert response.finish_reason == "stop"
    assert response.raw is None


async def test_chat_chat_response_element_passed_through_unchanged() -> None:
    """ChatResponse instance returned as-is, preserving usage/finish_reason."""
    from evercore.llm.types import Usage

    canned = ChatResponse(
        content="canned",
        model="custom-model",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        finish_reason="length",
    )
    client = FakeLLMClient(responses=[canned])
    response = await client.chat(messages=[ChatMessage(role="user", content="hi")])
    assert response is canned


async def test_chat_responses_popped_in_call_order() -> None:
    """Multiple responses returned in FIFO order."""
    client = FakeLLMClient(responses=["first", "second", "third"])
    msgs = [ChatMessage(role="user", content="hi")]
    r1 = await client.chat(messages=msgs)
    r2 = await client.chat(messages=msgs)
    r3 = await client.chat(messages=msgs)
    assert (r1.content, r2.content, r3.content) == ("first", "second", "third")


async def test_chat_exhausted_script_raises_runtime_error() -> None:
    """N+1th call raises RuntimeError with `(used N of N)` message."""
    client = FakeLLMClient(responses=["only"])
    msgs = [ChatMessage(role="user", content="hi")]
    await client.chat(messages=msgs)  # exhaust the script
    with pytest.raises(RuntimeError, match=r"script exhausted.*used 1 of 1"):
        await client.chat(messages=msgs)


# ---- FakeLLMClient.chat callable handler mode (Task 4) --------------------


async def test_chat_sync_handler_invoked_correctly() -> None:
    """Sync handler returning ChatResponse is dispatched normally."""

    def handler(messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(content="sync-out", model="fake-sync")

    client = FakeLLMClient(handler=handler)
    response = await client.chat(messages=[ChatMessage(role="user", content="hi")])
    assert response.content == "sync-out"
    assert response.model == "fake-sync"


async def test_chat_async_handler_awaited_correctly() -> None:
    """Async handler is awaited; result is the resolved ChatResponse."""

    async def handler(messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(content="async-out", model="fake-async")

    client = FakeLLMClient(handler=handler)
    response = await client.chat(messages=[ChatMessage(role="user", content="hi")])
    assert response.content == "async-out"
    assert response.model == "fake-async"


async def test_chat_handler_receives_messages_and_kwargs() -> None:
    """Handler sees messages + model + temperature + max_tokens + extras."""
    captured: dict[str, Any] = {}

    def handler(messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return ChatResponse(content="x", model="fake")

    client = FakeLLMClient(handler=handler)
    msgs = [ChatMessage(role="user", content="hi")]
    await client.chat(
        messages=msgs,
        model="gpt-4o-mini",
        temperature=0.5,
        max_tokens=64,
        response_format={"type": "json_object"},
        seed=42,
    )
    assert captured["messages"] == msgs
    assert captured["kwargs"]["model"] == "gpt-4o-mini"
    assert captured["kwargs"]["temperature"] == 0.5
    assert captured["kwargs"]["max_tokens"] == 64
    assert captured["kwargs"]["response_format"] == {"type": "json_object"}
    assert captured["kwargs"]["seed"] == 42


async def test_chat_handler_wrong_return_type_raises() -> None:
    """Handler returning non-ChatResponse, non-Awaitable raises TypeError."""

    def handler(messages: list[ChatMessage], **kwargs: Any) -> Any:
        return {"content": "wrong"}  # not ChatResponse

    client = FakeLLMClient(handler=handler)
    with pytest.raises(TypeError, match=r"must return ChatResponse.*got dict"):
        await client.chat(messages=[ChatMessage(role="user", content="hi")])


# ---- FakeLLMClient call recording + Protocol (Task 5) ---------------------

from evercore.llm.protocols import LLMClient  # noqa: E402


async def test_call_count_increments_per_invocation() -> None:
    """Call count tracks every chat() invocation, regardless of mode."""
    client = FakeLLMClient(responses=["a", "b"])
    assert client.call_count == 0
    msgs = [ChatMessage(role="user", content="hi")]
    await client.chat(messages=msgs)
    assert client.call_count == 1
    await client.chat(messages=msgs)
    assert client.call_count == 2


async def test_calls_property_records_messages_and_kwargs() -> None:
    """Calls property captures messages + each kwarg + extras into CallRecord."""
    client = FakeLLMClient(responses=["a"])
    msgs = [ChatMessage(role="user", content="hi")]
    await client.chat(
        messages=msgs,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=128,
        response_format={"type": "json_object"},
        seed=99,
    )
    assert len(client.calls) == 1
    record = client.calls[0]
    assert isinstance(record, CallRecord)
    assert record.messages == msgs
    assert record.model == "gpt-4o-mini"
    assert record.temperature == 0.7
    assert record.max_tokens == 128
    assert record.response_format == {"type": "json_object"}
    assert record.extra == {"seed": 99}


async def test_calls_property_returns_defensive_copy() -> None:
    """Mutating the returned list must not affect internal state."""
    client = FakeLLMClient(responses=["a"])
    msgs = [ChatMessage(role="user", content="hi")]
    await client.chat(messages=msgs)
    snapshot = client.calls
    snapshot.clear()
    assert client.call_count == 1
    assert len(client.calls) == 1


def test_fake_llm_client_satisfies_LLMClient_protocol() -> None:
    """Isinstance check works thanks to @runtime_checkable on LLMClient."""
    client = FakeLLMClient(responses=["a"])
    assert isinstance(client, LLMClient)
