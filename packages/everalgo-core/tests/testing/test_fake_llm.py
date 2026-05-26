"""Tests for everalgo.testing.fake_llm — CallRecord."""

from typing import Any, cast

import pytest
from pydantic import BaseModel

from everalgo.llm.types import ChatMessage, ChatResponse
from everalgo.testing.fake_llm import CallRecord, FakeLLMClient


def test_call_record_minimum_fields() -> None:
    """Messages is the only required field."""
    record = CallRecord(messages=[ChatMessage(role="user", content="hi")])
    assert record.messages == [ChatMessage(role="user", content="hi")]
    assert record.model is None
    assert record.temperature is None
    assert record.max_tokens is None
    assert record.response_format is None
    assert record.extra == {}


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
    from everalgo.llm.types import Usage

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
        seed=42,
    )
    assert captured["messages"] == msgs
    assert captured["kwargs"]["model"] == "gpt-4o-mini"
    assert captured["kwargs"]["temperature"] == 0.5
    assert captured["kwargs"]["max_tokens"] == 64
    assert captured["kwargs"]["seed"] == 42


async def test_chat_handler_wrong_return_type_raises() -> None:
    """Handler returning non-ChatResponse, non-Awaitable raises TypeError."""

    def handler(messages: list[ChatMessage], **kwargs: Any) -> Any:
        return {"content": "wrong"}  # not ChatResponse

    client = FakeLLMClient(handler=handler)
    with pytest.raises(TypeError, match=r"must return ChatResponse.*got dict"):
        await client.chat(messages=[ChatMessage(role="user", content="hi")])


# ---- FakeLLMClient call recording + Protocol (Task 5) ---------------------

from everalgo.llm.protocols import LLMClient  # noqa: E402


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
        seed=99,
    )
    assert len(client.calls) == 1
    record = client.calls[0]
    assert isinstance(record, CallRecord)
    assert record.messages == msgs
    assert record.model == "gpt-4o-mini"
    assert record.temperature == 0.7
    assert record.max_tokens == 128
    assert record.response_format is None
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


def test_fake_llm_client_satisfies_llm_client_protocol() -> None:
    """Isinstance check works thanks to @runtime_checkable on LLMClient."""
    client = FakeLLMClient(responses=["a"])
    assert isinstance(client, LLMClient)


# ---- FakeLLMClient BaseModel response_format (T4 — parse parity) ----------


class _Tag(BaseModel):
    """Minimal schema for BaseModel response_format tests."""

    label: str


async def test_basemodel_response_format_constructs_parsed_from_content() -> None:
    """When response_format is a BaseModel subclass, parsed is populated from content JSON."""
    client = FakeLLMClient(responses=['{"label": "news"}'])
    resp = await client.chat(
        messages=[ChatMessage(role="user", content="hi")],
        response_format=_Tag,
    )
    assert resp.parsed is not None
    assert isinstance(resp.parsed, _Tag)
    assert resp.parsed.label == "news"
    assert resp.content == '{"label": "news"}'


async def test_basemodel_response_format_records_schema_class_in_call_record() -> None:
    """CallRecord.response_format stores the BaseModel class itself (not a dict)."""
    client = FakeLLMClient(responses=['{"label": "x"}'])
    await client.chat(
        messages=[ChatMessage(role="user", content="hi")],
        response_format=_Tag,
    )
    record = client.calls[0]
    assert record.response_format is _Tag


async def test_basemodel_response_format_invalid_json_raises_llm_error() -> None:
    """Content that is not valid JSON for the schema must raise LLMError."""
    from everalgo.llm.errors import LLMError

    client = FakeLLMClient(responses=["not-valid-json"])
    with pytest.raises(LLMError):
        await client.chat(
            messages=[ChatMessage(role="user", content="hi")],
            response_format=_Tag,
        )


async def test_basemodel_response_format_wrong_schema_raises_llm_error() -> None:
    """Valid JSON that does not match the schema must raise LLMError."""
    from everalgo.llm.errors import LLMError

    client = FakeLLMClient(responses=['{"wrong_field": 123}'])
    with pytest.raises(LLMError):
        await client.chat(
            messages=[ChatMessage(role="user", content="hi")],
            response_format=_Tag,
        )


# ---- FakeLLMClient handler path BaseModel parity (fix for missing _attach_parsed) ----


async def test_fake_llm_handler_path_attaches_parsed_for_basemodel_response_format() -> None:
    """Handler mode + BaseModel response_format must auto-populate ChatResponse.parsed.

    T4 only wired _attach_parsed for the scripted-list path.  The handler path was
    left without it, causing response.parsed to remain None for any caller that used
    FakeLLMClient(handler=...) with response_format=SomeModel.  This test pins the
    corrected behaviour: if the handler returns a ChatResponse whose .parsed is None,
    FakeLLMClient must derive parsed by deserialising .content against the schema.
    """
    from typing import Any

    def handler(messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(content='{"label": "handler-tag"}', model="fake")

    client = FakeLLMClient(handler=handler)
    resp = await client.chat(
        messages=[ChatMessage(role="user", content="hi")],
        response_format=_Tag,
    )
    assert resp.parsed is not None
    assert isinstance(resp.parsed, _Tag)
    assert resp.parsed.label == "handler-tag"
    assert resp.content == '{"label": "handler-tag"}'


async def test_fake_llm_handler_path_preserves_explicit_parsed_override() -> None:
    """Handler that already sets parsed must not be overwritten by _attach_parsed.

    This validates the skip-if-already-set contract: if the handler explicitly
    returns ChatResponse(parsed=X), FakeLLMClient must honour that and not
    re-derive parsed from content.
    """
    from typing import Any

    explicit_tag = _Tag(label="explicit")

    def handler(messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        return ChatResponse(content='{"label": "content-tag"}', model="fake", parsed=explicit_tag)

    client = FakeLLMClient(handler=handler)
    resp = await client.chat(
        messages=[ChatMessage(role="user", content="hi")],
        response_format=_Tag,
    )
    assert resp.parsed is explicit_tag
    assert cast("_Tag", resp.parsed).label == "explicit"  # type: ignore[redundant-cast]
