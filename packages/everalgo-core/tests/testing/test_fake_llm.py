"""Tests for everalgo.testing.fake_llm — FakeLLMClient and CallRecord."""

from typing import Any

import pytest

from everalgo.llm import StrictOutputModel
from everalgo.llm.types import LLMMessage
from everalgo.testing.fake_llm import CallRecord, FakeLLMClient


class _Label(StrictOutputModel):
    label: str


class _HostError(Exception):
    """Stands in for an implementation's own error type, which the contract leaves to the implementation."""


def _msgs(content: str = "hi") -> list[LLMMessage]:
    return [LLMMessage(role="user", content=content)]


def test_call_record_minimum_fields() -> None:
    record = CallRecord(messages=_msgs(), response_format=_Label)
    assert record.messages == _msgs()
    assert record.response_format is _Label
    assert record.temperature is None
    assert record.max_tokens is None
    assert record.extra == {}


def test_call_record_extra_field_is_independent_per_instance() -> None:
    a = CallRecord(messages=_msgs("a"), response_format=_Label)
    b = CallRecord(messages=_msgs("b"), response_format=_Label)
    a.extra["key"] = "value"
    assert b.extra == {}


# ---- FakeLLMClient construction ------------------------------------------


def test_fake_llm_client_constructed_with_responses_only() -> None:
    client = FakeLLMClient(responses=["hello"])
    assert client.call_count == 0


def test_fake_llm_client_constructed_with_handler_only() -> None:
    def handler(messages: list[LLMMessage], **kwargs: Any) -> str:
        return "ok"

    client = FakeLLMClient(handler=handler)
    assert client.call_count == 0


def test_fake_llm_client_both_responses_and_handler_raises() -> None:
    def handler(messages: list[LLMMessage], **kwargs: Any) -> str:
        return "ok"

    with pytest.raises(ValueError, match="exactly one of"):
        FakeLLMClient(responses=["hi"], handler=handler)


def test_fake_llm_client_neither_responses_nor_handler_raises() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        FakeLLMClient()


def test_fake_llm_client_responses_invalid_element_type_raises() -> None:
    with pytest.raises(TypeError, match="must contain str or Exception"):
        FakeLLMClient(responses=[123])  # type: ignore[list-item]  # invalid element on purpose


# ---- FakeLLMClient.achat scripted-list mode -------------------------------


async def test_achat_returns_the_scripted_text() -> None:
    client = FakeLLMClient(responses=['{"label": "news"}'])
    reply = await client.achat(messages=_msgs(), response_format=_Label)
    assert reply == '{"label": "news"}'


async def test_achat_returns_text_even_when_it_does_not_match_the_model() -> None:
    """The double honours no response_format, so an operator test can script a malformed reply."""
    client = FakeLLMClient(responses=["not json"])
    assert await client.achat(messages=_msgs(), response_format=_Label) == "not json"


async def test_achat_responses_popped_in_call_order() -> None:
    client = FakeLLMClient(responses=["first", "second", "third"])
    r1 = await client.achat(messages=_msgs(), response_format=_Label)
    r2 = await client.achat(messages=_msgs(), response_format=_Label)
    r3 = await client.achat(messages=_msgs(), response_format=_Label)
    assert (r1, r2, r3) == ("first", "second", "third")


async def test_achat_exhausted_script_raises_runtime_error() -> None:
    client = FakeLLMClient(responses=["only"])
    await client.achat(messages=_msgs(), response_format=_Label)
    with pytest.raises(RuntimeError, match=r"script exhausted.*used 1 of 1"):
        await client.achat(messages=_msgs(), response_format=_Label)


async def test_achat_without_response_format_raises_type_error() -> None:
    """The double enforces the mandatory parameter exactly as the real provider does."""
    client = FakeLLMClient(responses=["only"])
    with pytest.raises(TypeError, match="response_format"):
        await client.achat(messages=_msgs())  # type: ignore[call-arg]  # response_format omitted on purpose


async def test_scripted_exception_is_raised_unchanged() -> None:
    """An implementation-defined error must reach the operator as the very object, so its type survives."""
    error = _HostError("upstream refused")
    client = FakeLLMClient(responses=[error])

    with pytest.raises(_HostError) as caught:
        await client.achat(messages=_msgs(), response_format=_Label)

    assert caught.value is error


async def test_scripted_exception_is_recorded_like_any_other_call() -> None:
    """The failing call is still recorded, so a test can assert what was sent before the failure."""
    client = FakeLLMClient(responses=[_HostError("nope"), "second"])

    with pytest.raises(_HostError):
        await client.achat(messages=_msgs("first"), response_format=_Label)
    reply = await client.achat(messages=_msgs("second"), response_format=_Label)

    assert client.call_count == 2
    assert client.calls[0].messages == _msgs("first")
    assert reply == "second"


# ---- FakeLLMClient.achat callable handler mode ----------------------------


async def test_achat_sync_handler_invoked_correctly() -> None:
    def handler(messages: list[LLMMessage], **kwargs: Any) -> str:
        return "sync-out"

    client = FakeLLMClient(handler=handler)
    assert await client.achat(messages=_msgs(), response_format=_Label) == "sync-out"


async def test_achat_async_handler_awaited_correctly() -> None:
    async def handler(messages: list[LLMMessage], **kwargs: Any) -> str:
        return "async-out"

    client = FakeLLMClient(handler=handler)
    assert await client.achat(messages=_msgs(), response_format=_Label) == "async-out"


async def test_achat_handler_receives_messages_and_kwargs() -> None:
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMMessage], **kwargs: Any) -> str:
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "x"

    client = FakeLLMClient(handler=handler)
    msgs = _msgs()
    await client.achat(messages=msgs, response_format=_Label, temperature=0.5, max_tokens=64, seed=42)
    assert captured["messages"] == msgs
    assert captured["kwargs"]["response_format"] is _Label
    assert captured["kwargs"]["temperature"] == 0.5
    assert captured["kwargs"]["max_tokens"] == 64
    assert captured["kwargs"]["seed"] == 42


async def test_achat_handler_wrong_return_type_raises() -> None:
    def handler(messages: list[LLMMessage], **kwargs: Any) -> Any:
        return {"content": "wrong"}

    client = FakeLLMClient(handler=handler)
    with pytest.raises(TypeError, match=r"must return str.*got dict"):
        await client.achat(messages=_msgs(), response_format=_Label)


# ---- FakeLLMClient call recording ----------------------------------------


async def test_call_count_increments_per_invocation() -> None:
    client = FakeLLMClient(responses=["a", "b"])
    assert client.call_count == 0
    await client.achat(messages=_msgs(), response_format=_Label)
    assert client.call_count == 1
    await client.achat(messages=_msgs(), response_format=_Label)
    assert client.call_count == 2


async def test_calls_property_records_messages_and_kwargs() -> None:
    client = FakeLLMClient(responses=["a"])
    msgs = _msgs()
    await client.achat(messages=msgs, response_format=_Label, temperature=0.7, max_tokens=128, seed=99)
    assert len(client.calls) == 1
    record = client.calls[0]
    assert isinstance(record, CallRecord)
    assert record.messages == msgs
    assert record.response_format is _Label
    assert record.temperature == 0.7
    assert record.max_tokens == 128
    assert record.extra == {"seed": 99}


async def test_handler_path_records_the_call_too() -> None:
    def handler(messages: list[LLMMessage], **kwargs: Any) -> str:
        return '{"label": "handler-tag"}'

    client = FakeLLMClient(handler=handler)
    await client.achat(messages=_msgs(), response_format=_Label, max_tokens=32)

    assert client.calls[0].response_format is _Label
    assert client.calls[0].max_tokens == 32


async def test_a_model_override_lands_in_extra() -> None:
    """``model`` is not a formal parameter, so it is recorded as the escape-hatch key it is."""
    client = FakeLLMClient(responses=["a"])

    await client.achat(messages=_msgs(), response_format=_Label, model="gpt-4o-mini")

    assert client.calls[0].extra == {"model": "gpt-4o-mini"}


async def test_calls_property_returns_defensive_copy() -> None:
    client = FakeLLMClient(responses=["a"])
    await client.achat(messages=_msgs(), response_format=_Label)
    snapshot = client.calls
    snapshot.clear()
    assert client.call_count == 1
    assert len(client.calls) == 1
