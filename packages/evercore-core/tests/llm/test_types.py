"""Tests for evercore.llm.types — ChatMessage / Usage / ChatResponse."""

import json

import pytest
from pydantic import ValidationError

from evercore.llm.types import ChatMessage, ChatResponse, Usage


def test_chat_message_minimum_required_fields() -> None:
    msg = ChatMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_chat_message_role_accepts_three_values() -> None:
    """Minimal set: system / user / assistant — tool / function are out of EPISODE scope."""
    for role in ("system", "user", "assistant"):
        msg = ChatMessage(role=role, content="x")
        assert msg.role == role


def test_chat_message_invalid_role_raises() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content="x")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ChatMessage(role="developer", content="x")  # type: ignore[arg-type]


def test_chat_message_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role="user")  # type: ignore[call-arg]


def test_chat_message_extra_fields_silently_ignored() -> None:
    """OpenAI payload may carry name / tool_call_id — drop them."""
    msg = ChatMessage.model_validate(
        {
            "role": "assistant",
            "content": "hi",
            "name": "ignored",
            "tool_call_id": "ignored",
        }
    )
    assert not hasattr(msg, "name")
    assert not hasattr(msg, "tool_call_id")


def test_chat_message_json_round_trip() -> None:
    msg = ChatMessage(role="user", content="hi")
    serialised = msg.model_dump_json()
    assert json.loads(serialised) == {"role": "user", "content": "hi"}
    assert ChatMessage.model_validate_json(serialised) == msg


def test_usage_default_fields_are_none() -> None:
    """Both fields default to None to distinguish 'missing' from 'zero tokens'."""
    u = Usage()
    assert u.prompt_tokens is None
    assert u.completion_tokens is None


def test_usage_explicit_values_round_trip() -> None:
    u = Usage(prompt_tokens=12, completion_tokens=4)
    rebuilt = Usage.model_validate_json(u.model_dump_json())
    assert rebuilt == u
    assert rebuilt.prompt_tokens == 12
    assert rebuilt.completion_tokens == 4


def test_usage_partial_values_allowed() -> None:
    """Provider may report only one side of usage; the other stays None."""
    u = Usage(prompt_tokens=42)
    assert u.prompt_tokens == 42
    assert u.completion_tokens is None


def test_chat_response_minimum_required_fields() -> None:
    resp = ChatResponse(content="hello", model="gpt-4o-mini")
    assert resp.content == "hello"
    assert resp.model == "gpt-4o-mini"
    assert resp.usage is None
    assert resp.finish_reason is None
    assert resp.raw is None


def test_chat_response_with_usage_and_finish_reason() -> None:
    resp = ChatResponse(
        content="ok",
        model="gpt-4o-mini",
        usage=Usage(prompt_tokens=5, completion_tokens=3),
        finish_reason="stop",
    )
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 5
    assert resp.finish_reason == "stop"


def test_chat_response_finish_reason_three_values() -> None:
    """finish_reason is a Literal of stop / length / content_filter."""
    for reason in ("stop", "length", "content_filter"):
        resp = ChatResponse(content="x", model="m", finish_reason=reason)
        assert resp.finish_reason == reason


def test_chat_response_invalid_finish_reason_raises() -> None:
    with pytest.raises(ValidationError):
        ChatResponse(content="x", model="m", finish_reason="tool_calls")  # type: ignore[arg-type]


def test_chat_response_missing_content_raises() -> None:
    with pytest.raises(ValidationError):
        ChatResponse(model="m")  # type: ignore[call-arg]


def test_chat_response_raw_optional_dict() -> None:
    resp = ChatResponse(content="x", model="m", raw={"id": "chatcmpl-xyz"})
    assert resp.raw == {"id": "chatcmpl-xyz"}


def test_chat_response_json_round_trip_with_nested_usage() -> None:
    resp = ChatResponse(
        content="x",
        model="m",
        usage=Usage(prompt_tokens=10),
        finish_reason="length",
    )
    rebuilt = ChatResponse.model_validate_json(resp.model_dump_json())
    assert rebuilt == resp
