"""Tests for everalgo.types.memcell — MessageRole + Message."""

import json

import pytest
from pydantic import ValidationError

from everalgo.types import Message, MessageRole


def test_message_role_enum_values_are_user_and_assistant() -> None:
    """Minimal set: USER + ASSISTANT only — TOOL / SYSTEM are out of EPISODE scope."""
    assert {r.value for r in MessageRole} == {"user", "assistant"}


def test_message_role_str_inheritance() -> None:
    """MessageRole should be a str enum so it serialises directly to its value."""
    assert MessageRole.USER == "user"  # type: ignore[comparison-overlap]
    assert MessageRole.ASSISTANT == "assistant"  # type: ignore[comparison-overlap]


def test_message_minimum_required_fields() -> None:
    msg = Message(role=MessageRole.USER, content="hello", timestamp=1700000000000)
    assert msg.role == MessageRole.USER
    assert msg.content == "hello"
    assert msg.timestamp == 1700000000000


def test_message_role_accepts_string_value() -> None:
    """Pydantic should coerce raw role string to the enum."""
    msg = Message(role="assistant", content="hi", timestamp=1)  # type: ignore[arg-type]
    assert msg.role == MessageRole.ASSISTANT


def test_message_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        Message(role=MessageRole.USER, content="hello")  # type: ignore[call-arg]


def test_message_invalid_role_raises() -> None:
    """System and tool roles must be rejected outside the minimal set."""
    with pytest.raises(ValidationError):
        Message(role="system", content="hello", timestamp=1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Message(role="tool", content="hello", timestamp=1)  # type: ignore[arg-type]


def test_message_extra_fields_silently_ignored() -> None:
    """Opensource payload may carry sender_id / tool_calls / message_id — they should be dropped."""
    msg = Message.model_validate(
        {
            "role": "user",
            "content": "hi",
            "timestamp": 1,
            "sender_id": "u1",
            "sender_name": "Alice",
            "tool_calls": [{"id": "x"}],
            "message_id": "m1",
        }
    )
    assert msg.role == MessageRole.USER
    assert msg.content == "hi"
    assert msg.timestamp == 1
    assert not hasattr(msg, "sender_id")
    assert not hasattr(msg, "tool_calls")


def test_message_json_round_trip() -> None:
    msg = Message(role=MessageRole.USER, content="hi", timestamp=42)
    serialised = msg.model_dump_json()
    assert json.loads(serialised) == {"role": "user", "content": "hi", "timestamp": 42}
    assert Message.model_validate_json(serialised) == msg
