"""Tests for everalgo.types.memcell — MessageRole + Message."""

import json

import pytest
from pydantic import ValidationError

from everalgo.types import Message, MessageRole


def test_message_role_enum_values_match_opensource() -> None:
    """Mirrors opensource ``MessageSenderRole``: USER / ASSISTANT / TOOL."""
    assert {r.value for r in MessageRole} == {"user", "assistant", "tool"}


def test_message_role_str_inheritance() -> None:
    """MessageRole is a str enum so it serialises directly to its value."""
    assert MessageRole.USER == "user"  # type: ignore[comparison-overlap]
    assert MessageRole.ASSISTANT == "assistant"  # type: ignore[comparison-overlap]
    assert MessageRole.TOOL == "tool"  # type: ignore[comparison-overlap]


def test_message_minimum_required_fields() -> None:
    msg = Message(role=MessageRole.USER, content="hello", timestamp=1700000000000)
    assert msg.role == MessageRole.USER
    assert msg.content == "hello"
    assert msg.timestamp == 1700000000000


def test_message_role_accepts_string_value() -> None:
    msg = Message(role="assistant", content="hi", timestamp=1)  # type: ignore[arg-type]
    assert msg.role == MessageRole.ASSISTANT


def test_message_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        Message(role=MessageRole.USER, content="hello")  # type: ignore[call-arg]


def test_message_invalid_role_raises() -> None:
    with pytest.raises(ValidationError):
        Message(role="system", content="hello", timestamp=1)  # type: ignore[arg-type]


def test_message_sender_id_sender_name_refer_list_are_first_class_fields() -> None:
    """``sender_id`` / ``sender_name`` / ``refer_list`` map to opensource ``MessageItem`` keys."""
    msg = Message.model_validate(
        {
            "role": "user",
            "content": "hi",
            "timestamp": 1,
            "sender_id": "u1",
            "sender_name": "Alice",
            "refer_list": [{"_id": "u2", "name": "Bob"}],
            "tool_calls": [{"id": "x"}],
            "message_id": "m1",
        }
    )
    assert msg.sender_id == "u1"
    assert msg.sender_name == "Alice"
    assert msg.refer_list == [{"_id": "u2", "name": "Bob"}]
    assert not hasattr(msg, "tool_calls")
    assert not hasattr(msg, "message_id")


def test_message_extra_fields_dropped_when_unrecognized() -> None:
    msg = Message.model_validate(
        {
            "role": "user",
            "content": "hi",
            "timestamp": 1,
            "tool_calls": [{"id": "x"}],
            "message_id": "m1",
        }
    )
    assert not hasattr(msg, "tool_calls")
    assert not hasattr(msg, "message_id")


def test_message_json_round_trip() -> None:
    msg = Message(role=MessageRole.USER, content="hi", timestamp=42)
    serialised = msg.model_dump_json()
    assert json.loads(serialised) == {
        "role": "user",
        "content": "hi",
        "timestamp": 42,
        "sender_id": None,
        "sender_name": None,
        "refer_list": [],
    }
    assert Message.model_validate_json(serialised) == msg


def test_message_json_round_trip_with_opensource_fields() -> None:
    msg = Message(
        role=MessageRole.USER,
        content="hi",
        timestamp=42,
        sender_id="u_a",
        sender_name="Alice",
        refer_list=[{"_id": "u_b"}],
    )
    serialised = msg.model_dump_json()
    parsed = json.loads(serialised)
    assert parsed["sender_id"] == "u_a"
    assert parsed["sender_name"] == "Alice"
    assert parsed["refer_list"] == [{"_id": "u_b"}]
    assert Message.model_validate_json(serialised) == msg
