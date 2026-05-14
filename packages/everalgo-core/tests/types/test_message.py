"""Tests for everalgo.types.memcell — MessageRole + Message + ToolCall."""

import json

import pytest
from pydantic import ValidationError

from everalgo.types import Message, MessageRole, ToolCall


def test_message_role_enum_covers_three_roles() -> None:
    """MessageRole covers the in-scope OpenAI / opensource roles: user / assistant / tool.

    ``system`` is intentionally excluded — boundary extractors strip system prompts before emitting
    MemCells, and opensource ``MessageSenderRole`` doesn't include it either.
    """
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
    assert msg.tool_calls is None
    assert msg.tool_call_id is None
    assert msg.sender_id is None
    assert msg.sender_name is None
    assert msg.refer_list == []


def test_message_role_accepts_string_value() -> None:
    msg = Message(role="assistant", content="hi", timestamp=1)  # type: ignore[arg-type]
    assert msg.role == MessageRole.ASSISTANT


def test_message_missing_timestamp_raises() -> None:
    with pytest.raises(ValidationError):
        Message(role=MessageRole.USER, content="hello")  # type: ignore[call-arg]


def test_message_system_role_rejected() -> None:
    """``system`` is intentionally outside the taxonomy — boundary extractors strip system prompts."""
    with pytest.raises(ValidationError):
        Message(role="system", content="You are an agent.", timestamp=1)  # type: ignore[arg-type]


def test_message_accepts_tool_role() -> None:
    """Tool role round-trips through validation for the agent-trace path."""
    tool_msg = Message(role="tool", content="ok", timestamp=1, tool_call_id="call_1")  # type: ignore[arg-type]
    assert tool_msg.role == MessageRole.TOOL
    assert tool_msg.tool_call_id == "call_1"


def test_message_assistant_with_tool_calls_null_content() -> None:
    """Assistant messages emitting only tool_calls have ``content=null`` (OpenAI wire format)."""
    msg = Message.model_validate(
        {
            "role": "assistant",
            "content": None,
            "timestamp": 1,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q": "x"}'}}
            ],
        }
    )
    assert msg.role == MessageRole.ASSISTANT
    assert msg.content is None
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0].id == "call_1"
    assert msg.tool_calls[0].function["name"] == "search"


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
            "message_id": "m1",
        }
    )
    assert msg.sender_id == "u1"
    assert msg.sender_name == "Alice"
    assert msg.refer_list == [{"_id": "u2", "name": "Bob"}]
    assert not hasattr(msg, "message_id")


def test_message_extra_fields_dropped_when_unrecognized() -> None:
    """Unknown OpenAI payload fields (e.g. ``message_id``) are silently dropped via ``extra='ignore'``."""
    msg = Message.model_validate(
        {
            "role": "user",
            "content": "hi",
            "timestamp": 1,
            "message_id": "m1",
            "parsed_summary": "ignored",
        }
    )
    assert msg.role == MessageRole.USER
    assert not hasattr(msg, "message_id")
    assert not hasattr(msg, "parsed_summary")


def test_message_json_round_trip() -> None:
    """Plain user message — defaulted optional fields serialise to their defaults."""
    msg = Message(role=MessageRole.USER, content="hi", timestamp=42)
    serialised = msg.model_dump_json()
    assert json.loads(serialised) == {
        "role": "user",
        "content": "hi",
        "timestamp": 42,
        "sender_id": None,
        "sender_name": None,
        "refer_list": [],
        "tool_calls": None,
        "tool_call_id": None,
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


def test_message_tool_round_trip() -> None:
    """Tool-call assistant message + its tool response — full agent-trace round-trip."""
    assistant = Message(
        role=MessageRole.ASSISTANT,
        content=None,
        timestamp=10,
        tool_calls=[ToolCall(id="c1", type="function", function={"name": "lookup", "arguments": '{"k":"v"}'})],
    )
    tool = Message(role=MessageRole.TOOL, content="result", timestamp=11, tool_call_id="c1")
    for original in (assistant, tool):
        assert Message.model_validate_json(original.model_dump_json()) == original


def test_tool_call_default_type_is_function() -> None:
    """ToolCall.type defaults to ``"function"`` (OpenAI's only tool-call type today)."""
    tc = ToolCall(id="c1", function={"name": "x", "arguments": "{}"})
    assert tc.type == "function"
