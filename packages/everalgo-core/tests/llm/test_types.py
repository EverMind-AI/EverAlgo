"""Tests for everalgo.llm.types — LLMMessage."""

import json

import pytest
from pydantic import ValidationError

import everalgo.llm.types
from everalgo.llm.types import LLMMessage


def test_llm_message_minimum_required_fields() -> None:
    msg = LLMMessage(role="user", content="hello")
    assert msg.role == "user"
    assert msg.content == "hello"


def test_llm_message_role_accepts_three_values() -> None:
    for role in ("system", "user", "assistant"):
        msg = LLMMessage(role=role, content="x")
        assert msg.role == role


def test_llm_message_invalid_role_raises() -> None:
    with pytest.raises(ValidationError):
        LLMMessage(role="tool", content="x")  # type: ignore[arg-type]  # invalid role on purpose
    with pytest.raises(ValidationError):
        LLMMessage(role="developer", content="x")  # type: ignore[arg-type]  # invalid role on purpose


def test_llm_message_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        LLMMessage(role="user")  # type: ignore[call-arg]  # content omitted on purpose


def test_llm_message_extra_fields_silently_ignored() -> None:
    """OpenAI payload may carry name / tool_call_id — drop them."""
    msg = LLMMessage.model_validate(
        {
            "role": "assistant",
            "content": "hi",
            "name": "ignored",
            "tool_call_id": "ignored",
        }
    )
    assert not hasattr(msg, "name")
    assert not hasattr(msg, "tool_call_id")


def test_llm_message_json_round_trip() -> None:
    msg = LLMMessage(role="user", content="hi")
    serialised = msg.model_dump_json()
    assert json.loads(serialised) == {"role": "user", "content": "hi"}
    assert LLMMessage.model_validate_json(serialised) == msg


@pytest.mark.parametrize(
    "removed",
    [
        "ChatMessage",
        "ChatResponse",
        "Usage",
        "TextFormat",
        "JsonObjectFormat",
        "JsonSchemaSpec",
        "JsonSchemaFormat",
        "ResponseFormat",
    ],
)
def test_removed_names_are_gone_without_aliases(removed: str) -> None:
    """The rename and the removals leave no alias behind: a stale import must fail, not silently resolve."""
    assert not hasattr(everalgo.llm.types, removed)
