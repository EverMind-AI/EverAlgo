"""Tests for everalgo.types.memcell.MemCell."""

import pytest
from pydantic import ValidationError

from everalgo.types import MemCell, Message, MessageRole


def _msg(content: str = "hi", ts: int = 1) -> Message:
    return Message(role=MessageRole.USER, content=content, timestamp=ts)


def test_memcell_minimum_required_fields() -> None:
    cell = MemCell(id="m1", messages=[_msg()], timestamp=1700000000000)
    assert cell.id == "m1"
    assert len(cell.messages) == 1
    assert cell.timestamp == 1700000000000


def test_memcell_messages_coerced_from_dicts() -> None:
    """Pydantic should rebuild Message objects from raw dicts inside `messages`."""
    cell = MemCell.model_validate(
        {
            "id": "m1",
            "timestamp": 1,
            "messages": [{"role": "user", "content": "hi", "timestamp": 1}],
        }
    )
    assert cell.messages[0].role == MessageRole.USER
    assert cell.messages[0].content == "hi"


def test_memcell_missing_id_raises() -> None:
    with pytest.raises(ValidationError):
        MemCell(messages=[_msg()], timestamp=1)  # type: ignore[call-arg]


def test_memcell_missing_messages_raises() -> None:
    with pytest.raises(ValidationError):
        MemCell(id="m1", timestamp=1)  # type: ignore[call-arg]


def test_memcell_empty_messages_allowed() -> None:
    """Type does not enforce min_length=1; caller (boundary extractor) decides."""
    cell = MemCell(id="m1", messages=[], timestamp=1)
    assert cell.messages == []


def test_memcell_extra_fields_silently_ignored() -> None:
    """Opensource MemCell carries source_type / sender_ids / user_id_list / group_id / participants — drop all."""
    cell = MemCell.model_validate(
        {
            "id": "m1",
            "messages": [],
            "timestamp": 1,
            "source_type": "chat",
            "sender_ids": ["u1"],
            "user_id_list": ["u1", "u2"],
            "group_id": "g1",
            "participants": ["u1"],
            "type": "Conversation",
        }
    )
    assert cell.id == "m1"
    assert not hasattr(cell, "source_type")
    assert not hasattr(cell, "group_id")
    assert not hasattr(cell, "user_id_list")


def test_memcell_json_round_trip() -> None:
    cell = MemCell(id="m1", messages=[_msg("hello", 5)], timestamp=10)
    serialised = cell.model_dump_json()
    assert MemCell.model_validate_json(serialised) == cell
