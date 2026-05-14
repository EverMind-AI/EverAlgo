"""Tests for everalgo.types.memcell.MemCell — aligned with new-release opensource schema."""

import pytest
from pydantic import ValidationError

from everalgo.types import MemCell, Message, MessageRole, RawDataType


def _msg(content: str = "hi", ts: int = 1) -> Message:
    return Message(role=MessageRole.USER, content=content, timestamp=ts)


def _od(msgs: list[Message]) -> list[dict[str, object]]:
    return [{"message": m.model_dump(exclude_none=True)} for m in msgs]


def test_memcell_minimum_required_fields() -> None:
    cell = MemCell(original_data=_od([_msg()]), timestamp=1700000000000)
    assert cell.timestamp == 1700000000000
    assert cell.event_id is None
    assert len(cell.original_data) == 1


def test_memcell_event_id_optional() -> None:
    cell = MemCell(original_data=_od([_msg()]), timestamp=1, event_id="mc_42")
    assert cell.event_id == "mc_42"


def test_memcell_messages_property_reconstructs_typed_messages() -> None:
    """The :attr:`MemCell.messages` property rebuilds typed Message instances from original_data."""
    cell = MemCell.model_validate(
        {
            "original_data": [{"message": {"role": "user", "content": "hi", "timestamp": 1}}],
            "timestamp": 1,
        }
    )
    assert cell.messages[0].role == MessageRole.USER
    assert cell.messages[0].content == "hi"


def test_memcell_missing_timestamp_raises() -> None:
    with pytest.raises(ValidationError):
        MemCell(original_data=_od([_msg()]))  # type: ignore[call-arg]


def test_memcell_empty_original_data_allowed() -> None:
    """Type does not enforce min_length=1 on original_data; caller decides semantics."""
    cell = MemCell(original_data=[], timestamp=1)
    assert cell.original_data == []
    assert cell.messages == []


def test_memcell_opensource_fields_accepted() -> None:
    """Opensource schema fields user_id_list / group_id / sender_ids / participants / type are first-class."""
    cell = MemCell.model_validate(
        {
            "user_id_list": ["u1", "u2"],
            "original_data": [],
            "timestamp": 1,
            "event_id": "evt_1",
            "group_id": "g1",
            "sender_ids": ["u1"],
            "participants": ["u1"],
            "type": "conversation",
        }
    )
    assert cell.user_id_list == ["u1", "u2"]
    assert cell.group_id == "g1"
    assert cell.sender_ids == ["u1"]
    assert cell.participants == ["u1"]
    assert cell.type == RawDataType.CONVERSATION


def test_memcell_extra_unknown_fields_silently_ignored() -> None:
    """`extra='ignore'` keeps unmodelled opensource payload keys deserialisable."""
    cell = MemCell.model_validate(
        {
            "original_data": [],
            "timestamp": 1,
            "source_type": "chat",  # unknown / legacy
            "extend": {"foo": "bar"},  # unknown / legacy
        }
    )
    assert not hasattr(cell, "source_type")
    assert not hasattr(cell, "extend")


def test_memcell_json_round_trip() -> None:
    cell = MemCell(
        original_data=_od([_msg("hello", 5)]),
        timestamp=10,
        event_id="mc_5_5",
        participants=["u1"],
        sender_ids=["u1"],
        type=RawDataType.CONVERSATION,
    )
    serialised = cell.model_dump_json()
    assert MemCell.model_validate_json(serialised) == cell


def test_memcell_messages_not_serialised() -> None:
    """`messages` is a property over `original_data`, not a stored field."""
    cell = MemCell(original_data=_od([_msg()]), timestamp=1)
    dumped = cell.model_dump()
    assert "messages" not in dumped
    assert "original_data" in dumped
