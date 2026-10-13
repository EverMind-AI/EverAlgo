"""Tests for Foresight's conversation rendering: the line format of the user message's ``<conversation>``."""

from __future__ import annotations

import pytest

from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory.foresight import _render_conversation

_WEDNESDAY_14_00_59 = 1_712_757_659_999  # 2024-04-10T14:00:59.999Z


def _cell(*messages: ChatMessage) -> MemCell:
    return MemCell(items=list(messages), timestamp=_WEDNESDAY_14_00_59)


def _message(content: str, *, timestamp: int = _WEDNESDAY_14_00_59) -> ChatMessage:
    return ChatMessage(
        id="m1", role="user", content=content, timestamp=timestamp, sender_id="u_alice", sender_name="Alice"
    )


def test_each_line_is_the_utc_minute_with_its_weekday_then_speaker_and_text() -> None:
    assert _render_conversation(_cell(_message("hi"))) == "[2024-04-10 14:00 UTC (Wednesday)] Alice: hi"


def test_a_message_without_text_is_skipped_unread() -> None:
    assert _render_conversation(_cell(_message("", timestamp=0))) == ""


def test_a_message_with_text_but_no_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"^chat message 'm1' has no timestamp$"):
        _render_conversation(_cell(_message("hi", timestamp=0)))
