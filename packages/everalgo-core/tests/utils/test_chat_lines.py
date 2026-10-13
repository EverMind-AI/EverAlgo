"""Tests for ``everalgo.utils.chat_lines``."""

from __future__ import annotations

import pytest

from everalgo.types import ChatMessage
from everalgo.utils import render_chat_line, speaker

_WEDNESDAY_14_00_59 = 1_712_757_659_999  # 2024-04-10T14:00:59.999Z


def _message(
    content: str = "hi", *, timestamp: int = _WEDNESDAY_14_00_59, sender_name: str | None = "Alice"
) -> ChatMessage:
    return ChatMessage(
        id="m1", role="user", content=content, timestamp=timestamp, sender_id="u_alice", sender_name=sender_name
    )


def test_line_is_the_utc_minute_with_its_weekday_then_speaker_and_text() -> None:
    """Seconds are truncated, not rounded: 14:00:59.999 still renders as 14:00."""
    assert render_chat_line(_message()) == "[2024-04-10 14:00 UTC (Wednesday)] Alice: hi"


def test_line_places_content_verbatim_newlines_included() -> None:
    assert render_chat_line(_message("first\nsecond")) == "[2024-04-10 14:00 UTC (Wednesday)] Alice: first\nsecond"


@pytest.mark.parametrize("sender_name", [None, ""])
def test_speaker_falls_back_to_the_sender_id(sender_name: str | None) -> None:
    message = _message(sender_name=sender_name)

    assert speaker(message) == "u_alice"
    assert render_chat_line(message) == "[2024-04-10 14:00 UTC (Wednesday)] u_alice: hi"


def test_speaker_prefers_the_sender_name() -> None:
    assert speaker(_message()) == "Alice"


def test_line_rejects_a_message_without_a_timestamp_naming_only_its_id() -> None:
    """A timestamp of 0 would render as 1970-01-01; the error never carries the text or the speaker."""
    with pytest.raises(ValueError, match=r"^chat message 'm1' has no timestamp: ") as raised:
        render_chat_line(_message("secret plan", timestamp=0))

    assert "secret plan" not in str(raised.value)
    assert "Alice" not in str(raised.value)
