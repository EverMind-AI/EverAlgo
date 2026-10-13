"""Tests for ``everalgo.utils.errors``: the type a host reads as "the caller's data is wrong, do not retry"."""

from __future__ import annotations

from datetime import datetime

import pytest

import everalgo.utils as utils
from everalgo.types import ChatMessage
from everalgo.utils import InvalidInputError, fetch_uri, format_utc_date, render_chat_line


def test_it_is_a_value_error_so_existing_handlers_keep_working() -> None:
    assert issubclass(InvalidInputError, ValueError)
    assert "InvalidInputError" in utils.__all__


def test_a_message_without_a_timestamp_is_invalid_input_that_names_its_id_and_the_remedy() -> None:
    message = ChatMessage(id="m-42", role="user", content="secret plan", timestamp=0, sender_id="u1")

    with pytest.raises(InvalidInputError, match=r"^chat message 'm-42' has no timestamp: ") as raised:
        render_chat_line(message)

    assert "ingress" in str(raised.value)
    assert "secret plan" not in str(raised.value)


def test_a_naive_datetime_is_invalid_input() -> None:
    with pytest.raises(InvalidInputError, match="naive datetime"):
        format_utc_date(datetime(2024, 4, 10))


@pytest.mark.parametrize(("uri", "named"), [("", "empty uri"), ("file:///etc/passwd", "'file'")])
async def test_an_unfetchable_uri_is_invalid_input(uri: str, named: str) -> None:
    with pytest.raises(InvalidInputError, match=named):
        await fetch_uri(uri)
