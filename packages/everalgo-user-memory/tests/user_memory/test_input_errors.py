"""The caller-input checks raise ``InvalidInputError``, with the offender located and the remedy named."""

from __future__ import annotations

import pytest

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, Episode, MemCell
from everalgo.user_memory import ProfileExtractor
from everalgo.user_memory._language import resolve_output_language
from everalgo.user_memory.reflect import _validate_inputs
from everalgo.utils import InvalidInputError


def _episode(timestamp: int) -> Episode:
    return Episode(owner_id=None, episode="narrative", subject="topic", summary="a preview", timestamp=timestamp)


def test_unsorted_episodes_name_the_pair_and_the_order_to_pass() -> None:
    with pytest.raises(InvalidInputError, match=r"episodes\[0\]\.timestamp=3000 is later than episodes\[1\]") as raised:
        _validate_inputs([_episode(3000), _episode(1000)], min_count=2)

    assert "oldest first" in str(raised.value)


def test_too_few_episodes_say_how_many_arrived_and_how_many_are_needed() -> None:
    with pytest.raises(InvalidInputError, match=r"episodes has 1 item\(s\), reflection needs at least 2"):
        _validate_inputs([_episode(1000)], min_count=2)


def test_an_unknown_language_lists_what_is_supported() -> None:
    with pytest.raises(InvalidInputError, match=r"output_language 'Klingon' names no supported language") as raised:
        resolve_output_language("Klingon")

    assert "Chinese" in str(raised.value)


async def test_a_sender_who_is_not_in_the_conversation_is_invalid_input_before_any_llm_call() -> None:
    fake = FakeLLMClient(responses=[])
    message = ChatMessage(id="m1", role="user", content="hi", timestamp=1, sender_id="u_alice")
    cell = MemCell(items=[message], timestamp=1)

    with pytest.raises(InvalidInputError, match=r"sender_id 'u_ghost' is not a user speaker in memcells"):
        await ProfileExtractor(llm=fake).aextract([cell], sender_id="u_ghost")

    assert fake.call_count == 0
