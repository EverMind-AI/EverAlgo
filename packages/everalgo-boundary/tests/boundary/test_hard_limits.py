"""The default hard limits: 65536 tokens / 500 messages on the batch path, 8192 tokens / 50 messages on the step path.

The two limits share a name but bound different things: on the batch path the window sent to the model, which holds
several segments; on the step path one segment, which every step call sends whole. Each limit is pinned by what it
does at the edge — an input one below it reaches the model, an input at it is cut without the model — rather than by
comparing the default constants with themselves.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from everalgo.boundary import BoundaryDetector, Cuts
from everalgo.boundary._render import _count_history_tokens
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

_TS = 1_700_000_000_000
_BATCH_TOKEN_LIMIT = 65_536
_BATCH_MESSAGE_LIMIT = 500
_STEP_TOKEN_LIMIT = 8_192
_STEP_MESSAGE_LIMIT = 50
_BATCH_REPLY = json.dumps({"reasoning": "r", "boundaries": []})
_STEP_REPLY = json.dumps({"reasoning": "r", "should_end": False})


def _message(index: int, words: int = 1) -> ChatMessage:
    """``u1: hello hello …`` — counts ``words + 3`` tokens under ``o200k_base``."""
    return ChatMessage(
        id=f"m{index}", role="user", content=" ".join(["hello"] * words), timestamp=_TS + index, sender_id="u1"
    )


def _messages(count: int) -> list[ChatMessage]:
    return [_message(index) for index in range(count)]


def _messages_totalling(tokens: int, *, count: int) -> list[ChatMessage]:
    """``count`` messages holding exactly ``tokens`` tokens between them, the last one carrying the bulk."""
    short = [_message(index) for index in range(count - 1)]
    short_tokens = _count_history_tokens(short)
    messages = [*short, _message(count - 1, words=tokens - short_tokens - 3)]
    assert _count_history_tokens(messages) == tokens, "token calibration drifted"
    return messages


async def _batch_cuts(messages: Sequence[ChatMessage]) -> Cuts:
    return await BoundaryDetector(llm=FakeLLMClient(responses=[_BATCH_REPLY])).adetect_batch(messages, is_final=True)


async def _step_calls_llm(items: Sequence[ChatMessage]) -> tuple[bool, Cuts]:
    fake = FakeLLMClient(responses=[_STEP_REPLY])
    cuts = await BoundaryDetector(llm=fake).adetect_step(items, smart_mask=False)
    return fake.call_count == 1, cuts


# ── Batch ────────────────────────────────────────────────────────────────────────────────────────────


async def test_batch_keeps_one_message_below_the_default_message_limit_whole() -> None:
    assert await _batch_cuts(_messages(_BATCH_MESSAGE_LIMIT - 1)) == Cuts([499], 499)


async def test_batch_pre_cuts_at_the_default_message_limit() -> None:
    assert await _batch_cuts(_messages(_BATCH_MESSAGE_LIMIT)) == Cuts([499, 500], 500)


async def test_batch_keeps_one_token_below_the_default_token_limit_whole() -> None:
    assert await _batch_cuts(_messages_totalling(_BATCH_TOKEN_LIMIT - 1, count=2)) == Cuts([2], 2)


async def test_batch_pre_cuts_at_the_default_token_limit() -> None:
    assert await _batch_cuts(_messages_totalling(_BATCH_TOKEN_LIMIT, count=2)) == Cuts([1, 2], 2)


# ── Step ─────────────────────────────────────────────────────────────────────────────────────────────


async def test_step_asks_the_model_one_message_below_the_default_message_limit() -> None:
    assert await _step_calls_llm(_messages(_STEP_MESSAGE_LIMIT - 1)) == (True, Cuts([], 0))


async def test_step_force_splits_at_the_default_message_limit() -> None:
    assert await _step_calls_llm(_messages(_STEP_MESSAGE_LIMIT)) == (False, Cuts([49], 49))


async def test_step_asks_the_model_one_token_below_the_default_token_limit() -> None:
    assert await _step_calls_llm(_messages_totalling(_STEP_TOKEN_LIMIT - 1, count=3)) == (True, Cuts([], 0))


async def test_step_force_splits_at_the_default_token_limit() -> None:
    assert await _step_calls_llm(_messages_totalling(_STEP_TOKEN_LIMIT, count=3)) == (False, Cuts([2], 2))
