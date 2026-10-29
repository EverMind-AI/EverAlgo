"""The default hard limits: 65536 tokens / 500 messages, on both the batch and the step path.

Each limit is pinned by what it does at the edge — an input one below it reaches the model whole, an input at it
is cut without the model — rather than by comparing the default constants with themselves.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from everalgo.boundary import BoundaryDetector, adetect_boundaries
from everalgo.boundary._render import _count_history_tokens
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

_TS = 1_700_000_000_000
_TOKEN_LIMIT = 65_536
_MESSAGE_LIMIT = 500
_BATCH_REPLY = json.dumps({"reasoning": "r", "boundaries": []})
_STEP_REPLY = json.dumps({"reasoning": "r", "should_end": False, "confidence": 0.5, "topic_summary": ""})


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


async def _batch_cell_sizes(messages: Sequence[ChatMessage]) -> list[int]:
    result = await adetect_boundaries(messages, llm=FakeLLMClient(responses=[_BATCH_REPLY]), is_final=True)
    return [len(cell.items) for cell in result.cells]


async def _step_calls_llm(history: list[ChatMessage], new: ChatMessage) -> bool:
    fake = FakeLLMClient(responses=[_STEP_REPLY])
    await BoundaryDetector(llm=fake).adetect_step(history, new)
    return fake.call_count == 1


# ── Batch ────────────────────────────────────────────────────────────────────────────────────────────


async def test_batch_keeps_one_message_below_the_default_message_limit_whole() -> None:
    assert await _batch_cell_sizes(_messages(_MESSAGE_LIMIT - 1)) == [_MESSAGE_LIMIT - 1]


async def test_batch_pre_cuts_at_the_default_message_limit() -> None:
    assert await _batch_cell_sizes(_messages(_MESSAGE_LIMIT)) == [_MESSAGE_LIMIT - 1, 1]


async def test_batch_keeps_one_token_below_the_default_token_limit_whole() -> None:
    assert await _batch_cell_sizes(_messages_totalling(_TOKEN_LIMIT - 1, count=2)) == [2]


async def test_batch_pre_cuts_at_the_default_token_limit() -> None:
    assert await _batch_cell_sizes(_messages_totalling(_TOKEN_LIMIT, count=2)) == [1, 1]


async def test_the_class_shares_the_batch_defaults() -> None:
    fake = FakeLLMClient(responses=[_BATCH_REPLY])

    result = await BoundaryDetector(llm=fake).adetect(_messages(_MESSAGE_LIMIT), is_final=True)

    assert [len(cell.items) for cell in result.cells] == [_MESSAGE_LIMIT - 1, 1]


# ── Step ─────────────────────────────────────────────────────────────────────────────────────────────


async def test_step_asks_the_model_one_message_below_the_default_message_limit() -> None:
    *history, new = _messages(_MESSAGE_LIMIT - 1)
    assert await _step_calls_llm(history, new)


async def test_step_force_splits_at_the_default_message_limit() -> None:
    *history, new = _messages(_MESSAGE_LIMIT)
    assert not await _step_calls_llm(history, new)


async def test_step_asks_the_model_one_token_below_the_default_token_limit() -> None:
    *history, new = _messages_totalling(_TOKEN_LIMIT - 1, count=3)
    assert await _step_calls_llm(history, new)


async def test_step_force_splits_at_the_default_token_limit() -> None:
    *history, new = _messages_totalling(_TOKEN_LIMIT, count=3)
    assert not await _step_calls_llm(history, new)
