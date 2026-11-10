"""Tests for BoundaryDetector.adetect_step — the step path (the EverAlgo LoCoMo reference run's incremental loop).

``adetect_step(items)`` judges the last item against the history before it and returns one of three ``Cuts``:

- ``Cuts([], 0)`` — no cut: the new message joins the history (warmup and an empty shown history included);
- ``Cuts([n], n)`` — the ``n``-message history closes, the new message starts the next segment;
- ``Cuts([n], n - 1)`` — the history closes and its last message, which the model was not shown, is bridged into
  the next segment as well.

A model cut and a hard-limit cut have the same shape.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from everalgo.boundary import BoundaryDetector, Cuts, to_memcells
from everalgo.boundary._render import _calculate_time_gap
from everalgo.boundary._replies import _StepReply
from everalgo.llm import LLMOutputError
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, ToolCallResult
from everalgo.utils import InvalidInputError

if TYPE_CHECKING:
    from everalgo.llm import LLMMessage


def _msg(idx: int, content: str = "hello world") -> ChatMessage:
    return ChatMessage(
        id=f"m{idx}",
        role="user",
        content=content,
        timestamp=1_700_000_000_000 + idx * 30_000,
        sender_id=f"u{idx}",
        sender_name=None,
    )


def _items(count: int) -> list[ChatMessage]:
    """``count - 1`` history messages followed by the new one."""
    return [_msg(i, f"text {i}") for i in range(count)]


def _reply(*, should_end: bool) -> str:
    return json.dumps({"reasoning": "ok", "should_end": should_end})


async def _step(items: list[ChatMessage], *replies: str, **kwargs: Any) -> tuple[Cuts, FakeLLMClient]:
    fake = FakeLLMClient(responses=list(replies))
    return await BoundaryDetector(llm=fake).adetect_step(items, **kwargs), fake


def _shown_history(fake: FakeLLMClient) -> str:
    user = str(fake.calls[0].messages[1].content)
    return user.split("<conversation_history>\n", 1)[1].split("\n</conversation_history>", 1)[0]


# ── The three outcomes ───────────────────────────────────────────────────────────────────────────────


async def test_no_cut_keeps_everything_for_the_next_call() -> None:
    items = _items(3)

    cuts, fake = await _step(items, _reply(should_end=False))

    assert cuts == Cuts(ends=[], tail_start=0)
    assert to_memcells(items, cuts) == []
    assert items[cuts.tail_start :] == items
    assert fake.call_count == 1


async def test_cut_without_bridge_closes_the_history_and_starts_at_the_new_message() -> None:
    items = _items(4)

    cuts, fake = await _step(items, _reply(should_end=True))

    assert cuts == Cuts(ends=[3], tail_start=3)
    assert [cell.items for cell in to_memcells(items, cuts)] == [items[:3]]
    assert items[cuts.tail_start :] == [items[3]]
    assert _shown_history(fake).count("\n") == 2, "the whole history is shown"


async def test_cut_with_bridge_carries_the_masked_message_into_the_next_segment() -> None:
    """Above the threshold the model is not shown the history's last message; a cut bridges it forward."""
    items = _items(8)

    cuts, fake = await _step(items, _reply(should_end=True))

    assert cuts == Cuts(ends=[7], tail_start=6)
    assert [cell.items for cell in to_memcells(items, cuts)] == [items[:7]]
    assert items[cuts.tail_start :] == [items[6], items[7]]
    shown = _shown_history(fake)
    assert "text 5" in shown
    assert "text 6" not in shown


@pytest.mark.parametrize(("count", "expected"), [(6, Cuts([5], 5)), (7, Cuts([6], 5))], ids=["n=5", "n=6"])
async def test_the_bridge_starts_above_the_threshold(count: int, expected: Cuts) -> None:
    cuts, _ = await _step(_items(count), _reply(should_end=True))

    assert cuts == expected


async def test_without_smart_mask_the_model_sees_everything_and_nothing_is_bridged() -> None:
    items = _items(8)

    cuts, fake = await _step(items, _reply(should_end=True), smart_mask=False)

    assert cuts == Cuts(ends=[7], tail_start=7)
    assert "text 6" in _shown_history(fake)


async def test_the_masked_history_sets_the_time_gap() -> None:
    """The gap runs from the last message the model is shown to the new one."""
    items = [_msg(i) for i in range(7)]
    items[-1] = items[-1].model_copy(update={"timestamp": items[-2].timestamp + 30_000})

    _, fake = await _step(items, _reply(should_end=False))

    assert "<time_gap>\nTime gap: 1 minutes (recent conversation)\n</time_gap>" in str(
        fake.calls[0].messages[1].content
    )


# ── Warmup ───────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("count", [1, 2])
async def test_a_history_shorter_than_the_warmup_takes_the_new_message_without_the_model(count: int) -> None:
    cuts, fake = await _step(_items(count))

    assert cuts == Cuts(ends=[], tail_start=0)
    assert fake.call_count == 0


async def test_the_first_judgement_comes_at_the_warmup_length() -> None:
    _, fake = await _step(_items(3), _reply(should_end=False))

    assert fake.call_count == 1


async def test_the_warmup_is_a_keyword() -> None:
    _, fake = await _step(_items(2), _reply(should_end=False), warmup_messages=1)

    assert fake.call_count == 1


async def test_an_empty_shown_history_is_never_put_to_the_model() -> None:
    """Masking a one-message history leaves nothing to show; the first message cannot end an episode."""
    cuts, fake = await _step(_items(2), warmup_messages=0, smart_mask_threshold=0)

    assert cuts == Cuts(ends=[], tail_start=0)
    assert fake.call_count == 0


# ── Hard limits ──────────────────────────────────────────────────────────────────────────────────────


async def test_force_split_on_message_limit_cuts_without_the_model() -> None:
    items = _items(11)

    cuts, fake = await _step(items, hard_message_limit=10)
    no_mask, _ = await _step(items, hard_message_limit=10, smart_mask=False)

    assert cuts == Cuts(ends=[10], tail_start=9), "bridged: the history is longer than the threshold"
    assert no_mask == Cuts(ends=[10], tail_start=10)
    assert fake.call_count == 0


async def test_force_split_on_token_limit_cuts_without_the_model() -> None:
    items = [*_items(2), _msg(2, "word " * 200)]

    cuts, fake = await _step(items, hard_token_limit=100)

    assert cuts == Cuts(ends=[2], tail_start=2)
    assert fake.call_count == 0


async def test_force_split_with_a_too_short_history_falls_through_to_the_model() -> None:
    """A single oversized history message is never cut alone: the model still decides."""
    items = [_msg(0, "x" * 100_000), _msg(1)]

    cuts, fake = await _step(items, _reply(should_end=False), warmup_messages=1, hard_token_limit=100)

    assert cuts == Cuts(ends=[], tail_start=0)
    assert fake.call_count == 1


# ── Input contract ───────────────────────────────────────────────────────────────────────────────────


async def test_empty_items_are_refused() -> None:
    with pytest.raises(InvalidInputError, match=r"^adetect_step got no items: "):
        await _step([])


@pytest.mark.parametrize("position", [0, 2])
async def test_a_tool_item_is_refused_wherever_it_is(position: int) -> None:
    items: list[Any] = _items(3)
    items[position] = ToolCallResult(tool_call_id="c1", content="result", timestamp=1_700_000_000_000)
    fake = FakeLLMClient(responses=[])

    with pytest.raises(
        InvalidInputError, match=rf"^adetect_step item {position} is a ToolCallResult, not a ChatMessage"
    ):
        await BoundaryDetector(llm=fake).adetect_step(items)

    assert fake.call_count == 0


# ── The model call ───────────────────────────────────────────────────────────────────────────────────


async def test_the_call_requests_the_step_reply_model() -> None:
    _, fake = await _step(_items(3), _reply(should_end=False))

    assert [call.response_format for call in fake.calls] == [_StepReply]


async def test_a_prompt_only_reply_with_only_should_end_is_accepted() -> None:
    """Only ``should_end`` decides a cut; ``reasoning`` defaults to ``""`` as the measured parser filled it."""
    cuts, _ = await _step(_items(3), json.dumps({"should_end": True}))

    assert cuts == Cuts(ends=[2], tail_start=2)


_SECRET = "STEP-SECRET-9b2d"
_VALID = {"reasoning": _SECRET, "should_end": False}


class _ClientError(Exception):
    """A host's own call-failure type, which the operator must neither wrap nor retype."""


@pytest.mark.parametrize(
    ("reply", "failing_location"),
    [
        pytest.param(f"junk {_SECRET}", "<root>", id="not-json"),
        pytest.param(json.dumps({**_VALID, "note": "x"}), "note:", id="extra-field-forbidden"),
        pytest.param(json.dumps({**_VALID, "confidence": 0.7}), "confidence:", id="removed-confidence-forbidden"),
        pytest.param(
            json.dumps({**_VALID, "topic_summary": ""}), "topic_summary:", id="removed-topic-summary-forbidden"
        ),
        pytest.param(
            json.dumps({k: v for k, v in _VALID.items() if k != "should_end"}),
            "should_end:",
            id="should-end-missing",
        ),
        pytest.param(json.dumps({**_VALID, "should_end": "false"}), "should_end:", id="string-bool-not-coerced"),
    ],
)
async def test_fails_loud_on_bad_llm_response(reply: str, failing_location: str) -> None:
    """Algorithm fails loud on first bad response — caller owns retry/fallback policy."""
    fake = FakeLLMClient(responses=[reply])

    with pytest.raises(LLMOutputError) as excinfo:
        await BoundaryDetector(llm=fake).adetect_step(_items(3))

    assert isinstance(excinfo.value.__cause__, ValidationError)
    assert "boundary step reply is not a valid _StepReply" in str(excinfo.value)
    assert failing_location in str(excinfo.value)
    assert _SECRET not in str(excinfo.value)
    assert fake.call_count == 1


async def test_client_exception_propagates_unchanged() -> None:
    error = _ClientError("upstream down")

    with pytest.raises(_ClientError) as excinfo:
        await BoundaryDetector(llm=FakeLLMClient(responses=[error])).adetect_step(_items(3))

    assert excinfo.value is error


async def test_step_reply_is_logged_without_content(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="everalgo.boundary"):
        await _step(_items(3), json.dumps(_VALID))

    assert caplog.messages == ["boundary step reply: should_end=False"]
    assert _SECRET not in caplog.text


async def test_a_cut_logs_one_info_line_and_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """The reply carries no topic summary any more, so nothing warns about a missing one."""
    with caplog.at_level(logging.INFO, logger="everalgo.boundary"):
        await _step(_items(3), json.dumps({"should_end": True}))

    assert [(record.levelname, record.getMessage()) for record in caplog.records] == [
        ("INFO", "boundary step reply: should_end=True"),
    ]


async def test_the_shown_history_is_rendered_into_the_prompt() -> None:
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMMessage], **_: Any) -> str:
        captured["content"] = messages[1].content
        return _reply(should_end=False)

    await BoundaryDetector(llm=FakeLLMClient(handler=handler)).adetect_step(
        [_msg(0, "alpha"), _msg(1, "beta"), _msg(2, "gamma")]
    )

    assert "alpha" in captured["content"]
    assert "beta" in captured["content"]
    assert "<new_messages>\n[2023-11-14 22:14 UTC (Tuesday)] u2: gamma\n</new_messages>" in captured["content"]


# ===========================================================================
# time gap wording — the reference run's labels (everalgo-boundary 0.2.1), unit always plural
# ===========================================================================


@pytest.mark.parametrize(
    ("gap_ms", "expected"),
    [
        (1_000, "Time gap: 1 seconds (immediate response)"),
        (59_999, "Time gap: 59 seconds (immediate response)"),
        (90_000, "Time gap: 1 minutes (recent conversation)"),
        (3_599_999, "Time gap: 59 minutes (recent conversation)"),
        (3_600_000, "Time gap: 1 hours (same day, but significant pause)"),
        (86_399_999, "Time gap: 23 hours (same day, but significant pause)"),
        (93_600_000, "Time gap: 1 days (long gap, likely new conversation)"),
        (172_800_000, "Time gap: 2 days (long gap, likely new conversation)"),
        (-1_000, "Time gap: Messages appear to be out of order"),
    ],
)
def test_time_gap_uses_the_reference_labels(gap_ms: int, expected: str) -> None:
    """The step prompt keeps the reference run's wording byte for byte, ``1 minutes`` included."""
    before, after = _msg(0), _msg(1)
    after = after.model_copy(update={"timestamp": before.timestamp + gap_ms})
    assert _calculate_time_gap(before, after) == expected


def test_time_gap_reads_a_zero_timestamp_as_missing() -> None:
    assert _calculate_time_gap(_msg(0).model_copy(update={"timestamp": 0}), _msg(1)) == (
        "No timestamp information available"
    )
