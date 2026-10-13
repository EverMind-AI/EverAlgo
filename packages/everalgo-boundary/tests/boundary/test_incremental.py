"""Tests for adetect_boundary_step — single-step LLM boundary primitive."""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from everalgo.boundary import BoundaryDecision, adetect_boundary_step
from everalgo.boundary._replies import _StepReply
from everalgo.llm import LLMOutputError
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage


def _msg(idx: int, content: str = "hi") -> ChatMessage:
    return ChatMessage(
        id=f"m{idx}",
        role="user",
        content=content,
        timestamp=1_700_000_000_000 + idx * 30_000,
        sender_id=f"u{idx}",
        sender_name=None,
    )


async def test_empty_history_short_circuits_without_llm() -> None:
    """93 alignment: empty conversation_history returns first-msg decision, no LLM call."""
    fake = FakeLLMClient(responses=[])
    decision = await adetect_boundary_step([], [_msg(0)], llm=fake)
    assert decision == BoundaryDecision(
        should_end=False,
        reasoning="First messages in conversation",
        confidence=1.0,
        topic_summary="",
    )
    assert fake.call_count == 0


async def test_committable_decision_parsed_from_llm_json() -> None:
    payload = json.dumps(
        {
            "should_end": True,
            "reasoning": "topic shift",
            "confidence": 0.9,
            "topic_summary": "Cooking discussion",
        }
    )
    fake = FakeLLMClient(responses=[payload])
    decision = await adetect_boundary_step([_msg(0), _msg(1)], [_msg(2)], llm=fake)
    assert decision == BoundaryDecision(
        should_end=True, reasoning="topic shift", confidence=0.9, topic_summary="Cooking discussion"
    )
    assert [call.response_format for call in fake.calls] == [_StepReply]


async def test_keep_waiting_decision_parsed_from_llm_json() -> None:
    payload = json.dumps({"should_end": False, "reasoning": "same topic", "confidence": 0.7, "topic_summary": ""})
    fake = FakeLLMClient(responses=[payload])
    decision = await adetect_boundary_step([_msg(0), _msg(1)], [_msg(2)], llm=fake)
    assert decision.should_end is False


_SECRET = "STEP-SECRET-9b2d"
_VALID = {"reasoning": _SECRET, "should_end": False, "confidence": 0.7, "topic_summary": ""}


class _ClientError(Exception):
    """A host's own call-failure type, which the operator must neither wrap nor retype."""


@pytest.mark.parametrize(
    ("reply", "failing_location"),
    [
        pytest.param(f"junk {_SECRET}", "<root>", id="not-json"),
        pytest.param(json.dumps({**_VALID, "note": "x"}), "note:", id="extra-field-forbidden"),
        pytest.param(
            json.dumps({k: v for k, v in _VALID.items() if k != "topic_summary"}),
            "topic_summary:",
            id="topic-summary-missing",
        ),
        pytest.param(json.dumps({**_VALID, "should_end": "false"}), "should_end:", id="string-bool-not-coerced"),
    ],
)
async def test_fails_loud_on_bad_llm_response(reply: str, failing_location: str) -> None:
    """Algorithm fails loud on first bad response — caller owns retry/fallback policy."""
    fake = FakeLLMClient(responses=[reply])

    with pytest.raises(LLMOutputError) as excinfo:
        await adetect_boundary_step([_msg(0), _msg(1)], [_msg(2)], llm=fake)

    assert isinstance(excinfo.value.__cause__, ValidationError)
    assert "boundary step reply is not a valid _StepReply" in str(excinfo.value)
    assert failing_location in str(excinfo.value)
    assert _SECRET not in str(excinfo.value)
    assert fake.call_count == 1


async def test_client_exception_propagates_unchanged() -> None:
    error = _ClientError("upstream down")

    with pytest.raises(_ClientError) as excinfo:
        await adetect_boundary_step([_msg(0)], [_msg(1)], llm=FakeLLMClient(responses=[error]))

    assert excinfo.value is error


async def test_step_reply_is_logged_without_content(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="everalgo.boundary"):
        await adetect_boundary_step([_msg(0)], [_msg(1)], llm=FakeLLMClient(responses=[json.dumps(_VALID)]))

    assert caplog.messages == ["boundary step reply: should_end=False confidence=0.70"]
    assert _SECRET not in caplog.text
