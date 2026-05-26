"""Tests for adetect_boundary_step — single-step LLM boundary primitive."""

from __future__ import annotations

import json

import pytest

from everalgo.boundary import BoundaryDecision, adetect_boundary_step
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
    assert decision.should_end is True
    assert decision.topic_summary == "Cooking discussion"


async def test_keep_waiting_decision_parsed_from_llm_json() -> None:
    payload = json.dumps({"should_end": False, "reasoning": "same topic", "confidence": 0.7, "topic_summary": ""})
    fake = FakeLLMClient(responses=[payload])
    decision = await adetect_boundary_step([_msg(0), _msg(1)], [_msg(2)], llm=fake)
    assert decision.should_end is False


async def test_fails_loud_on_bad_llm_response() -> None:
    """Algorithm fails loud on first bad response — caller owns retry/fallback policy."""
    fake = FakeLLMClient(responses=["junk"])
    with pytest.raises(ValueError, match="JSON"):
        await adetect_boundary_step([_msg(0), _msg(1)], [_msg(2)], llm=fake)
