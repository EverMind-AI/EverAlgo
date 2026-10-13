"""Every boundary call site sends the model messages with no slot left in them.

Asserting on the templates alone would not catch the failure the call sites are exposed to: a call
site that went back to ``str.format`` would turn the user template's ``{{new_messages}}`` into a
literal ``{new_messages}`` and ship it — no exception, no missing field. So each test drives the
real function against a ``FakeLLMClient`` and reads the system and user messages the client actually
received.

The check rejects both brace shapes for that reason: ``{{messages}}`` means the renderer never
substituted it, ``{messages}`` means something other than the renderer did. A call with
``custom_instructions`` gets a case of its own.
"""

from __future__ import annotations

import json
import re

import pytest

from everalgo.boundary import adetect_boundaries, adetect_boundary_step
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage

_STEP_REPLY = json.dumps({"reasoning": "r", "should_end": False, "confidence": 0.9, "topic_summary": ""})
_SLOT_LEFT_BEHIND = re.compile(r"\{\{[A-Za-z_][A-Za-z0-9_]*\}\}|(?<!\{)\{[A-Za-z_][A-Za-z0-9_]*\}(?!\})")


def _assert_no_slot_reached_the_model(client: FakeLLMClient) -> None:
    assert client.call_count > 0, "the call site issued no LLM call, so nothing was checked"
    for record in client.calls:
        for message in record.messages:
            leftover = _SLOT_LEFT_BEHIND.findall(str(message.content))
            assert not leftover, f"unsubstituted slots {leftover} in: {message.content}"


def _msg(idx: int) -> ChatMessage:
    return ChatMessage(
        id=f"m{idx}",
        role="user",
        content=f"message {idx}",
        timestamp=1_700_000_000_000 + idx * 30_000,
        sender_id=f"u{idx}",
        sender_name=None,
    )


@pytest.mark.parametrize(
    "custom_instructions",
    [pytest.param(None, id="default"), pytest.param("Prefer fewer, longer episodes.", id="custom_instructions")],
)
async def test_detect_boundaries_sends_a_fully_rendered_prompt(custom_instructions: str | None) -> None:
    client = FakeLLMClient(responses=[json.dumps({"reasoning": "r", "boundaries": [], "tail_complete": True})])

    await adetect_boundaries([_msg(0), _msg(1)], llm=client, custom_instructions=custom_instructions)

    _assert_no_slot_reached_the_model(client)


async def test_detect_boundary_step_sends_a_fully_rendered_prompt() -> None:
    client = FakeLLMClient(responses=[_STEP_REPLY])

    await adetect_boundary_step([_msg(0)], [_msg(1)], llm=client, time_gap_info="3 days")

    _assert_no_slot_reached_the_model(client)


async def test_detect_boundary_step_renders_the_empty_time_gap_fallback() -> None:
    """``time_gap_info=""`` is falsy, and a renderer strict about ``None`` must still fill the slot."""
    client = FakeLLMClient(responses=[_STEP_REPLY])

    await adetect_boundary_step([_msg(0)], [_msg(1)], llm=client)

    _assert_no_slot_reached_the_model(client)
    assert "<time_gap>\nNo time gap information available\n</time_gap>" in str(client.calls[0].messages[1].content)
