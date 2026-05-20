"""End-to-end pipeline tests: messages → boundary → user-memory extractors.

Verifies the full boundary→{episode, foresight, atomic_fact} data flow with a FakeLLMClient handler that
returns distinct JSON per call. Each pipeline file is the cross-distribution acceptance test for one
extractor on the EPISODE path (per `tests/README.md`). All downstream JSON shapes follow the schemas
defined in prompts under ``user_memory/prompts/{en,zh}/``.
"""

from __future__ import annotations

from typing import Any

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.assertions import (
    assert_atomic_fact_shape,
    assert_episode_shape,
    assert_foresight_shape,
)
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage
from everalgo.user_memory import BoundaryDetector
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.episode import EpisodeExtractor
from everalgo.user_memory.foresight import ForesightExtractor

_BOUNDARY_CONTINUE_JSON = '{"reasoning": "single coherent topic", "boundaries": [], "should_wait": false}'


def _two_msg_dialogue() -> list[ChatMessage]:
    """Reusable two-message dialogue suitable for all 3 downstream extractors."""
    return [
        ChatMessage(
            id="m1",
            role="user",
            content="Schedule a meeting with Alice at 3pm and I'll follow up next week.",
            timestamp=1700000000000,
            sender_id="u_test",
            sender_name="Alice",
        ),
        ChatMessage(
            id="m2",
            role="assistant",
            content="Done. Sent invite for 3pm. I'll keep an eye on the follow-up.",
            timestamp=1700000001000,
            sender_id="assistant",
        ),
    ]


async def test_boundary_to_episode_pipeline_e2e() -> None:
    """Boundary detects 1 MemCell, episode extracts 1 Episode with {title, content} shape."""
    call_count = 0

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            content = _BOUNDARY_CONTINUE_JSON
        else:
            content = (
                '{"title": "Meeting with Alice at 3pm",'
                ' "content": "User scheduled a meeting with Alice at 3pm and plans to follow up next week."}'
            )
        return ChatResponse(content=content, model="fake")

    fake = FakeLLMClient(handler=handler)
    output = await BoundaryDetector(llm=fake).adetect(_two_msg_dialogue(), is_final=True)
    assert output.tail == []
    assert len(output.cells) == 1
    mc = output.cells[0]

    episode = await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_test")

    ep = assert_episode_shape(episode)
    assert "Alice" in ep.episode
    assert ep.subject == "Meeting with Alice at 3pm"
    assert call_count == 2


async def test_boundary_to_foresight_pipeline_e2e() -> None:
    """Boundary detects 1 MemCell, foresight extracts forward-looking commitments (top-level JSON array)."""
    call_count = 0

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            content = _BOUNDARY_CONTINUE_JSON
        else:
            content = (
                '{"foresights": ['
                '{"content": "User will follow up on the Alice meeting next week",'
                ' "evidence": "I will follow up next week",'
                ' "start_time": "2024-01-01", "end_time": "2024-01-08", "duration_days": 7}'
                "]}"
            )
        return ChatResponse(content=content, model="fake")

    fake = FakeLLMClient(handler=handler)
    output = await BoundaryDetector(llm=fake).adetect(_two_msg_dialogue(), is_final=True)
    assert output.tail == []
    assert len(output.cells) == 1
    mc = output.cells[0]

    foresights = await ForesightExtractor(llm=fake).aextract(mc, sender_id="u_test")
    assert len(foresights) == 1

    fs = assert_foresight_shape(foresights[0])
    assert "follow up" in fs.foresight
    assert fs.start_time == "2024-01-01"
    assert fs.duration_days == 7
    assert call_count == 2


async def test_boundary_to_atomic_fact_pipeline_e2e() -> None:
    """Boundary detects 1 MemCell, atomic_fact splits ``atomic_facts.atomic_fact`` into entities."""
    call_count = 0

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            content = _BOUNDARY_CONTINUE_JSON
        else:
            content = (
                '{"atomic_facts": {'
                '"time": "March 14, 2024(Thursday) at 3:00 PM UTC", '
                '"atomic_fact": ['
                '"User scheduled a meeting with Alice at 3pm.",'
                '"Assistant sent the invite for 3pm."'
                "]}}"
            )
        return ChatResponse(content=content, model="fake")

    fake = FakeLLMClient(handler=handler)
    output = await BoundaryDetector(llm=fake).adetect(_two_msg_dialogue(), is_final=True)
    assert output.tail == []
    assert len(output.cells) == 1
    mc = output.cells[0]

    facts = await AtomicFactExtractor(llm=fake).aextract(mc, sender_id="u_test")
    assert len(facts) == 2

    for f in facts:
        assert_atomic_fact_shape(f)
    assert call_count == 2
