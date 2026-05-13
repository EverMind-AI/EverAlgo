"""End-to-end pipeline tests: messages → boundary → user-memory extractors.

Verifies the full boundary→{episode, foresight, atomic_fact} data flow with a FakeLLMClient handler that
returns distinct JSON per call (boundary call returns split decision, downstream call returns extractor
JSON). Each pipeline file is the cross-distribution acceptance test for one extractor on the EPISODE path
(per `tests/README.md`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import everalgo.llm
from everalgo.boundary.chat import ChatMemCellExtractor
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.assertions import (
    assert_atomic_fact_shape,
    assert_episode_shape,
    assert_foresight_shape,
)
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Message, MessageRole
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.episode import EpisodeExtractor
from everalgo.user_memory.foresight import ForesightExtractor

pytestmark = pytest.mark.skip(reason="boundary.chat.adetect stub — full implementation pending")

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def reset_everalgo_llm_state() -> Iterator[None]:
    """Reset everalgo.llm._default + _active per test.

    Without this, test pollution between e2e and other test files could leak global LLM state.
    """
    saved_default = everalgo.llm._default
    token = everalgo.llm._active.set(None)
    try:
        everalgo.llm._default = None
        yield
    finally:
        everalgo.llm._default = saved_default
        everalgo.llm._active.reset(token)


def _two_msg_dialogue() -> list[Message]:
    """Reusable two-message dialogue suitable for all 3 downstream extractors."""
    return [
        Message(
            role=MessageRole.USER,
            content="Schedule a meeting with Alice at 3pm and I'll follow up next week.",
            timestamp=1700000000000,
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="Done. Sent invite for 3pm. I'll keep an eye on the follow-up.",
            timestamp=1700000001000,
        ),
    ]


async def test_boundary_to_episode_pipeline_e2e() -> None:
    """Boundary detects 1 MemCell, episode extracts 1 Episode from it.

    Uses FakeLLMClient handler mode to return distinct JSON per call:
    - Call 1 (boundary detect): {"split_at": null} (no split)
    - Call 2 (episode extract): episode JSON

    Verifies the full integration of types, LLM stack, prompts, 3-layer resolve, FakeLLMClient, and the
    structural assertion helper.
    """
    call_count = 0

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            content = '{"split_at": null}'
        else:
            content = (
                '{"episodes": [{"id": "ep_test_001", '
                '"owner_id": "u_test", '
                '"episode": "User scheduled a meeting with Alice at 3pm.", '
                '"timestamp": 1700000000000}]}'
            )
        return ChatResponse(content=content, model="fake")

    fake = FakeLLMClient(handler=handler)
    memcells, _tail = await ChatMemCellExtractor().adetect(_two_msg_dialogue(), llm=fake, is_final=True)
    assert len(memcells) == 1
    mc = memcells[0]

    episodes = await EpisodeExtractor().aextract(mc, llm=fake)
    assert len(episodes) == 1

    ep = assert_episode_shape(episodes[0])
    assert ep.parent_id == mc.id
    assert ep.parent_type == "memcell"
    assert "Alice" in ep.episode
    assert call_count == 2


async def test_boundary_to_foresight_pipeline_e2e() -> None:
    """Boundary detects 1 MemCell, foresight extracts forward-looking commitments from it."""
    call_count = 0

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            content = '{"split_at": null}'
        else:
            content = (
                '{"foresights": [{"id": "fs_test_001", '
                '"owner_id": "u_test", '
                '"foresight": "User will follow up on the Alice meeting next week.", '
                '"evidence": "I\'ll follow up next week.", '
                '"timestamp": 1700000000000}]}'
            )
        return ChatResponse(content=content, model="fake")

    fake = FakeLLMClient(handler=handler)
    memcells, _tail = await ChatMemCellExtractor().adetect(_two_msg_dialogue(), llm=fake, is_final=True)
    assert len(memcells) == 1
    mc = memcells[0]

    foresights = await ForesightExtractor().aextract(mc, llm=fake)
    assert len(foresights) == 1

    fs = assert_foresight_shape(foresights[0])
    assert fs.parent_id == mc.id
    assert fs.parent_type == "memcell"
    assert "follow up" in fs.foresight
    assert "follow up" in fs.evidence
    assert call_count == 2


async def test_boundary_to_atomic_fact_pipeline_e2e() -> None:
    """Boundary detects 1 MemCell, atomic_fact extracts verifiable assertions from it."""
    call_count = 0

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            content = '{"split_at": null}'
        else:
            content = (
                '{"atomic_facts": ['
                '{"id": "af_test_001", "owner_id": "u_test", '
                '"fact": "User scheduled a meeting with Alice at 3pm.", '
                '"timestamp": 1700000000000},'
                '{"id": "af_test_002", "owner_id": "u_test", '
                '"fact": "Assistant sent the invite for 3pm.", '
                '"timestamp": 1700000001000}'
                "]}"
            )
        return ChatResponse(content=content, model="fake")

    fake = FakeLLMClient(handler=handler)
    memcells, _tail = await ChatMemCellExtractor().adetect(_two_msg_dialogue(), llm=fake, is_final=True)
    assert len(memcells) == 1
    mc = memcells[0]

    facts = await AtomicFactExtractor().aextract(mc, llm=fake)
    assert len(facts) == 2

    for f in facts:
        af = assert_atomic_fact_shape(f)
        assert af.parent_id == mc.id
        assert af.parent_type == "memcell"
    assert call_count == 2
