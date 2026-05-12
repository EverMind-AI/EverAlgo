"""End-to-end pipeline test: messages → boundary → episode.

Verifies the full boundary→episode data flow with a FakeLLMClient handler that returns distinct JSON per
call (boundary call returns split decision, episode call returns episode JSON). This is the sub-project 4
reference implementation acceptance test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import everalgo.llm
from everalgo.boundary.chat import ChatMemCellExtractor
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.assertions import assert_episode_shape
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Message, MessageRole
from everalgo.user_memory.episode import EpisodeExtractor

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def reset_everalgo_llm_state() -> Iterator[None]:
    """Reset everalgo.llm._default + _active per test (sub-project 2.5 fixture).

    Without this, test pollution between e2e and other test files could leak global state.
    """
    saved_default = everalgo.llm._default
    token = everalgo.llm._active.set(None)
    try:
        everalgo.llm._default = None
        yield
    finally:
        everalgo.llm._default = saved_default
        everalgo.llm._active.reset(token)


async def test_boundary_to_episode_pipeline_e2e() -> None:
    """Boundary detects 1 MemCell, episode extracts 1 Episode from it.

    Uses FakeLLMClient handler mode to return distinct JSON per call:
    - Call 1 (boundary detect): {"split_at": null} (no split)
    - Call 2 (episode extract): episode JSON

    Verifies:
    1. Both extractors run without errors (full integration of types,
       LLM stack, prompts, sub-project 2.5 resolve, sub-project 3
       FakeLLMClient + assert_episode_shape).
    2. parent_id flows from MemCell.id to Episode.parent_id.
    3. assert_episode_shape passes (sub-project 3 helper).
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
    msgs = [
        Message(
            role=MessageRole.USER,
            content="Schedule a meeting with Alice at 3pm",
            timestamp=1700000000000,
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="Done. Sent invite for 3pm.",
            timestamp=1700000001000,
        ),
    ]

    memcells = await ChatMemCellExtractor().adetect(msgs, llm=fake)
    assert len(memcells) == 1
    mc = memcells[0]

    episodes = await EpisodeExtractor().aextract(mc, llm=fake)
    assert len(episodes) == 1

    ep = assert_episode_shape(episodes[0])
    assert ep.parent_id == mc.id
    assert ep.parent_type == "memcell"
    assert "Alice" in ep.episode
    assert call_count == 2
