"""Tests for evercore.user_memory.episode — EpisodeExtractor."""

from __future__ import annotations

from typing import Any

from evercore.llm.types import ChatMessage as LLMChatMessage
from evercore.llm.types import ChatResponse
from evercore.testing.fake_llm import FakeLLMClient
from evercore.types import MemCell, Message, MessageRole
from evercore.user_memory.episode import EpisodeExtractor


def _memcell() -> MemCell:
    """Helper: build a minimal MemCell."""
    return MemCell(
        id="mc_test_001",
        messages=[
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
        ],
        timestamp=1700000001000,
    )


async def test_aextract_returns_episode_list_from_llm_json() -> None:
    """Valid LLM JSON yields a list[Episode] with all fields populated."""
    llm_json = (
        '{"episodes": [{"id": "ep_001", '
        '"owner_id": "u_alice", '
        '"episode": "User scheduled a meeting with Alice at 3pm.", '
        '"timestamp": 1700000000000}]}'
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    episodes = await EpisodeExtractor().aextract(_memcell(), llm=fake)

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.id == "ep_001"
    assert ep.owner_id == "u_alice"
    assert "Alice" in ep.episode
    assert ep.timestamp == 1700000000000


async def test_aextract_auto_fills_parent_id_from_memcell() -> None:
    """LLM-emitted JSON without parent_id gets parent_id from the source MemCell."""
    llm_json = '{"episodes": [{"id": "ep_002", "owner_id": "u_x", "episode": "x", "timestamp": 1700000000000}]}'
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])
    mc = _memcell()

    episodes = await EpisodeExtractor().aextract(mc, llm=fake)

    assert episodes[0].parent_id == mc.id
    assert episodes[0].parent_type == "memcell"


async def test_aextract_per_call_llm_overrides_default() -> None:
    """Per-call llm= argument is the one used by the extractor."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["called"] = True
        return ChatResponse(
            content=('{"episodes": [{"id": "ep_x", "owner_id": "u_x", "episode": "x", "timestamp": 1700000000000}]}'),
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)

    await EpisodeExtractor().aextract(_memcell(), llm=fake)

    assert captured["called"] is True
