"""Tests for everalgo.user_memory.profile — ProfileExtractor."""

from __future__ import annotations

from typing import Any

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import MemCell, Message, MessageRole
from everalgo.user_memory.profile import ProfileExtractor


def _memcell(idx: str = "mc_now", ts: int = 1700000010000, content: str = "Default content") -> MemCell:
    """Helper: build a minimal MemCell."""
    return MemCell(
        id=idx,
        messages=[
            Message(role=MessageRole.USER, content=content, timestamp=ts),
        ],
        timestamp=ts,
    )


def _cluster() -> list[MemCell]:
    """Helper: build a 2-cell prior cluster."""
    return [
        _memcell(idx="mc_old_1", ts=1690000000000, content="User asked about Python async patterns."),
        _memcell(idx="mc_old_2", ts=1695000000000, content="User mentioned they prefer ruff over black."),
    ]


async def test_aextract_returns_profile_with_required_fields() -> None:
    """Valid LLM JSON yields a Profile with id / owner_id / summary / timestamp."""
    llm_json = (
        '{"id": "pf_alice", "owner_id": "u_alice", '
        '"summary": "Alice is a Python developer who prefers ruff for linting.", '
        '"timestamp": 1700000010000}'
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    profile = await ProfileExtractor().aextract(
        _memcell(content="Alice discussing tooling preferences"),
        cluster_episodes=_cluster(),
        llm=fake,
    )

    assert profile.id == "pf_alice"
    assert profile.owner_id == "u_alice"
    assert "Python" in profile.summary
    assert profile.timestamp == 1700000010000


async def test_aextract_cluster_episodes_appear_in_prompt() -> None:
    """Cluster summaries are rendered into the prompt for the LLM."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(
            content=('{"id": "pf_x", "owner_id": "u_x", "summary": "s", "timestamp": 1700000000000}'),
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)

    await ProfileExtractor().aextract(_memcell(), cluster_episodes=_cluster(), llm=fake)

    assert "mc_old_1" in captured["content"]
    assert "mc_old_2" in captured["content"]
    assert "Python async patterns" in captured["content"]
    assert "ruff over black" in captured["content"]


async def test_aextract_empty_cluster_renders_no_prior_marker() -> None:
    """Empty cluster yields an explicit '(no prior MemCells ...)' marker in the prompt."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(
            content=('{"id": "pf_y", "owner_id": "u_y", "summary": "s", "timestamp": 1700000000000}'),
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)

    await ProfileExtractor().aextract(_memcell(), cluster_episodes=[], llm=fake)

    assert "(no prior MemCells" in captured["content"]


async def test_aextract_extra_fields_preserved_via_extra_allow() -> None:
    """LLM-emitted optional fields (interests / habits / ...) survive via extra='allow'."""
    llm_json = (
        '{"id": "pf_z", "owner_id": "u_z", '
        '"summary": "s", "timestamp": 1700000000000, '
        '"interests": ["python", "math"], '
        '"communication_style": "concise"}'
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    profile = await ProfileExtractor().aextract(
        _memcell(),
        cluster_episodes=_cluster(),
        llm=fake,
    )

    # extra='allow' surfaces these as model attributes
    assert profile.interests == ["python", "math"]  # type: ignore[attr-defined]
    assert profile.communication_style == "concise"  # type: ignore[attr-defined]


async def test_aextract_timestamp_falls_back_to_memcell_when_omitted() -> None:
    """If LLM omits timestamp, the source MemCell's timestamp is used."""
    llm_json = '{"id": "pf_t", "owner_id": "u_t", "summary": "s"}'
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])
    mc = _memcell(ts=1700000099999)

    profile = await ProfileExtractor().aextract(mc, cluster_episodes=[], llm=fake)

    assert profile.timestamp == 1700000099999


async def test_aextract_per_call_prompt_overrides_default() -> None:
    """Per-call prompt= argument is the rendered prompt, not the default."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(
            content=('{"id": "pf_w", "owner_id": "u_w", "summary": "s", "timestamp": 1700000000000}'),
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)
    custom = "CUSTOM PROFILE PROMPT current={current_memcell_text} cluster={cluster_summaries} ts={timestamp}"

    await ProfileExtractor().aextract(
        _memcell(),
        cluster_episodes=_cluster(),
        llm=fake,
        prompt=custom,
    )

    assert captured["content"].startswith("CUSTOM PROFILE PROMPT")
    assert "mc_old_1" in captured["content"]
