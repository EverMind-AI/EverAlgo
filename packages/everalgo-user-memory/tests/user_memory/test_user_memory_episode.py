"""Tests for everalgo.user_memory.episode — EpisodeExtractor (opensource port)."""

from __future__ import annotations

from typing import Any

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import MemCell, Message, MessageRole
from everalgo.user_memory.episode import (
    EpisodeExtractor,
    _derive_owner_id,
    _parse_llm_response,
    _render_conversation,
)


def _memcell() -> MemCell:
    """Helper: minimal MemCell with a 2-message dialogue and one participant."""
    msgs = [
        Message(
            role=MessageRole.USER,
            content="Schedule a meeting with Alice at 3pm",
            timestamp=1700000000000,
            sender_id="u_alice",
            sender_name="Alice",
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="Done. Sent invite for 3pm.",
            timestamp=1700000001000,
        ),
    ]
    return MemCell(
        event_id="mc_test_001",
        original_data=[{"message": m.model_dump(exclude_none=True)} for m in msgs],
        timestamp=1700000001000,
        participants=["u_alice"],
        sender_ids=["u_alice"],
    )


async def test_aextract_returns_episode_from_opensource_title_content_json() -> None:
    """LLM emits opensource ``{title, content}`` -> Episode with subject=title, episode=content."""
    llm_json = '{"title": "Meeting with Alice at 3pm", "content": "User scheduled a meeting with Alice at 3pm."}'
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    episodes = await EpisodeExtractor().aextract(_memcell(), llm=fake)

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep.subject == "Meeting with Alice at 3pm"
    assert "Alice" in ep.episode
    assert ep.timestamp == 1700000001000


async def test_aextract_auto_fills_parent_id_and_owner_id() -> None:
    """parent_id from memcell.event_id; owner_id derived from participants[0]."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"title": "T", "content": "c"}', model="fake")])
    mc = _memcell()

    episodes = await EpisodeExtractor().aextract(mc, llm=fake)

    assert episodes[0].parent_id == mc.event_id
    assert episodes[0].parent_type == "memcell"
    assert episodes[0].owner_id == "u_alice"


async def test_aextract_owner_id_falls_back_to_u_default_when_no_participants() -> None:
    """If MemCell has no participants and no sender_id, owner_id is 'u_default'."""
    msg = Message(role=MessageRole.USER, content="x", timestamp=1)
    mc = MemCell(
        event_id="mc_x",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1,
    )
    fake = FakeLLMClient(responses=[ChatResponse(content='{"title": "T", "content": "c"}', model="fake")])

    episodes = await EpisodeExtractor().aextract(mc, llm=fake)

    assert episodes[0].owner_id == "u_default"


async def test_aextract_raises_runtimeerror_when_content_missing_after_5_retries() -> None:
    """Empty content → ValueError → retry 5 times → RuntimeError (matches opensource line 287-298)."""
    import pytest

    bad = ChatResponse(content='{"title": "T", "content": ""}', model="fake")
    fake = FakeLLMClient(responses=[bad, bad, bad, bad, bad])

    with pytest.raises(RuntimeError, match="all 5 retries exhausted"):
        await EpisodeExtractor().aextract(_memcell(), llm=fake)
    assert fake.call_count == 5


async def test_aextract_retries_then_succeeds_on_late_valid_response() -> None:
    """4 garbage responses + 1 valid → 5 calls total, returns valid Episode."""
    bad = ChatResponse(content="not json", model="fake")
    good = ChatResponse(content='{"title": "OK", "content": "Final answer."}', model="fake")
    fake = FakeLLMClient(responses=[bad, bad, bad, bad, good])

    episodes = await EpisodeExtractor().aextract(_memcell(), llm=fake)

    assert len(episodes) == 1
    assert episodes[0].subject == "OK"
    assert fake.call_count == 5


async def test_aextract_per_call_prompt_overrides_default() -> None:
    """Per-call prompt= is what's rendered."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"title": "T", "content": "c"}', model="fake")

    fake = FakeLLMClient(handler=handler)
    custom = "CUSTOM EPISODE start={conversation_start_time} conv={conversation} cust={custom_instructions}"

    await EpisodeExtractor().aextract(_memcell(), llm=fake, prompt=custom)

    assert captured["content"].startswith("CUSTOM EPISODE")
    assert "Alice: Schedule a meeting" in captured["content"]


async def test_aextract_per_call_llm_overrides_default() -> None:
    """Per-call llm= is the one actually invoked."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["called"] = True
        return ChatResponse(content='{"title": "T", "content": "c"}', model="fake")

    fake = FakeLLMClient(handler=handler)

    await EpisodeExtractor().aextract(_memcell(), llm=fake)

    assert captured["called"] is True


# ==========================================================================
# Schema validation — missing title (line 85)
# ==========================================================================


async def test_aextract_raises_runtimeerror_when_title_missing_after_5_retries() -> None:
    """Empty title field → ValueError → retry 5 times → RuntimeError (line 85)."""
    import pytest

    bad = ChatResponse(content='{"title": "", "content": "c"}', model="fake")
    fake = FakeLLMClient(responses=[bad, bad, bad, bad, bad])

    with pytest.raises(RuntimeError, match="all 5 retries exhausted"):
        await EpisodeExtractor().aextract(_memcell(), llm=fake)
    assert fake.call_count == 5


# ==========================================================================
# _render_conversation skips empty content (line 137)
# ==========================================================================


def test_render_conversation_skips_empty_content() -> None:
    """Messages with empty content are silently dropped (line 137)."""
    real = Message(role=MessageRole.USER, content="hi", timestamp=1700000000000, sender_name="Alice")
    empty = Message(role=MessageRole.USER, content="", timestamp=1700000001000, sender_name="Bob")
    cell = MemCell(
        event_id="mc_render",
        original_data=[
            {"message": real.model_dump(exclude_none=True)},
            {"message": empty.model_dump(exclude_none=True)},
        ],
        timestamp=1700000001000,
    )
    rendered = _render_conversation(cell)
    assert "Alice: hi" in rendered
    assert "Bob" not in rendered


# ==========================================================================
# _parse_llm_response strategies — ```json fence + regex (lines 155-167)
# ==========================================================================


def test_parse_llm_response_handles_json_fence() -> None:
    """`````json ... ````` fenced response is tier-1 parsed (lines 155-161)."""
    raw = '```json\n{"title": "T", "content": "c"}\n```'
    parsed = _parse_llm_response(raw)
    assert parsed == {"title": "T", "content": "c"}


def test_parse_llm_response_falls_back_to_regex_embedded_object() -> None:
    """Prose surrounds an embedded ``{title...content}`` object → regex tier extracts it (lines 166-167)."""
    raw = 'Some preamble {"title": "T", "content": "c"} trailing text'
    parsed = _parse_llm_response(raw)
    assert parsed == {"title": "T", "content": "c"}


# ==========================================================================
# _derive_owner_id fallback to message sender_id (line 177)
# ==========================================================================


def test_derive_owner_id_falls_back_to_message_sender_id() -> None:
    """No participants → first message with sender_id wins (line 177)."""
    msg = Message(role=MessageRole.USER, content="x", timestamp=1, sender_id="u_from_msg")
    cell = MemCell(
        event_id="mc_x",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1,
    )
    assert _derive_owner_id(cell) == "u_from_msg"
