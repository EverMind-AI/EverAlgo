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
    """parent_id from memcell.event_id; owner_id directly taken from the per-sender iteration."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"title": "T", "content": "c"}', model="fake")])
    mc = _memcell()

    episodes = await EpisodeExtractor().aextract(mc, llm=fake)

    assert episodes[0].parent_id == mc.event_id
    assert episodes[0].parent_type == "memcell"
    assert episodes[0].owner_id == "u_alice"


async def test_aextract_owner_id_falls_back_to_u_default_when_no_participants() -> None:
    """Empty sender_ids → group fallback. With no participants and no sender_id, owner_id is 'u_default'."""
    msg = Message(role=MessageRole.USER, content="x", timestamp=1)
    mc = MemCell(
        event_id="mc_x",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1,
    )
    fake = FakeLLMClient(responses=[ChatResponse(content='{"title": "T", "content": "c"}', model="fake")])

    episodes = await EpisodeExtractor().aextract(mc, llm=fake)

    assert episodes[0].owner_id == "u_default"
    assert fake.call_count == 1  # group fallback issues a single LLM call


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


# ==========================================================================
# Per-sender fan-out — main new behaviour (opensource memory_manager loops over senders;
# EverAlgo internalises the loop inside the extractor).
# ==========================================================================


def _multi_sender_memcell() -> MemCell:
    """Three-message MemCell with two distinct senders u_alice and u_bob."""
    msgs = [
        Message(
            role=MessageRole.USER,
            content="Let's plan the offsite together",
            timestamp=1700000000000,
            sender_id="u_alice",
            sender_name="Alice",
        ),
        Message(
            role=MessageRole.USER,
            content="Sounds good — I can host on Friday",
            timestamp=1700000001000,
            sender_id="u_bob",
            sender_name="Bob",
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="I'll send invites to both of you.",
            timestamp=1700000002000,
        ),
    ]
    return MemCell(
        event_id="mc_multi_001",
        original_data=[{"message": m.model_dump(exclude_none=True)} for m in msgs],
        timestamp=1700000002000,
        participants=["u_alice", "u_bob"],
        sender_ids=["u_alice", "u_bob"],
    )


async def test_aextract_produces_one_episode_per_sender_id_with_matching_owner_ids() -> None:
    """N sender_ids → N LLM calls → N episodes; owner_ids match sender_ids in order."""
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content='{"title": "Alice offsite plan", "content": "Alice proposed the offsite."}', model="fake"
            ),
            ChatResponse(content='{"title": "Bob offsite plan", "content": "Bob offered to host."}', model="fake"),
        ]
    )

    episodes = await EpisodeExtractor().aextract(_multi_sender_memcell(), llm=fake)

    assert len(episodes) == 2
    assert fake.call_count == 2
    assert [ep.owner_id for ep in episodes] == ["u_alice", "u_bob"]
    assert episodes[0].subject == "Alice offsite plan"
    assert episodes[1].subject == "Bob offsite plan"


async def test_aextract_per_sender_substitutes_user_name_in_prompt() -> None:
    """The personal prompt receives the resolved sender_name for each sender_id."""
    captured: list[str] = []

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured.append(messages[0].content)
        return ChatResponse(content='{"title": "T", "content": "c"}', model="fake")

    fake = FakeLLMClient(handler=handler)
    custom = "user_name={user_name} conv={conversation}"

    await EpisodeExtractor().aextract(_multi_sender_memcell(), llm=fake, prompt=custom)

    assert len(captured) == 2
    assert captured[0].startswith("user_name=Alice ")
    assert captured[1].startswith("user_name=Bob ")


async def test_aextract_per_sender_user_name_falls_back_to_sender_id_when_map_misses() -> None:
    """sender_id absent from sender_name map (e.g., no matching message) → user_name = sender_id string."""
    captured: list[str] = []

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured.append(messages[0].content)
        return ChatResponse(content='{"title": "T", "content": "c"}', model="fake")

    # MemCell whose sender_ids references a user id that never appears in any message.
    msg = Message(
        role=MessageRole.USER,
        content="hello",
        timestamp=1700000000000,
        sender_id="u_alice",
        sender_name="Alice",
    )
    mc = MemCell(
        event_id="mc_miss",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1700000000000,
        participants=["u_alice", "u_ghost"],
        sender_ids=["u_alice", "u_ghost"],
    )
    fake = FakeLLMClient(handler=handler)
    custom = "user_name={user_name}"

    episodes = await EpisodeExtractor().aextract(mc, llm=fake, prompt=custom)

    assert [ep.owner_id for ep in episodes] == ["u_alice", "u_ghost"]
    assert captured[0] == "user_name=Alice"
    assert captured[1] == "user_name=u_ghost"  # opensource line 261 parity: map.get(uid, uid)


async def test_aextract_mid_loop_retry_exhaustion_raises_runtimeerror() -> None:
    """First sender succeeds, second sender exhausts 5 retries → RuntimeError; no partial result returned."""
    import pytest

    good = ChatResponse(content='{"title": "Alice", "content": "ok"}', model="fake")
    bad = ChatResponse(content="garbage", model="fake")
    fake = FakeLLMClient(responses=[good, bad, bad, bad, bad, bad])

    with pytest.raises(RuntimeError, match="all 5 retries exhausted"):
        await EpisodeExtractor().aextract(_multi_sender_memcell(), llm=fake)
    assert fake.call_count == 6  # 1 success for alice + 5 retries for bob
