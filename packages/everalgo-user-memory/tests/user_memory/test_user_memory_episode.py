"""Tests for everalgo.user_memory.episode — EpisodeExtractor (single owner_id, no fan-out)."""

from __future__ import annotations

from typing import Any

import pytest

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.user_memory.episode import (
    EpisodeExtractor,
    _render_conversation,
    _resolve_user_name,
)


def _memcell() -> MemCell:
    """Helper: minimal MemCell with a 2-message dialogue (single sender u_alice)."""
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="Schedule a meeting with Alice at 3pm",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ChatMessage(
                id="m2",
                role="assistant",
                content="Done. Sent invite for 3pm.",
                timestamp=1700000001000,
                sender_id="assistant",
            ),
        ],
        timestamp=1700000001000,
    )


# ==========================================================================
# (1) Single successful extraction
# ==========================================================================


async def test_aextract_returns_single_episode_from_title_content_json() -> None:
    """LLM emits {title, content} -> single Episode with subject=title, episode=content."""
    llm_json = '{"title": "Meeting with Alice at 3pm", "content": "User scheduled a meeting with Alice at 3pm."}'
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    ep = await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert ep.subject == "Meeting with Alice at 3pm"
    assert "Alice" in ep.episode
    assert ep.timestamp == 1700000001000


async def test_aextract_episode_fields_filled_correctly() -> None:
    """owner_id on the returned Episode equals the supplied sender_id argument."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"title": "T", "content": "c"}', model="fake")])

    ep = await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert ep.owner_id == "u_alice"
    assert ep.subject == "T"
    assert ep.episode == "c"


async def test_aextract_owner_id_comes_from_argument_not_inferred() -> None:
    """owner_id on the returned Episode must equal the caller-supplied sender_id, not be inferred."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"title": "T", "content": "c"}', model="fake")])

    ep = await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_explicit")

    assert ep.owner_id == "u_explicit"


# ==========================================================================
# (2) LLMError propagates immediately (no retry)
# ==========================================================================


async def test_aextract_llm_error_propagates_immediately() -> None:
    """LLMError from the client propagates without swallowing or retrying."""
    from everalgo.llm.errors import LLMError

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        raise LLMError("upstream failure")

    fake = FakeLLMClient(handler=handler)

    with pytest.raises(LLMError, match="upstream failure"):
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert fake.call_count == 1  # exactly one attempt, no retry


# ==========================================================================
# (3) JSON parse failure propagates immediately (no retry)
# ==========================================================================


async def test_aextract_invalid_json_propagates_immediately() -> None:
    """Unparseable LLM response raises LLMError (via FakeLLMClient schema validation) after a single call."""
    from everalgo.llm.errors import LLMError

    bad = ChatResponse(content="not json at all", model="fake")
    fake = FakeLLMClient(responses=[bad])

    with pytest.raises(LLMError):
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert fake.call_count == 1


# ==========================================================================
# (6) Custom prompt override
# ==========================================================================


async def test_aextract_per_call_prompt_overrides_default() -> None:
    """Per-call prompt= is what's rendered; standard placeholders are substituted."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"title": "T", "content": "c"}', model="fake")

    fake = FakeLLMClient(handler=handler)
    custom = "CUSTOM EPISODE start={conversation_start_time} conv={conversation} cust={custom_instructions}"

    await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice", prompt=custom)

    assert captured["content"].startswith("CUSTOM EPISODE")
    assert "Alice: Schedule a meeting" in captured["content"]


# ==========================================================================
# (7) custom_instructions override
# ==========================================================================


async def test_aextract_custom_instructions_override() -> None:
    """custom_instructions= replaces the default block in the rendered prompt."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"title": "T", "content": "c"}', model="fake")

    fake = FakeLLMClient(handler=handler)
    custom_prompt = "instruct={custom_instructions}"

    await EpisodeExtractor(llm=fake).aextract(
        _memcell(), sender_id="u_alice", prompt=custom_prompt, custom_instructions="MY_CUSTOM_INSTR"
    )

    assert "MY_CUSTOM_INSTR" in captured["content"]


# ==========================================================================
# (8) owner_id from argument, not inferred — user_name resolved from messages
# ==========================================================================


async def test_aextract_user_name_resolved_from_messages() -> None:
    """user_name in the prompt is taken from the matching message's sender_name, not owner_id literal."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"title": "T", "content": "c"}', model="fake")

    fake = FakeLLMClient(handler=handler)
    custom_prompt = "user_name={user_name}"

    await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice", prompt=custom_prompt)

    # u_alice has sender_name="Alice" in _memcell()
    assert "Alice" in captured["content"]


async def test_aextract_user_name_falls_back_to_sender_id_when_no_sender_name() -> None:
    """When no message carries sender_name for sender_id, user_name falls back to the sender_id string."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"title": "T", "content": "c"}', model="fake")

    mc = MemCell(
        items=[
            ChatMessage(
                id="m20",
                role="user",
                content="hello",
                timestamp=1700000000000,
                sender_id="u_ghost",
                sender_name=None,
            ),
        ],
        timestamp=1700000000000,
    )
    fake = FakeLLMClient(handler=handler)
    custom_prompt = "user_name={user_name}"

    await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_ghost", prompt=custom_prompt)

    assert "u_ghost" in captured["content"]


# ==========================================================================
# Two independent owner_id calls on the same memcell (replaces fan-out test)
# ==========================================================================


async def test_aextract_two_owner_ids_produce_independent_episodes() -> None:
    """Calling aextract twice (once per owner_id) yields two independent Episodes."""
    mc = MemCell(
        items=[
            ChatMessage(
                id="m10",
                role="user",
                content="Let's plan the offsite",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ChatMessage(
                id="m11",
                role="user",
                content="I can host on Friday",
                timestamp=1700000001000,
                sender_id="u_bob",
                sender_name="Bob",
            ),
            ChatMessage(
                id="m12",
                role="assistant",
                content="I'll send invites.",
                timestamp=1700000002000,
                sender_id="assistant",
            ),
        ],
        timestamp=1700000002000,
    )

    fake_alice = FakeLLMClient(
        responses=[
            ChatResponse(content='{"title": "Alice offsite", "content": "Alice proposed offsite."}', model="fake")
        ]
    )
    fake_bob = FakeLLMClient(
        responses=[ChatResponse(content='{"title": "Bob offsite", "content": "Bob offered to host."}', model="fake")]
    )

    ep_alice = await EpisodeExtractor(llm=fake_alice).aextract(mc, sender_id="u_alice")
    ep_bob = await EpisodeExtractor(llm=fake_bob).aextract(mc, sender_id="u_bob")

    assert ep_alice.owner_id == "u_alice"
    assert ep_bob.owner_id == "u_bob"
    assert ep_alice.subject == "Alice offsite"
    assert ep_bob.subject == "Bob offsite"
    assert fake_alice.call_count == 1
    assert fake_bob.call_count == 1


# ==========================================================================
# _render_conversation skips empty content
# ==========================================================================


def test_render_conversation_skips_empty_content() -> None:
    """Messages with empty content are silently dropped."""
    cell = MemCell(
        items=[
            ChatMessage(
                id="m1", role="user", content="hi", timestamp=1700000000000, sender_id="u_alice", sender_name="Alice"
            ),
            ChatMessage(
                id="m2", role="user", content="", timestamp=1700000001000, sender_id="u_bob", sender_name="Bob"
            ),
        ],
        timestamp=1700000001000,
    )
    rendered = _render_conversation(cell)
    assert "Alice: hi" in rendered
    assert "Bob" not in rendered


# ==========================================================================
# _resolve_user_name
# ==========================================================================


def test_resolve_user_name_returns_sender_name_when_present() -> None:
    """_resolve_user_name returns the matching sender_name from messages."""
    cell = MemCell(
        items=[ChatMessage(id="m1", role="user", content="x", timestamp=1, sender_id="u_alice", sender_name="Alice")],
        timestamp=1,
    )
    assert _resolve_user_name(cell, "u_alice") == "Alice"


def test_resolve_user_name_falls_back_to_owner_id_when_no_match() -> None:
    """_resolve_user_name falls back to the owner_id literal when no message has a matching sender_name."""
    cell = MemCell(
        items=[ChatMessage(id="m1", role="user", content="x", timestamp=1, sender_id="u_other", sender_name="Other")],
        timestamp=1,
    )
    assert _resolve_user_name(cell, "u_missing") == "u_missing"


# ==========================================================================
# Silent-skip contract — agent → user-memory pipeline
# ==========================================================================


async def test_aextract_silently_skips_non_chat_items() -> None:
    """EpisodeExtractor must silently skip ToolCallRequest / ToolCallResult items.

    Locks the agent → user-memory pipeline contract: a MemCell with mixed items (ChatMessage +
    tool calls) must produce the same Episode as a chat-only MemCell with the same ChatMessages.
    """
    chat_only_cell = MemCell(
        items=[
            ChatMessage(
                id="c1",
                role="user",
                content="Let's schedule the demo for Friday",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            ),
        ],
        timestamp=1700000000000,
    )
    mixed_cell = MemCell(
        items=[
            ChatMessage(
                id="c1",
                role="user",
                content="Let's schedule the demo for Friday",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ToolCallRequest(
                tool_calls=[
                    ToolCall(
                        id="tc1", function=ToolCallFunction(name="calendar.create", arguments='{"time": "Friday"}')
                    )
                ],
                timestamp=1700000001000,
                sender_id="assistant",
            ),
            ToolCallResult(
                tool_call_id="tc1",
                content="Event created.",
                timestamp=1700000002000,
            ),
        ],
        timestamp=1700000002000,
    )

    llm_json = '{"title": "Demo on Friday", "content": "Alice scheduled a demo for Friday."}'

    fake_chat = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])
    fake_mixed = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    ep_chat = await EpisodeExtractor(llm=fake_chat).aextract(chat_only_cell, sender_id="u_alice")
    ep_mixed = await EpisodeExtractor(llm=fake_mixed).aextract(mixed_cell, sender_id="u_alice")

    assert ep_chat.subject == ep_mixed.subject
    assert ep_chat.episode == ep_mixed.episode
    assert ep_chat.owner_id == ep_mixed.owner_id


# ==========================================================================
# (12) Instance-level llm= binding
# ==========================================================================


async def test_aextract_uses_instance_llm_when_per_call_omitted() -> None:
    """Instance-level llm= is used when aextract() is called without a per-call llm= argument."""
    instance_fake = FakeLLMClient(
        responses=[ChatResponse(content='{"title": "Instance test", "content": "From instance llm"}', model="inst")]
    )
    extractor = EpisodeExtractor(llm=instance_fake)
    ep = await extractor.aextract(_memcell(), sender_id="u_alice")
    assert ep.subject == "Instance test"
    assert instance_fake.call_count == 1


@pytest.mark.asyncio
async def test_extract_generic_when_sender_id_is_none() -> None:
    """sender_id=None → generic prompt (EPISODE_GENERATION_PROMPT, no user_name), owner_id=None."""
    fake = FakeLLMClient(
        responses=['{"title": "Bug fix discussion", "content": "Alice and Bob debugged the login flow."}']
    )
    ep = await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id=None)
    assert ep.owner_id is None
    assert ep.subject == "Bug fix discussion"
    rendered = fake.calls[0].messages[0].content
    # Generic prompt: must NOT contain user_name injection.
    assert "User name:" not in rendered
    # Generic prompt header signature.
    assert "You are an episodic memory generation expert" in rendered
