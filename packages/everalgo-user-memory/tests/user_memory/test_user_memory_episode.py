"""Tests for everalgo.user_memory.episode — EpisodeExtractor (single owner_id, no fan-out)."""

from __future__ import annotations

import re
from typing import Any

import pytest

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.user_memory.episode import (
    EpisodeExtractor,
    _format_prompt_time,
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
# Language rules — both variants, both languages, mixed-input clauses
# ==========================================================================

_MIXED_INPUT_CLAUSES_EN = (
    "themselves compose",  # judgement source restricted to participants' own writing
    "dominates the conversation by volume",  # long quoted material must not flip the judgement
    # An operational test for "is this pasted" — the negative instruction alone left the model unable to
    # recognise unmarked prose as pasted, which cost ~35% of judgements on that shape of input.
    "Apply this test to decide what is pasted",
    "whether or not it is wrapped in quotation marks or a code fence",
    "sentence structure",  # embedded foreign terms do not flip the judgement
    "keep their original form",  # proper nouns / technical terms stay untranslated
)


def test_en_generic_variant_has_language_rule() -> None:
    """The generic variant had no language rule at all — that was the main source of mixed output."""
    from everalgo.user_memory.prompts.en.episode import EPISODE_GENERATION_PROMPT

    assert "CRITICAL LANGUAGE RULE" in EPISODE_GENERATION_PROMPT


def test_en_generic_variant_states_language_rule_at_both_ends() -> None:
    """Long prompts lose middle instructions; the rule is repeated at head and tail."""
    from everalgo.user_memory.prompts.en.episode import EPISODE_GENERATION_PROMPT

    assert EPISODE_GENERATION_PROMPT.count("CRITICAL LANGUAGE RULE") == 2


@pytest.mark.parametrize("clause", _MIXED_INPUT_CLAUSES_EN)
def test_en_generic_variant_covers_mixed_input(clause: str) -> None:
    """Chinese question plus long English pasted material must still yield Chinese output."""
    from everalgo.user_memory.prompts.en.episode import EPISODE_GENERATION_PROMPT

    assert clause in EPISODE_GENERATION_PROMPT


@pytest.mark.parametrize("clause", _MIXED_INPUT_CLAUSES_EN)
def test_en_user_variant_covers_mixed_input(clause: str) -> None:
    """The user variant already had a rule, but it never defined what to judge the language from."""
    from everalgo.user_memory.prompts.en.episode import USER_EPISODE_GENERATION_PROMPT

    assert clause in USER_EPISODE_GENERATION_PROMPT


# zh must not rot relative to en — it is a public prompt selectable via `prompt=` (see README.md).
# Each substring sits after the subject noun phrase in the mixed-input judgement clause, so it is
# invariant across both the generic ("对话参与者") and user (`` `{user_name}` ``) variants.
_MIXED_INPUT_CLAUSES_ZH = (
    "本人撰写的内容",  # judgement source restricted to participants' own writing
    "在篇幅上占据对话主体",  # long quoted material must not flip the judgement
    "判断何为粘贴材料时适用以下检验",  # operational test, mirrors the en clause above
    "也无论是否被引号或代码块包裹",
    "句子结构",  # embedded foreign terms do not flip the judgement
    "保留原文形式",  # proper nouns / technical terms stay untranslated
)


@pytest.mark.parametrize("clause", _MIXED_INPUT_CLAUSES_ZH)
def test_zh_generic_variant_covers_mixed_input(clause: str) -> None:
    """The zh prompt must not rot relative to en — it is a public prompt selectable via `prompt=`."""
    from everalgo.user_memory.prompts.zh.episode import EPISODE_GENERATION_PROMPT

    assert clause in EPISODE_GENERATION_PROMPT


@pytest.mark.parametrize("clause", _MIXED_INPUT_CLAUSES_ZH)
def test_zh_user_variant_covers_mixed_input(clause: str) -> None:
    """Same mixed-input clauses as the generic variant."""
    from everalgo.user_memory.prompts.zh.episode import USER_EPISODE_GENERATION_PROMPT

    assert clause in USER_EPISODE_GENERATION_PROMPT


def test_zh_and_en_have_same_language_rule_count() -> None:
    """Structural parity guard: each variant states its rule at head and tail in both languages."""
    from everalgo.user_memory.prompts.en.episode import EPISODE_GENERATION_PROMPT as EN_GENERIC
    from everalgo.user_memory.prompts.en.episode import USER_EPISODE_GENERATION_PROMPT as EN_USER
    from everalgo.user_memory.prompts.zh.episode import EPISODE_GENERATION_PROMPT as ZH_GENERIC
    from everalgo.user_memory.prompts.zh.episode import USER_EPISODE_GENERATION_PROMPT as ZH_USER

    assert EN_GENERIC.count("CRITICAL LANGUAGE RULE") == ZH_GENERIC.count("关键语言规则") == 2
    assert EN_USER.count("CRITICAL LANGUAGE RULE") == ZH_USER.count("关键语言规则") == 2


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
    assert ep.episode.endswith("c")


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
# (3) JSON parse failure: ValueError on first attempt (no internal retry)
# ==========================================================================


async def test_aextract_invalid_json_raises_value_error() -> None:
    """Unparseable LLM response raises ValueError immediately; retry is caller's job."""
    bad_responses: list[str | ChatResponse] = [ChatResponse(content="not json at all", model="fake")]
    fake = FakeLLMClient(responses=bad_responses)

    with pytest.raises(ValueError):
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert fake.call_count == 1


async def test_aextract_raises_on_empty_content() -> None:
    """An empty ``content`` must fail loud at the extractor, not travel downstream as an empty episode.

    A truncated or empty structured-output response is an extraction failure, and the caller learns
    far more from a ``ValueError`` here than from an empty-body guard firing later in
    ``assert_episode_shape`` or the benchmark extract stage, where the originating call is long gone.
    """
    fake = FakeLLMClient(responses=[ChatResponse(content='{"title": "T", "content": ""}', model="fake")])

    with pytest.raises(ValueError, match="empty content"):
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")


async def test_aextract_raises_on_whitespace_only_content() -> None:
    """Whitespace-only ``content`` must be treated the same as empty ``content``."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"title": "T", "content": "   "}', model="fake")])

    with pytest.raises(ValueError, match="empty content"):
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")


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
    # pseudo-JSON format: speaker and content are unquoted values
    assert "Alice" in captured["content"]
    assert "Schedule a meeting" in captured["content"]


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
    """Messages with empty content are silently dropped; pseudo-JSON format used."""
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
    # pseudo-JSON format: "speaker": Alice (unquoted value) and "content": hi (unquoted value)
    assert "Alice" in rendered
    assert "hi" in rendered
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


# ==========================================================================
# Time formatting helpers — input side keeps the weekday, output side does not
# ==========================================================================

# 2026-05-29 12:25:10 UTC, a Friday.
_TS_FRIDAY_MS = 1780057510000


def test_format_prompt_time_is_24h_with_weekday() -> None:
    """Input-side format: 24-hour clock, explicit UTC, weekday retained for relative-time reasoning."""
    assert _format_prompt_time(_TS_FRIDAY_MS) == "2026-05-29 12:25 UTC (Friday)"


def test_format_prompt_time_has_no_am_pm_marker() -> None:
    """A 12-hour clock makes noon/midnight ambiguous for the LLM; 24-hour clock removes that."""
    rendered = _format_prompt_time(_TS_FRIDAY_MS)
    assert "AM" not in rendered
    assert "PM" not in rendered


def test_format_prompt_time_keeps_weekday_for_relative_reasoning() -> None:
    """Regression guard: the weekday label is what lets the LLM resolve "last Friday".

    Dropping it re-introduces the off-by-one-week errors fixed in user-memory 0.3.1 (commit d9fe22e).
    """
    assert "(Friday)" in _format_prompt_time(_TS_FRIDAY_MS)


# ==========================================================================
# Prompt injection uses the 24-hour input-side format
# ==========================================================================


def test_render_conversation_uses_24h_input_format() -> None:
    """Per-message timestamps in the conversation block use the input-side format."""
    mc = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hello",
                timestamp=_TS_FRIDAY_MS,
                sender_id="u_alice",
                sender_name="Alice",
            )
        ],
        timestamp=_TS_FRIDAY_MS,
    )

    rendered = _render_conversation(mc)

    assert "2026-05-29 12:25 UTC (Friday)" in rendered
    assert "PM" not in rendered


async def test_conversation_start_time_uses_24h_input_format() -> None:
    """The {conversation_start_time} slot is filled with the input-side format."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"title": "T", "content": "c"}', model="fake")

    fake = FakeLLMClient(handler=handler)
    mc = MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="hello",
                timestamp=_TS_FRIDAY_MS,
                sender_id="u_alice",
                sender_name="Alice",
            )
        ],
        timestamp=_TS_FRIDAY_MS,
    )

    await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_alice", prompt="start={conversation_start_time}")

    assert captured["content"] == "start=2026-05-29 12:25 UTC (Friday)"


# ==========================================================================
# Episode body is stored verbatim; times come from the prompt, not from code
# ==========================================================================


def _memcell_spanning_70_minutes() -> MemCell:
    """MemCell whose first and closing item differ, to pin down which timestamp each field uses."""
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="start",
                timestamp=_TS_FRIDAY_MS,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ChatMessage(
                id="m2",
                role="assistant",
                content="end",
                timestamp=_TS_FRIDAY_MS + 70 * 60 * 1000,
                sender_id="assistant",
            ),
        ],
        timestamp=_TS_FRIDAY_MS + 70 * 60 * 1000,
    )


@pytest.mark.parametrize("sender_id", ["u_alice", None])
async def test_episode_body_is_stored_verbatim(sender_id: str | None) -> None:
    """Both prompt variants store ``content`` exactly as the LLM returned it."""
    fake = FakeLLMClient(
        responses=[ChatResponse(content='{"title": "T", "content": "Alice asked about hiking."}', model="fake")]
    )

    ep = await EpisodeExtractor(llm=fake).aextract(_memcell_spanning_70_minutes(), sender_id=sender_id)

    assert ep.episode == "Alice asked about hiking."


async def test_episode_body_gets_no_code_built_timestamp_prefix() -> None:
    """Code must not prepend a timestamp — the prompt owns the narrative's times.

    A prefix was tried and removed: its value was ``items[0].timestamp``, the same instant the
    narrative's own timeline opens on, so it restated the opening message's time a second time
    (``2026-05-29 12:25 UTC — 2026-05-29 12:25 UTC, Alice…``). A real-LLM run over 35 extractions
    found the model always wrote an absolute UTC time of its own, including for single-message
    conversations with no temporal content, so the prefix guarded a case that did not occur. The
    format the prefix guaranteed is now guaranteed by the prompt's UTC rule instead.
    """
    fake = FakeLLMClient(responses=[ChatResponse(content='{"title": "T", "content": "body"}', model="fake")])

    ep = await EpisodeExtractor(llm=fake).aextract(_memcell_spanning_70_minutes(), sender_id="u_alice")

    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC — ", ep.episode) is None


async def test_episode_timestamp_field_still_holds_closing_time() -> None:
    """Dropping the prefix must not disturb the ``timestamp`` field: still ``memcell.timestamp``."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"title": "T", "content": "body"}', model="fake")])
    mc = _memcell_spanning_70_minutes()

    ep = await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_alice")

    assert ep.timestamp == mc.timestamp == _TS_FRIDAY_MS + 70 * 60 * 1000


async def test_episode_summary_fallback_slices_verbatim_body() -> None:
    """When the LLM omits `summary`, the fallback slices the body, which no longer carries a prefix."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"title": "T", "content": "body"}', model="fake")])

    ep = await EpisodeExtractor(llm=fake).aextract(_memcell_spanning_70_minutes(), sender_id="u_alice")

    assert ep.model_dump()["summary"] == "body"


# ==========================================================================
# Date-related prompt text
# ==========================================================================

_MONTH_NAME_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}\b"
)
_LEADING_TIMESTAMP_RE = re.compile(r"^(?:\d|(?:On|At|In)\s+(?:\S+\s+){0,3}?\d)")


def _extract_example_field(prompt_text: str, heading: str, field: str) -> str:
    """Extract a `"field": "..."` value from the JSON block following `heading` in the worked example."""
    section = prompt_text.split(heading, 1)[1]
    match = re.search(rf'"{field}":\s*"([^"]*)"', section)
    assert match is not None, f"could not find {field!r} field after {heading!r}"
    return match.group(1)


def test_en_dual_format_examples_use_iso_dates() -> None:
    """The dual-format rule used to show three different date shapes; unify on YYYY-MM-DD."""
    from everalgo.user_memory.prompts.en.episode import EPISODE_GENERATION_PROMPT

    assert "(2023-07-21)" in EPISODE_GENERATION_PROMPT
    assert "July 21, 2023" not in EPISODE_GENERATION_PROMPT


def test_en_generic_example_body_has_no_timestamp_prefix() -> None:
    """The generic example was what taught the model to open the body with a date.

    Guards the invariant (no leading digit, no "On/At/In <date>" opening), not just the one
    historical string, so reintroducing ANY timestamp-shaped opening fails this test.
    """
    from everalgo.user_memory.prompts.en.episode import EPISODE_GENERATION_PROMPT

    content = _extract_example_field(EPISODE_GENERATION_PROMPT, "If the conversation start time is", "content")
    assert not content[0].isdigit()
    assert _LEADING_TIMESTAMP_RE.match(content) is None


def test_en_generic_example_title_uses_iso_date() -> None:
    """Generic title example used `March 14, 2024` while the user variant used `2024-03-14`.

    The negative half guards against ANY month-name date shape being reintroduced, not just the
    one historical string.
    """
    from everalgo.user_memory.prompts.en.episode import EPISODE_GENERATION_PROMPT

    title = _extract_example_field(EPISODE_GENERATION_PROMPT, "If the conversation start time is", "title")
    assert "2024-03-14" in title
    assert _MONTH_NAME_DATE_RE.search(title) is None


def test_en_example_start_time_uses_input_side_format() -> None:
    """The example's conversation-start-time value must match what the code now injects."""
    from everalgo.user_memory.prompts.en.episode import EPISODE_GENERATION_PROMPT

    assert "2024-03-14 15:00 UTC (Thursday)" in EPISODE_GENERATION_PROMPT


def test_all_variants_require_utc_label_on_absolute_clock_times() -> None:
    """Absolute times stating a clock time must be labelled UTC, in all four prompt variants.

    The negative half guards a rule that was tried and removed: an instruction forbidding ``content``
    from opening with a date or time. It contradicted the same prompt's own demand for a chronological
    account with per-event times, and a real-LLM run showed the model ignoring it 15/15 — correctly,
    since the narrative's times belong to the events. Do not reintroduce it.
    """
    import everalgo.user_memory.prompts.en.episode as en_mod
    import everalgo.user_memory.prompts.zh.episode as zh_mod

    for name in ("EPISODE_GENERATION_PROMPT", "USER_EPISODE_GENERATION_PROMPT"):
        assert "MUST carry the UTC zone label" in getattr(en_mod, name)
        assert "UTC 时区标识" in getattr(zh_mod, name)
        assert "must NOT begin with a timestamp" not in getattr(en_mod, name)
        assert "不要以时间戳开头" not in getattr(zh_mod, name)


def test_zh_example_start_time_uses_input_side_format() -> None:
    """The zh example's conversation-start-time value must match what the code now injects."""
    from everalgo.user_memory.prompts.zh.episode import EPISODE_GENERATION_PROMPT

    assert "2024-03-14 15:00 UTC (Thursday)" in EPISODE_GENERATION_PROMPT
    assert "3:00 PM" not in EPISODE_GENERATION_PROMPT


def test_zh_generic_example_content_uses_iso_dates() -> None:
    """The zh generic example content used Chinese month-name dates; unify on YYYY-MM-DD."""
    from everalgo.user_memory.prompts.zh.episode import EPISODE_GENERATION_PROMPT

    content = _extract_example_field(EPISODE_GENERATION_PROMPT, "如果对话开始时间为", "content")
    assert "2024-03-16" in content
    assert "2024 年 3 月" not in content
