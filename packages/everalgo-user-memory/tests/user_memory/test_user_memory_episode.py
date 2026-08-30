"""Tests for everalgo.user_memory.episode — EpisodeExtractor (single owner_id, no fan-out)."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.user_memory import OutputLanguage
from everalgo.user_memory.episode import (
    _SUMMARY_WIDTH_CAP,
    EpisodeExtractor,
    _format_prompt_time,
    _render_conversation,
    _resolve_user_name,
    _truncate_at_sentence_boundary,
)
from everalgo.user_memory.prompts.en.episode import SUMMARY_COMPRESS_PROMPT


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

_LANGUAGE_PROMPTS = ("EPISODE_GENERATION_PROMPT", "USER_EPISODE_GENERATION_PROMPT")


@pytest.mark.parametrize("name", _LANGUAGE_PROMPTS)
def test_both_variants_carry_the_language_placeholder_at_both_ends(name: str) -> None:
    """Long prompts lose middle instructions, so the rule is spliced at head and tail."""
    import everalgo.user_memory.prompts.en.episode as mod

    assert getattr(mod, name).count("{language_rule}") == 2


@pytest.mark.parametrize("sender_id", [None, "u_alice"])
async def test_rendering_injects_the_participant_rule_when_no_language_is_named(sender_id: str | None) -> None:
    rendered = await _render_episode_prompt(sender_id=sender_id)

    assert rendered.count("CRITICAL LANGUAGE RULE") == 2
    assert "the language the participants use" in rendered
    assert "{language_rule}" not in rendered


@pytest.mark.parametrize("sender_id", [None, "u_alice"])
async def test_rendering_injects_the_named_language(sender_id: str | None) -> None:
    rendered = await _render_episode_prompt(sender_id=sender_id, output_language=OutputLanguage.GERMAN)

    assert rendered.count("CRITICAL LANGUAGE RULE") == 2
    assert "Write ALL output fields in German." in rendered
    assert "the language the participants use" not in rendered


async def _render_episode_prompt(*, sender_id: str | None, **kwargs: object) -> str:
    """Capture what the extractor hands the LLM; the rule only exists after rendering."""
    captured: list[str] = []

    class Capture:
        async def chat(self, messages: list[LLMChatMessage], **_: object) -> ChatResponse:
            assert isinstance(messages[0].content, str)  # narrow for test
            captured.append(messages[0].content)
            raise _PromptCapturedError

    with pytest.raises(_PromptCapturedError):
        await EpisodeExtractor(llm=Capture()).aextract(_memcell(), sender_id=sender_id, **kwargs)  # type: ignore[arg-type]
    return captured[0]


class _PromptCapturedError(Exception):
    """Ends the call once the prompt has been captured — no LLM response is needed."""


# ==========================================================================
# (1) Single successful extraction
# ==========================================================================


async def test_aextract_returns_single_episode_from_title_content_json() -> None:
    """LLM emits {title, content} -> single Episode with subject=title, episode=content."""
    llm_json = '{"title": "Meeting with Alice at 3pm", "content": "User scheduled a meeting with Alice at 3pm.", "summary": "Preview of User scheduled a meeting with Alice at 3pm."}'
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    ep = await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert ep.subject == "Meeting with Alice at 3pm"
    assert "Alice" in ep.episode
    assert ep.timestamp == 1700000001000


async def test_aextract_episode_fields_filled_correctly() -> None:
    """owner_id on the returned Episode equals the supplied sender_id argument."""
    fake = FakeLLMClient(
        responses=[ChatResponse(content='{"title": "T", "content": "c", "summary": "Preview of c"}', model="fake")]
    )

    ep = await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert ep.owner_id == "u_alice"
    assert ep.subject == "T"
    assert ep.episode.endswith("c")


async def test_aextract_owner_id_comes_from_argument_not_inferred() -> None:
    """owner_id on the returned Episode must equal the caller-supplied sender_id, not be inferred."""
    fake = FakeLLMClient(
        responses=[ChatResponse(content='{"title": "T", "content": "c", "summary": "Preview of c"}', model="fake")]
    )

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
    fake = FakeLLMClient(
        responses=[ChatResponse(content='{"title": "T", "content": "", "summary": "Preview"}', model="fake")]
    )

    with pytest.raises(ValueError, match="empty content"):
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")


async def test_aextract_raises_on_whitespace_only_content() -> None:
    """Whitespace-only ``content`` must be treated the same as empty ``content``."""
    fake = FakeLLMClient(
        responses=[ChatResponse(content='{"title": "T", "content": "   ", "summary": "Preview"}', model="fake")]
    )

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
        return ChatResponse(content='{"title": "T", "content": "c", "summary": "Preview of c"}', model="fake")

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
        return ChatResponse(content='{"title": "T", "content": "c", "summary": "Preview of c"}', model="fake")

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
        return ChatResponse(content='{"title": "T", "content": "c", "summary": "Preview of c"}', model="fake")

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
        return ChatResponse(content='{"title": "T", "content": "c", "summary": "Preview of c"}', model="fake")

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
            ChatResponse(
                content='{"title": "Alice offsite", "content": "Alice proposed offsite.", "summary": "Preview of Alice proposed offsite."}',
                model="fake",
            )
        ]
    )
    fake_bob = FakeLLMClient(
        responses=[
            ChatResponse(
                content='{"title": "Bob offsite", "content": "Bob offered to host.", "summary": "Preview of Bob offered to host."}',
                model="fake",
            )
        ]
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

    llm_json = '{"title": "Demo on Friday", "content": "Alice scheduled a demo for Friday.", "summary": "Preview of Alice scheduled a demo for Friday."}'

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
        responses=[
            ChatResponse(
                content='{"title": "Instance test", "content": "From instance llm", "summary": "Preview of From instance llm"}',
                model="inst",
            )
        ]
    )
    extractor = EpisodeExtractor(llm=instance_fake)
    ep = await extractor.aextract(_memcell(), sender_id="u_alice")
    assert ep.subject == "Instance test"
    assert instance_fake.call_count == 1


@pytest.mark.asyncio
async def test_extract_generic_when_sender_id_is_none() -> None:
    """sender_id=None → generic prompt (EPISODE_GENERATION_PROMPT, no user_name), owner_id=None."""
    fake = FakeLLMClient(
        responses=[
            '{"title": "Bug fix discussion", "content": "Alice and Bob debugged the login flow.", "summary": "Preview of Alice and Bob debugged the login flow."}'
        ]
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
        return ChatResponse(content='{"title": "T", "content": "c", "summary": "Preview of c"}', model="fake")

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
        responses=[
            ChatResponse(
                content='{"title": "T", "content": "Alice asked about hiking.", "summary": "Preview of Alice asked about hiking."}',
                model="fake",
            )
        ]
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
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content='{"title": "T", "content": "body", "summary": "Preview of body"}', model="fake")
        ]
    )

    ep = await EpisodeExtractor(llm=fake).aextract(_memcell_spanning_70_minutes(), sender_id="u_alice")

    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC — ", ep.episode) is None


async def test_episode_timestamp_field_still_holds_closing_time() -> None:
    """Dropping the prefix must not disturb the ``timestamp`` field: still ``memcell.timestamp``."""
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content='{"title": "T", "content": "body", "summary": "Preview of body"}', model="fake")
        ]
    )
    mc = _memcell_spanning_70_minutes()

    ep = await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_alice")

    assert ep.timestamp == mc.timestamp == _TS_FRIDAY_MS + 70 * 60 * 1000


# ==========================================================================
# summary — a model-written display preview, not a slice of the body
#
# For 0.1 through 0.4 this field was `body[:200]`: the prompts never asked for it, so the extractor's
# `data.get("summary")` branch was unreachable and every caller got a blind truncation — cut mid-word in
# English, and a verbatim copy of the whole body in Chinese, where 200 characters is most of an episode.
# ==========================================================================

_SUMMARY = "Alice raised async retries and the assistant promised a doc."


def _episode_response(*, title: str = "T", content: str = "A long narrative body.", summary: str = _SUMMARY) -> str:
    return json.dumps({"title": title, "content": content, "summary": summary})


@pytest.mark.parametrize("sender_id", [None, "u_alice"])
async def test_summary_comes_from_the_model_verbatim(sender_id: str | None) -> None:
    """Both prompt variants must carry the model's summary through untouched."""
    fake = FakeLLMClient(responses=[ChatResponse(content=_episode_response(), model="fake")])

    ep = await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id=sender_id)

    assert ep.summary == _SUMMARY


async def test_summary_is_not_a_slice_of_the_body() -> None:
    """The regression that prompted the change: a body long enough that a 200-char slice is visible."""
    body = "Alice asked about async retries. " * 20
    fake = FakeLLMClient(responses=[ChatResponse(content=_episode_response(content=body), model="fake")])

    ep = await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert ep.summary == _SUMMARY
    assert ep.summary != body[:200]
    assert not body.startswith(ep.summary)


async def test_missing_summary_raises_rather_than_substituting_one() -> None:
    """There is no honest value to invent for a preview the model did not write."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"title": "T", "content": "body"}', model="fake")])

    with pytest.raises(ValueError, match=r"missing required keys \['summary'\]"):
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")


@pytest.mark.parametrize("summary", ["", "   ", "\n\t"])
async def test_blank_summary_raises_like_blank_content(summary: str) -> None:
    """A blank preview is a defect the caller cannot see, so it fails here where the call is still in hand."""
    fake = FakeLLMClient(responses=[ChatResponse(content=_episode_response(summary=summary), model="fake")])

    with pytest.raises(ValueError, match="empty summary"):
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")


@pytest.mark.parametrize("name", _LANGUAGE_PROMPTS)
def test_both_variants_ask_for_summary_after_content(name: str) -> None:
    """Field order is generation order: a summary emitted first would summarise a body that does not exist yet.

    That would make it a second independent pass over the conversation, which is exactly what the field is
    not — it is a compression of `content`. Guarded here because the ordering is invisible at a glance and
    a well-meaning edit that alphabetises the JSON example would silently break it.
    """
    import everalgo.user_memory.prompts.en.episode as mod

    prompt = getattr(mod, name)
    assert prompt.count('"summary"') >= 1
    assert prompt.count('"summary"') == prompt.count('"content"')
    for index in range(len(prompt)):
        if prompt.startswith('"summary"', index):
            assert prompt.rfind('"content"', 0, index) > prompt.rfind('"summary"', 0, index), (
                f"{name}: a summary at offset {index} is not preceded by its own content"
            )


@pytest.mark.parametrize("name", _LANGUAGE_PROMPTS)
def test_both_variants_bound_the_summary_length_and_self_containment(name: str) -> None:
    """The three rigid clauses of the spec: bounded, faithful, and readable without the record."""
    import everalgo.user_memory.prompts.en.episode as mod

    prompt = getattr(mod, name)
    assert "1-3 short sentences" in prompt
    assert "COMPRESS, never restate" in prompt
    assert "Introduce no fact that is not already in content" in prompt
    assert "Do not refer to the record itself" in prompt


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

    for name in ("EPISODE_GENERATION_PROMPT", "USER_EPISODE_GENERATION_PROMPT"):
        assert "MUST carry the UTC zone label" in getattr(en_mod, name)
        assert "must NOT begin with a timestamp" not in getattr(en_mod, name)


# ==========================================================================
# Summary width guard (three tiers)
# ==========================================================================

_LONG_BODY = "The team met and decided many things. " * 30  # ~1170 units, over the cap


def _episode_json(summary: str) -> str:
    return json.dumps({"title": "T", "content": _LONG_BODY, "summary": summary})


async def test_compliant_summary_passes_untouched_with_one_llm_call() -> None:
    """Tier 1: a within-cap summary is stored verbatim; no repair call is spent."""
    calls: list[str] = []

    async def handler(messages: list[LLMChatMessage], **_: object) -> ChatResponse:
        assert isinstance(messages[0].content, str)
        calls.append(messages[0].content)
        return ChatResponse(content=_episode_json("Alice and Bob met and agreed."), model="fake")

    ep = await EpisodeExtractor(llm=FakeLLMClient(handler=handler)).aextract(_memcell(), sender_id="u_alice")

    assert ep.summary == "Alice and Bob met and agreed."  # type: ignore[attr-defined]
    assert len(calls) == 1


async def test_overwide_summary_is_replaced_by_the_compress_call() -> None:
    """Tier 2: an over-cap summary triggers one repair call over content alone."""
    calls: list[str] = []

    async def handler(messages: list[LLMChatMessage], **_: object) -> ChatResponse:
        assert isinstance(messages[0].content, str)
        calls.append(messages[0].content)
        if len(calls) == 1:
            return ChatResponse(content=_episode_json(_LONG_BODY), model="fake")
        return ChatResponse(content="A short faithful preview.", model="fake")

    ep = await EpisodeExtractor(llm=FakeLLMClient(handler=handler)).aextract(_memcell(), sender_id="u_alice")

    assert ep.summary == "A short faithful preview."  # type: ignore[attr-defined]
    assert len(calls) == 2
    # the repair call sees the extracted content, not the raw conversation
    assert _LONG_BODY[:80] in calls[1]
    assert "speaker" not in calls[1]


async def test_failed_compress_degrades_to_sentence_truncation_not_an_error() -> None:
    """Tier 3: repair call dies -> truncated prefix stored; extraction itself never fails."""
    calls: list[int] = []

    async def handler(messages: list[LLMChatMessage], **_: object) -> ChatResponse:
        calls.append(1)
        if len(calls) == 1:
            return ChatResponse(content=_episode_json(_LONG_BODY), model="fake")
        raise RuntimeError("compress model down")

    ep = await EpisodeExtractor(llm=FakeLLMClient(handler=handler)).aextract(_memcell(), sender_id="u_alice")

    summary = ep.summary  # type: ignore[attr-defined]
    assert summary.endswith("\u2026")
    assert len(summary) <= _SUMMARY_WIDTH_CAP
    # whole sentences kept, with the ellipsis REPLACING the final terminator
    body = summary.removesuffix("\u2026")
    assert not body.endswith((".", "!", "?"))
    assert (body + ". ") in _LONG_BODY  # the kept prefix ends exactly where a sentence did


async def test_compress_output_still_over_cap_falls_through_to_truncation() -> None:
    """Tier 2 output that itself violates the cap is not stored; tier 3 truncates it."""
    calls: list[int] = []

    async def handler(messages: list[LLMChatMessage], **_: object) -> ChatResponse:
        calls.append(1)
        if len(calls) == 1:
            return ChatResponse(content=_episode_json(_LONG_BODY), model="fake")
        return ChatResponse(content="Still very wordy. " * 40, model="fake")

    ep = await EpisodeExtractor(llm=FakeLLMClient(handler=handler)).aextract(_memcell(), sender_id="u_alice")

    summary = ep.summary  # type: ignore[attr-defined]
    assert summary.endswith("\u2026")
    assert len(summary) <= _SUMMARY_WIDTH_CAP


def test_truncation_cuts_at_a_sentence_boundary_and_replaces_the_terminator() -> None:
    text = "First sentence here. Second one follows! Third asks a question? " * 20
    out = _truncate_at_sentence_boundary(text, 100)
    assert out.endswith("\u2026")
    body = out.removesuffix("\u2026")
    assert not body.endswith((".", "!", "?"))  # ellipsis replaced the terminator
    assert text.startswith(body) and text[len(body)] in ".!?"  # cut sat ON a boundary
    assert len(out) <= 100


def test_truncation_handles_cjk_width_and_terminators() -> None:
    text = "\u4ed6\u5468\u516d\u65e9\u4e0a\u51fa\u53d1\u53bb\u770b\u65e5\u51fa\u3002" * 30
    out = _truncate_at_sentence_boundary(text, 100)
    assert out.endswith("\u2026")
    body = out.removesuffix("\u2026")
    assert not body.endswith("\u3002")  # ellipsis replaced the CJK full stop
    assert text.startswith(body) and text[len(body)] == "\u3002"
    from everalgo.user_memory._width import ascii_width

    assert ascii_width(out) <= 100


def test_truncation_without_any_sentence_end_hard_cuts_with_ellipsis() -> None:
    text = "word " * 200  # no terminator at all
    out = _truncate_at_sentence_boundary(text, 60)
    assert out.endswith("\u2026")
    assert len(out) <= 60


async def test_guard_logs_the_violation_and_the_repair(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Tier-2 entry logs at WARNING, a successful repair at INFO.

    These two lines are what make the production violation rate and repair success
    rate observable — neither is recorded anywhere else.
    """
    calls: list[int] = []

    async def handler(messages: list[LLMChatMessage], **_: object) -> ChatResponse:
        calls.append(1)
        if len(calls) == 1:
            return ChatResponse(content=_episode_json(_LONG_BODY), model="fake")
        return ChatResponse(content="A short faithful preview.", model="fake")

    with caplog.at_level("INFO", logger="everalgo.user_memory.episode"):
        await EpisodeExtractor(llm=FakeLLMClient(handler=handler)).aextract(_memcell(), sender_id="u_alice")

    messages_logged = [r.message for r in caplog.records]
    assert any("over cap" in m and "attempting compress repair" in m for m in messages_logged)
    assert any("repaired by compress" in m for m in messages_logged)


async def test_guard_logs_the_truncation_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """A repair that dies leaves both the failure and the truncation in the log."""
    calls: list[int] = []

    async def handler(messages: list[LLMChatMessage], **_: object) -> ChatResponse:
        calls.append(1)
        if len(calls) == 1:
            return ChatResponse(content=_episode_json(_LONG_BODY), model="fake")
        raise RuntimeError("compress model down")

    with caplog.at_level("WARNING", logger="everalgo.user_memory.episode"):
        await EpisodeExtractor(llm=FakeLLMClient(handler=handler)).aextract(_memcell(), sender_id="u_alice")

    messages_logged = [r.message for r in caplog.records]
    assert any("compress call failed" in m for m in messages_logged)
    assert any("truncating" in m for m in messages_logged)


async def test_extraction_logs_entry_and_exit(caplog: pytest.LogCaptureFixture) -> None:
    """One extraction reads as one story in the log: an entry line and an exit line.

    Entry carries scale, template and language; exit carries the product widths.
    """
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content='{"title": "T", "content": "Alice met Bob.", "summary": "Alice met Bob."}',
                model="fake",
            )
        ]
    )

    with caplog.at_level("INFO", logger="everalgo.user_memory.episode"):
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    messages_logged = [r.message for r in caplog.records]
    assert any("extracting episode:" in m and "template=user-centred" in m for m in messages_logged)
    assert any("episode extracted:" in m and "summary" in m for m in messages_logged)


def test_compress_prompt_carries_its_contract() -> None:
    """Placeholders present; data slot last, matching the assembly arc convention."""
    assert "{language_rule}" in SUMMARY_COMPRESS_PROMPT
    assert "{episode_text}" in SUMMARY_COMPRESS_PROMPT
    assert SUMMARY_COMPRESS_PROMPT.rstrip().endswith("{episode_text}")
    assert "1-3 short sentences" in SUMMARY_COMPRESS_PROMPT
