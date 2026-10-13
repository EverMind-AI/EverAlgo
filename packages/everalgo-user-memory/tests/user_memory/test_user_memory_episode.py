"""Tests for everalgo.user_memory.episode — EpisodeExtractor: one call, a strict reply, no retry.

Coverage layers:
- Contract  — the call requests ``_EpisodeReply``; the model obeys docs/concepts/llm-contract.md §2.2; field order
- Error path — an invalid reply or a blank stored field raises LLMOutputError; client exceptions propagate unchanged
- Signature — ``prompt=`` is gone; ``custom_instructions`` fills its own block; ``sender_id`` only sets ``owner_id``
- Language  — ``<output_language>`` holds the named language or the participants'; the Episode reports none
- Prompt    — skeleton and examples parse and validate; rules the stored record depends on
- Rendering — the conversation as ``[time] speaker: text`` lines in the user message's ``<conversation>``
- Summary   — the width guard truncates in place, with no second call
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any

import pytest
from pydantic import ValidationError

from everalgo.llm import LLMOutputError, StrictOutputModel, to_openai_response_format
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.user_memory import EpisodeExtractor, OutputLanguage
from everalgo.user_memory._language import PARTICIPANT_LANGUAGE_RULE, build_language_rule
from everalgo.user_memory.episode._render import _render_conversation
from everalgo.user_memory.episode._replies import _EpisodeReply
from everalgo.user_memory.episode._summary import _SUMMARY_WIDTH_CAP, _truncate_at_sentence_boundary
from everalgo.user_memory.prompts.en import episode as prompts
from everalgo.user_memory.prompts.en.episode import (
    EPISODE_SYSTEM_PROMPT,
    EPISODE_USER_PROMPT,
    assemble_system_prompt,
)
from everalgo.utils import ascii_width

_LOGGER = "everalgo.user_memory.episode"
_FIELD_ORDER = ["content", "summary", "title"]
_SECRET = "EPISODE-SECRET-7d2a"
_TS_MS = 1_700_000_000_000

# 2026-05-29 12:25:10 UTC, a Friday.
_TS_FRIDAY_MS = 1780057510000


def _memcell() -> MemCell:
    """A 2-message dialogue: a named user and an unnamed assistant."""
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="Schedule a meeting with Alice at 3pm",
                timestamp=_TS_MS,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ChatMessage(
                id="m2",
                role="assistant",
                content="Done. Sent invite for 3pm.",
                timestamp=_TS_MS + 1000,
                sender_id="assistant",
            ),
        ],
        timestamp=_TS_MS + 1000,
    )


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


def _reply(
    *,
    content: str = "Alice scheduled a meeting at 3pm.",
    summary: str = "Alice scheduled a 3pm meeting.",
    title: str = "Alice schedules a 3pm meeting",
) -> str:
    return json.dumps({"content": content, "summary": summary, "title": title})


async def _extract(reply: str = "", **kwargs: Any) -> tuple[Any, FakeLLMClient]:
    """Run one extraction over ``_memcell()`` against a scripted reply; returns the Episode and the fake."""
    fake = FakeLLMClient(responses=[reply or _reply()])
    kwargs.setdefault("sender_id", None)
    episode = await EpisodeExtractor(llm=fake).aextract(kwargs.pop("memcell", _memcell()), **kwargs)
    return episode, fake


def _sent(fake: FakeLLMClient) -> tuple[str, str]:
    """The one request the extractor sent, as its system and user message contents."""
    (call,) = fake.calls
    system, user = call.messages
    assert (system.role, user.role) == ("system", "user")
    assert isinstance(system.content, str)
    assert isinstance(user.content, str)
    return system.content, user.content


class _ClientError(Exception):
    """A host's own call-failure type, which the operator must neither wrap nor retype."""


# ===========================================================================
# Contract — docs/concepts/llm-contract.md §2.2 / §4.3
# ===========================================================================


def test_reply_model_is_strict_one_level_and_default_free() -> None:
    """The reply model inherits StrictOutputModel directly, requires every field and carries no validators."""
    assert _EpisodeReply.__bases__ == (StrictOutputModel,)
    assert _EpisodeReply.model_config.get("strict") is True
    assert _EpisodeReply.model_config.get("extra") == "forbid"
    for name, field in _EpisodeReply.model_fields.items():
        assert field.is_required(), f"_EpisodeReply.{name} has a default"
    assert not _EpisodeReply.__pydantic_decorators__.field_validators
    assert not _EpisodeReply.__pydantic_decorators__.model_validators


def test_reply_model_fields_follow_the_writing_order() -> None:
    """Content first; summary after the content it compresses; title last."""
    assert list(_EpisodeReply.model_fields) == _FIELD_ORDER


def test_reply_model_fits_the_strict_subset() -> None:
    """The request builders raise on anything outside the strict subset or a non-object top level (§8.3)."""
    response_format = to_openai_response_format(_EpisodeReply)

    assert response_format is not None
    assert response_format["json_schema"]["schema"]["type"] == "object"


async def test_the_call_requests_the_reply_model_with_a_system_and_a_user_message() -> None:
    _, fake = await _extract()

    assert [call.response_format for call in fake.calls] == [_EpisodeReply]
    assert [message.role for message in fake.calls[0].messages] == ["system", "user"]


async def test_valid_reply_becomes_the_episode() -> None:
    episode, fake = await _extract(sender_id="u_alice")

    assert episode.owner_id == "u_alice"
    assert episode.subject == "Alice schedules a 3pm meeting"
    assert episode.episode == "Alice scheduled a meeting at 3pm."
    assert episode.summary == "Alice scheduled a 3pm meeting."
    assert episode.timestamp == _TS_MS + 1000
    assert fake.call_count == 1


def test_sync_bridge_returns_the_episode() -> None:
    fake = FakeLLMClient(responses=[_reply()])

    episode = EpisodeExtractor(llm=fake).extract(_memcell(), sender_id=None)

    assert episode.subject == "Alice schedules a 3pm meeting"
    assert fake.call_count == 1


# ===========================================================================
# Error path — single call, no retry
# ===========================================================================


@pytest.mark.parametrize(
    ("reply", "failing_location"),
    [
        pytest.param(f"not json {_SECRET}", "<root>", id="not-json"),
        pytest.param(json.dumps({"content": _SECRET, "summary": "s"}), "title:", id="title-missing"),
        pytest.param(
            json.dumps({"language": "English", "content": _SECRET, "summary": "s", "title": "t"}),
            "language:",
            id="language-declaration-forbidden",
        ),
        pytest.param(
            json.dumps({"content": _SECRET, "summary": "s", "title": "t", "note": "x"}),
            "note:",
            id="extra-field-forbidden",
        ),
        pytest.param(
            json.dumps({"content": _SECRET, "summary": "s", "title": 7}),
            "title:",
            id="number-not-coerced",
        ),
        pytest.param(
            json.dumps({"content": _SECRET, "summary": None, "title": "t"}),
            "summary:",
            id="null-not-accepted",
        ),
    ],
)
async def test_reply_failing_the_model_raises_llm_output_error(reply: str, failing_location: str) -> None:
    """An invalid reply is an LLMOutputError at once — no retry, no reply text in the message."""
    fake = FakeLLMClient(responses=[reply])

    with pytest.raises(LLMOutputError) as excinfo:
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id=None)

    assert excinfo.type is LLMOutputError
    assert isinstance(excinfo.value.__cause__, ValidationError)
    assert "episode reply is not a valid _EpisodeReply" in str(excinfo.value)
    assert failing_location in str(excinfo.value)
    assert _SECRET not in str(excinfo.value)
    assert fake.call_count == 1


@pytest.mark.parametrize("field", ["content", "summary", "title"])
@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
async def test_blank_stored_field_raises_llm_output_error(field: str, blank: str) -> None:
    """A blank stored field has no honest stand-in, so it fails here where the call is still in hand."""
    values = {"content": _SECRET, "summary": _SECRET, "title": _SECRET, field: blank}
    fake = FakeLLMClient(responses=[json.dumps(values)])

    with pytest.raises(LLMOutputError) as excinfo:
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id=None)

    assert str(excinfo.value) == f"episode reply leaves fields blank: {field}: empty after stripping whitespace"
    assert _SECRET not in str(excinfo.value)
    assert fake.call_count == 1


async def test_every_blank_field_is_named() -> None:
    reply = json.dumps({"content": " ", "summary": "s", "title": ""})

    with pytest.raises(LLMOutputError, match=r"content: empty after stripping whitespace; title: empty"):
        await _extract(reply)


async def test_client_exception_propagates_unchanged() -> None:
    """The host recognises call failures by its own types, so the operator neither wraps nor retypes them."""
    error = _ClientError("upstream down")
    fake = FakeLLMClient(responses=[error])

    with pytest.raises(_ClientError) as excinfo:
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id=None)

    assert excinfo.value is error
    assert fake.call_count == 1


# ===========================================================================
# Signature — prompt= is gone, custom_instructions fills its block, sender_id only sets owner_id
# ===========================================================================


def test_aextract_takes_no_prompt_override() -> None:
    assert list(inspect.signature(EpisodeExtractor.aextract).parameters) == [
        "self",
        "memcell",
        "sender_id",
        "custom_instructions",
        "output_language",
    ]


async def test_custom_instructions_are_the_whole_block_verbatim() -> None:
    """No bundled principles precede them, and as a rendered value they are never scanned for placeholders."""
    _, fake = await _extract(custom_instructions="Mention every {{conversation}} literally.")

    system, user = _sent(fake)
    block = system.split("<custom_instructions>\n", 1)[1].split("\n</custom_instructions>", 1)[0]
    assert block == "Mention every {{conversation}} literally."
    assert "Mention every" not in user


@pytest.mark.parametrize("custom_instructions", [None, "", "  \n"])
async def test_no_or_blank_custom_instructions_leave_the_block_out(custom_instructions: str | None) -> None:
    _, with_none = await _extract()
    _, with_value = await _extract(custom_instructions=custom_instructions)

    assert _sent(with_value) == _sent(with_none)
    assert "<custom_instructions>" not in _sent(with_value)[0]


async def test_sender_id_only_sets_owner_id() -> None:
    """The prompt narrates the whole memcell either way; there is no user-centred variant."""
    alice, alice_fake = await _extract(sender_id="u_alice")
    generic, generic_fake = await _extract(sender_id=None)

    assert _sent(alice_fake) == _sent(generic_fake)
    assert alice.owner_id == "u_alice"
    assert generic.owner_id is None
    assert all("User name:" not in content for content in _sent(alice_fake))


@pytest.mark.parametrize(
    "name",
    [
        "USER_EPISODE_SYSTEM_PROMPT",
        "SUMMARY_COMPRESS_PROMPT",
        "DEFAULT_CUSTOM_INSTRUCTIONS",
        "EPISODE_DECLARE_LANGUAGE_ADDITION",
        "EPISODE_PARTICIPANT_LANGUAGE_RULE",
        "EPISODE_CALLER_LANGUAGE_RULE",
    ],
)
def test_retired_prompt_constants_are_gone(name: str) -> None:
    assert not hasattr(prompts, name)


def test_user_centred_prompt_slot_is_gone() -> None:
    assert "{{user_name}}" not in EPISODE_SYSTEM_PROMPT + EPISODE_USER_PROMPT


async def test_default_call_sends_the_templates_filled() -> None:
    """Pins the host's call (no language, no instructions): the static rules and the participant rule only."""
    _, fake = await _extract()

    assert _sent(fake) == (
        assemble_system_prompt(custom_instructions=None, language_rule=PARTICIPANT_LANGUAGE_RULE),
        '<conversation count="2">\n'
        "[2023-11-14 22:13 UTC (Tuesday)] Alice: Schedule a meeting with Alice at 3pm\n"
        "[2023-11-14 22:13 UTC (Tuesday)] assistant: Done. Sent invite for 3pm.\n"
        "</conversation>",
    )


# ===========================================================================
# Language — the rule the model writes in; no declaration
# ===========================================================================


async def test_rendering_injects_the_participant_rule_when_no_language_is_named() -> None:
    _, fake = await _extract()

    system, user = _sent(fake)
    output_language = system.split("<output_language>\n", 1)[1].split("\n</output_language>", 1)[0]
    assert output_language == PARTICIPANT_LANGUAGE_RULE
    assert "{{" not in system
    assert PARTICIPANT_LANGUAGE_RULE not in user


async def test_rendering_injects_the_named_language_instead_of_the_participant_rule() -> None:
    _, fake = await _extract(output_language=OutputLanguage.GERMAN)

    system, _ = _sent(fake)
    assert system.count(build_language_rule(OutputLanguage.GERMAN)) == 1
    assert "in language" not in system
    assert PARTICIPANT_LANGUAGE_RULE not in system


@pytest.mark.parametrize("output_language", [None, OutputLanguage.CHINESE])
async def test_episode_reports_no_language(output_language: OutputLanguage | None) -> None:
    """The model declares none and a named language is not echoed back: a host judges it from the text."""
    episode, _ = await _extract(output_language=output_language)

    assert getattr(episode, "user_language", None) is None
    assert episode.model_extra == {}


async def test_unsupported_output_language_raises_before_any_call() -> None:
    fake = FakeLLMClient(responses=[_reply()])

    with pytest.raises(ValueError, match="unsupported output_language"):
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id=None, output_language="Klingon")

    assert fake.call_count == 0


# ===========================================================================
# Prompt — skeleton and examples, and the rules the stored record depends on
# ===========================================================================


def _shown_replies() -> dict[str, dict[str, Any]]:
    """The skeleton and the two worked examples' replies, parsed."""
    english, chinese = re.findall(r"<reply>\n(.*?)\n</reply>", EPISODE_SYSTEM_PROMPT, re.DOTALL)
    skeleton = EPISODE_SYSTEM_PROMPT.split("<reply_format>\n", 1)[1].split("\n</reply_format>", 1)[0].split("\n", 1)[1]
    return {
        name: json.loads(text) for name, text in (("skeleton", skeleton), ("english", english), ("chinese", chinese))
    }


@pytest.mark.parametrize("name", ["skeleton", "english", "chinese"])
def test_skeleton_and_examples_are_valid_replies_in_field_order(name: str) -> None:
    """What the prompt shows is what the code accepts: each block parses, validates, and keeps the field order."""
    block = _shown_replies()[name]

    assert list(block) == _FIELD_ORDER
    _EpisodeReply.model_validate_json(json.dumps(block))


def test_field_descriptions_follow_the_field_order() -> None:
    section = EPISODE_SYSTEM_PROMPT.split("<what_to_write>\n", 1)[1].split("\n</what_to_write>", 1)[0]

    assert [line.partition(":")[0] for line in section.splitlines()] == [f"- {name}" for name in _FIELD_ORDER]


def test_prompt_text_never_says_user_language() -> None:
    """No prompt text names ``user_language``: the model declares no language under any name."""
    for name, value in vars(prompts).items():
        if name.isupper() and isinstance(value, str):
            assert "user_language" not in value, name


def test_participant_rule_asks_for_no_declaration() -> None:
    """The shared rule the episode takes asks only for the participants' language, never for it to be declared."""
    assert "the language the participants use" in PARTICIPANT_LANGUAGE_RULE
    assert "English name" not in PARTICIPANT_LANGUAGE_RULE
    assert "in language" not in PARTICIPANT_LANGUAGE_RULE


def test_chinese_example_keeps_technical_terms_in_chinese_output() -> None:
    example = _shown_replies()["chinese"]

    assert example["title"] == "林然和周敏排查 payment API 超时并制定 retry 方案"
    assert "payment API 在 staging 环境" in example["content"]
    assert "timeout" in example["content"]
    assert example["summary"].startswith("林然和周敏排查 staging 环境的 payment API 超时")
    assert all("2026-08-20 09:30 UTC" not in value for value in example.values())


def test_summary_spec_bounds_length_and_self_containment() -> None:
    """The rigid clauses of the spec: bounded, faithful, and readable without the record."""
    for clause in (
        "1-3 short sentences",
        "Compress, never restate",
        "You may omit minor details.",
        "Do not refer to the record itself",
    ):
        assert clause in EPISODE_SYSTEM_PROMPT
    assert "Do not invent facts" not in EPISODE_SYSTEM_PROMPT


_MONTH_NAME_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},?\s+\d{4}\b"
)
_LEADING_TIMESTAMP_RE = re.compile(r"^(?:\d|(?:On|At|In)\s+(?:\S+\s+){0,3}?\d)")


def test_dual_format_examples_preserve_granularity_and_use_iso_dates() -> None:
    for clause in (
        "Resolve it against the timestamp of the message that contains it",
        "last Friday (2023-05-12)",
        "last week (2023-05-08 to 2023-05-14)",
        "ISO Monday-Sunday weeks",
        "at the same granularity",
    ):
        assert clause in EPISODE_SYSTEM_PROMPT
    assert _MONTH_NAME_DATE_RE.search(EPISODE_SYSTEM_PROMPT) is None


def test_time_rules_keep_vague_references_vague_and_name_no_start_time() -> None:
    for clause in (
        '("recently", "later", "someday")',
        "never invent a date or range for them",
        "do not copy message timestamps into content",
    ):
        assert clause in EPISODE_SYSTEM_PROMPT
    assert "start time" not in EPISODE_SYSTEM_PROMPT.lower()


def test_absolute_clock_times_require_the_utc_label() -> None:
    """The negative half guards a rule tried and removed: forbidding ``content`` from opening with a time.

    It contradicted the prompt's own demand for a chronological account with per-event times, and a real-LLM run
    showed the model ignoring it 15/15 — correctly, since the narrative's times belong to the events.
    """
    assert (
        "Only a clock time you derive yourself from a message timestamp, when a relative expression resolves to one "
        '("in two hours"), carries the UTC label, written "2024-03-14 15:00 UTC"' in EPISODE_SYSTEM_PROMPT
    )
    assert "do not convert it and do not add a zone label" in EPISODE_SYSTEM_PROMPT
    assert "must NOT begin with a timestamp" not in EPISODE_SYSTEM_PROMPT


def test_english_example_body_has_no_timestamp_prefix() -> None:
    """The example was what taught the model to open the body with a date; guard the invariant, not one string."""
    content = _shown_replies()["english"]["content"]

    assert _LEADING_TIMESTAMP_RE.match(content) is None


def test_english_example_title_prioritises_event_and_outcome_over_date() -> None:
    title = _shown_replies()["english"]["title"]

    assert "Caroline" in title
    assert "Mount Rainier" in title
    assert "Early Saturday" in title
    assert "2024-03-14" not in title
    assert _MONTH_NAME_DATE_RE.search(title) is None


def test_title_length_has_only_an_upper_bound() -> None:
    assert "fewer than 20 words, and only as many as needed" in EPISODE_SYSTEM_PROMPT
    assert "10-20 words" not in EPISODE_SYSTEM_PROMPT


def test_example_lines_use_the_input_side_format() -> None:
    assert "\n[2024-03-14 15:00 UTC (Thursday)] Caroline: " in EPISODE_SYSTEM_PROMPT


# ===========================================================================
# Rendering — the conversation
# ===========================================================================


def test_render_conversation_skips_empty_content() -> None:
    cell = MemCell(
        items=[
            ChatMessage(id="m1", role="user", content="hi", timestamp=_TS_MS, sender_id="u_alice", sender_name="Alice"),
            ChatMessage(id="m2", role="user", content="", timestamp=_TS_MS + 1, sender_id="u_bob", sender_name="Bob"),
        ],
        timestamp=_TS_MS + 1,
    )

    rendered = _render_conversation(cell)

    assert rendered.startswith('<conversation count="1">\n')
    assert "] Alice: hi\n" in rendered
    assert "Bob" not in rendered


def test_render_conversation_uses_24h_input_format_with_the_weekday() -> None:
    """The weekday is what lets the LLM resolve "last Friday"; dropping it re-introduces off-by-one-week errors."""
    cell = MemCell(
        items=[
            ChatMessage(
                id="m1", role="user", content="hello", timestamp=_TS_FRIDAY_MS, sender_id="u_alice", sender_name="Alice"
            )
        ],
        timestamp=_TS_FRIDAY_MS,
    )

    rendered = _render_conversation(cell)

    assert "\n[2026-05-29 12:25 UTC (Friday)] Alice: hello\n" in rendered
    assert "PM" not in rendered


def test_render_conversation_falls_back_to_the_sender_id() -> None:
    cell = MemCell(
        items=[ChatMessage(id="m1", role="user", content="hi", timestamp=_TS_MS, sender_id="u_alice")],
        timestamp=_TS_MS,
    )

    lines = _render_conversation(cell).splitlines()[1:-1]

    assert lines == ["[2023-11-14 22:13 UTC (Tuesday)] u_alice: hi"]


def test_render_conversation_rejects_a_message_without_a_timestamp() -> None:
    """Every line carries its time, so a timestamp of 0 is an input error, named by id, never by text or speaker."""
    cell = MemCell(
        items=[
            ChatMessage(id="m1", role="user", content="hi", timestamp=_TS_MS, sender_id="u_alice"),
            ChatMessage(id="m2", role="user", content="secret plan", timestamp=0, sender_id="u_bob", sender_name="Bob"),
        ],
        timestamp=_TS_MS,
    )

    with pytest.raises(ValueError, match=r"^chat message 'm2' has no timestamp$") as raised:
        _render_conversation(cell)

    assert "secret plan" not in str(raised.value)
    assert "Bob" not in str(raised.value)


async def test_extraction_without_a_message_timestamp_fails_before_calling_the_model() -> None:
    cell = MemCell(
        items=[ChatMessage(id="m1", role="user", content="hi", timestamp=0, sender_id="u_alice")],
        timestamp=_TS_MS,
    )
    fake = FakeLLMClient(responses=[])

    with pytest.raises(ValueError, match="has no timestamp"):
        await EpisodeExtractor(llm=fake).aextract(cell, sender_id=None)

    assert fake.calls == []


async def test_user_message_carries_no_start_time_only_per_message_times() -> None:
    """The first line's time is when the conversation began; nothing restates it as a start time."""
    _, fake = await _extract(memcell=_memcell_spanning_70_minutes())

    _, user = _sent(fake)
    assert user.startswith('<conversation count="2">\n[2026-05-29 12:25 UTC (Friday)] Alice: start\n')
    assert "start_time" not in user


async def test_non_chat_items_are_silently_skipped() -> None:
    """Agent → user-memory contract: tool items in a MemCell change nothing the model is shown."""
    chat = ChatMessage(
        id="c1",
        role="user",
        content="Let's schedule the demo for Friday",
        timestamp=_TS_MS,
        sender_id="u_alice",
        sender_name="Alice",
    )
    mixed = MemCell(
        items=[
            chat,
            ToolCallRequest(
                tool_calls=[
                    ToolCall(
                        id="tc1", function=ToolCallFunction(name="calendar.create", arguments='{"time": "Friday"}')
                    )
                ],
                timestamp=_TS_MS + 1000,
                sender_id="assistant",
            ),
            ToolCallResult(tool_call_id="tc1", content="Event created.", timestamp=_TS_MS + 2000),
        ],
        timestamp=_TS_MS + 2000,
    )

    _, chat_fake = await _extract(memcell=MemCell(items=[chat], timestamp=_TS_MS + 2000))
    _, mixed_fake = await _extract(memcell=mixed)

    assert _sent(chat_fake) == _sent(mixed_fake)


# ===========================================================================
# Stored record — body verbatim, times from the model, summary from the model
# ===========================================================================


async def test_episode_body_is_stored_verbatim_with_no_code_built_prefix() -> None:
    """Code must not prepend a timestamp — the prompt owns the narrative's times.

    A prefix was tried and removed: its value was ``items[0].timestamp``, the instant the narrative's own timeline
    opens on, so it restated the opening message's time a second time.
    """
    episode, _ = await _extract(_reply(content="Alice asked about hiking."), memcell=_memcell_spanning_70_minutes())

    assert episode.episode == "Alice asked about hiking."
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC — ", episode.episode) is None


async def test_episode_timestamp_is_the_memcell_closing_time() -> None:
    cell = _memcell_spanning_70_minutes()

    episode, _ = await _extract(memcell=cell)

    assert episode.timestamp == cell.timestamp == _TS_FRIDAY_MS + 70 * 60 * 1000


async def test_summary_comes_from_the_model_not_a_slice_of_the_body() -> None:
    """For 0.1 through 0.4 the summary was ``body[:200]``; it is the model's own preview now."""
    body = "Alice asked about async retries. " * 20
    summary = "Alice raised async retries and the assistant promised a doc."

    episode, _ = await _extract(_reply(content=body, summary=summary))

    assert episode.summary == summary
    assert not body.startswith(episode.summary)


# ===========================================================================
# Summary width guard — truncation in place, never a second call
# ===========================================================================

_LONG_SUMMARY = "SUMMARY_ONLY_MARKER. The team met and decided many things. " * 30


async def test_summary_at_the_cap_passes_untouched() -> None:
    at_cap = "a" * _SUMMARY_WIDTH_CAP

    episode, fake = await _extract(_reply(summary=at_cap))

    assert episode.summary == at_cap
    assert fake.call_count == 1


async def test_overwide_summary_is_truncated_without_a_second_call() -> None:
    episode, fake = await _extract(_reply(summary=_LONG_SUMMARY))

    assert fake.call_count == 1
    assert episode.summary.endswith("…")
    assert ascii_width(episode.summary) <= _SUMMARY_WIDTH_CAP
    body = episode.summary.removesuffix("…")
    assert not body.endswith((".", "!", "?"))
    assert (body + ". ") in _LONG_SUMMARY


async def test_one_unit_over_the_cap_is_truncated() -> None:
    episode, _ = await _extract(_reply(summary="a" * (_SUMMARY_WIDTH_CAP + 1)))

    assert episode.summary.endswith("…")
    assert ascii_width(episode.summary) <= _SUMMARY_WIDTH_CAP


async def test_guard_logs_the_violation_and_the_truncation(caplog: pytest.LogCaptureFixture) -> None:
    """The violation line's rate is the production model's; the truncation line records what was stored."""
    with caplog.at_level("INFO", logger=_LOGGER):
        episode, _ = await _extract(_reply(summary=_LONG_SUMMARY))

    by_level = {(record.levelname, record.message) for record in caplog.records}
    original = ascii_width(_LONG_SUMMARY)
    assert ("WARNING", f"episode summary over cap ({original} > {_SUMMARY_WIDTH_CAP} units)") in by_level
    assert ("INFO", f"episode summary truncated: {original} -> {ascii_width(episode.summary)} units") in by_level


async def test_compliant_summary_logs_no_guard_line(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger=_LOGGER):
        await _extract()

    assert not any("episode summary" in record.message for record in caplog.records)


def test_truncation_cuts_at_a_sentence_boundary_and_replaces_the_terminator() -> None:
    text = "First sentence here. Second one follows! Third asks a question? " * 20

    out = _truncate_at_sentence_boundary(text, 100)

    assert out.endswith("…")
    body = out.removesuffix("…")
    assert not body.endswith((".", "!", "?"))
    assert text.startswith(body) and text[len(body)] in ".!?"
    assert len(out) <= 100


def test_truncation_handles_cjk_width_and_terminators() -> None:
    text = "他周六早上出发去看日出。" * 30

    out = _truncate_at_sentence_boundary(text, 100)

    assert out.endswith("…")
    body = out.removesuffix("…")
    assert not body.endswith("。")
    assert text.startswith(body) and text[len(body)] == "。"
    assert ascii_width(out) <= 100


def test_truncation_without_any_sentence_end_hard_cuts_with_ellipsis() -> None:
    out = _truncate_at_sentence_boundary("word " * 200, 60)

    assert out.endswith("…")
    assert len(out) <= 60


# ===========================================================================
# Log lines — one extraction reads as one story
# ===========================================================================


async def test_extraction_logs_entry_and_exit(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger=_LOGGER):
        await _extract(_reply(content="Alice met Bob.", summary="Alice met Bob."))

    logged = [record.message for record in caplog.records]
    assert any(m.startswith("extracting episode: 2 items, ") and m.endswith("output_language=None") for m in logged)
    assert "episode extracted: content 14 units, summary 14 units" in logged
