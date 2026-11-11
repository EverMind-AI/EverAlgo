"""Tests for everalgo.user_memory.episode — EpisodeExtractor: one call, a strict reply, no retry.

Coverage layers (the prompt text itself, against the reference run's, is ``test_episode_prompt_split``):
- Contract  — the call requests ``_EpisodeReply``; the model obeys docs/concepts/llm-contract.md §2.2; field order
- Error path — an invalid reply, a ``summary`` key or a blank title or content raises LLMOutputError; client
               exceptions propagate unchanged
- Signature — ``prompt=`` is gone; ``custom_instructions`` replaces the default principles in their block;
              ``sender_id`` only sets ``owner_id``
- Language  — ``<output_language>``: the named language, or the participant rule; the Episode reports none
- Rendering — the start time and the reference run's message blocks in the user message
- Stored record — the body verbatim, the summary the content's first 200 characters, the memcell's timestamp
- Log lines — entry and exit
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
from everalgo.user_memory.prompts.en import episode as prompts
from everalgo.user_memory.prompts.en.episode import (
    DEFAULT_CUSTOM_INSTRUCTIONS,
    EPISODE_SYSTEM_PROMPT,
    EPISODE_USER_PROMPT,
    assemble_system_prompt,
)

_LOGGER = "everalgo.user_memory.episode"
_FIELD_ORDER = ["title", "content"]
_SECRET = "EPISODE-SECRET-7d2a"
_TS_MS = 1_700_000_000_000

# 2026-05-29 12:25:10 UTC, a Friday.
_TS_FRIDAY_MS = 1780057510000

# The user message ``_memcell()`` renders to: the first message's time, then the reference run's pseudo-JSON blocks.
_MEMCELL_USER_MESSAGE = """\
<conversation_start_time>
2023-11-14 22:13 UTC (Tuesday)
</conversation_start_time>

<conversation>
                {
                    "timestamp": 2023-11-14 22:13 UTC (Tuesday),
                    "speaker": Alice,
                    "content": Schedule a meeting with Alice at 3pm
                }

                {
                    "timestamp": 2023-11-14 22:13 UTC (Tuesday),
                    "speaker": assistant,
                    "content": Done. Sent invite for 3pm.
                }
</conversation>

<reminder>
Write the episode for the conversation above, following every rule in the system message: give every relative \
time expression its absolute date in parentheses, and keep every fact, name, number, date and stated reason from \
the conversation.
</reminder>"""


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


def _reply(*, title: str = "Alice schedules a 3pm meeting", content: str = "Alice scheduled a meeting at 3pm.") -> str:
    return json.dumps({"title": title, "content": content})


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


def test_reply_model_fields_follow_the_reference_order() -> None:
    """Title, then content, as the reference run's skeleton listed them; there is no summary field."""
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
    assert episode.summary == "Alice scheduled a meeting at 3pm."
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
        pytest.param(f"{{not json {_SECRET}", "<root>", id="not-json"),
        pytest.param(json.dumps({"content": _SECRET}), "title:", id="title-missing"),
        pytest.param(
            json.dumps({"title": "t", "content": _SECRET, "summary": "s"}),
            "summary:",
            id="summary-not-accepted",
        ),
        pytest.param(
            json.dumps({"language": "English", "title": "t", "content": _SECRET}),
            "language:",
            id="language-declaration-forbidden",
        ),
        pytest.param(
            json.dumps({"title": "t", "content": _SECRET, "note": "x"}),
            "note:",
            id="extra-field-forbidden",
        ),
        pytest.param(json.dumps({"title": 7, "content": _SECRET}), "title:", id="number-not-coerced"),
        pytest.param(json.dumps({"title": None, "content": _SECRET}), "title:", id="null-not-accepted"),
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


@pytest.mark.parametrize("field", ["title", "content"])
@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
async def test_blank_stored_field_raises_llm_output_error(field: str, blank: str) -> None:
    """A blank stored field has no honest stand-in, so it fails here where the call is still in hand."""
    values = {"title": _SECRET, "content": _SECRET, field: blank}
    fake = FakeLLMClient(responses=[json.dumps(values)])

    with pytest.raises(LLMOutputError) as excinfo:
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id=None)

    assert str(excinfo.value) == f"episode reply leaves fields blank: {field}: empty after stripping whitespace"
    assert _SECRET not in str(excinfo.value)
    assert fake.call_count == 1


async def test_every_blank_field_is_named() -> None:
    reply = json.dumps({"title": "", "content": " "})

    with pytest.raises(LLMOutputError, match=r"title: empty after stripping whitespace; content: empty"):
        await _extract(reply)


async def test_client_exception_propagates_unchanged() -> None:
    """The host recognises call failures by its own types, so the operator neither wraps nor retypes them."""
    error = _ClientError("upstream down")
    fake = FakeLLMClient(responses=[error])

    with pytest.raises(_ClientError) as excinfo:
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id=None)

    assert excinfo.value is error
    assert fake.call_count == 1


async def test_aextract_plain_text_fallback_and_windows_backslash_sanitizer() -> None:
    r"""Plain text without braces falls back to Session Note; unescaped Windows backslashes in JSON are sanitized."""
    plain_fake = FakeLLMClient(responses=["Plain narrative summary"])
    ep_plain = await EpisodeExtractor(llm=plain_fake).aextract(_memcell(), sender_id="u_alice")
    assert ep_plain.subject == "Session Note"
    assert ep_plain.episode == "Plain narrative summary"

    win_json = r'{"title": "Edited C:\Users\demo\app.py", "content": "Updated C:\workspace\project\main.py"}'
    win_fake = FakeLLMClient(responses=[win_json])
    ep_win = await EpisodeExtractor(llm=win_fake).aextract(_memcell(), sender_id="u_alice")
    assert r"C:\Users\demo\app.py" in ep_win.subject
    assert r"C:\workspace\project\main.py" in ep_win.episode


# ===========================================================================
# Signature — prompt= is gone, custom_instructions replaces the default principles, sender_id only sets owner_id
# ===========================================================================


def test_aextract_takes_no_prompt_override() -> None:
    assert list(inspect.signature(EpisodeExtractor.aextract).parameters) == [
        "self",
        "memcell",
        "sender_id",
        "custom_instructions",
        "output_language",
        "tz",
    ]


def _custom_instructions_block(system: str) -> str:
    assert system.endswith("\n</custom_instructions>")
    return system.split("<custom_instructions>\n", 1)[1].removesuffix("\n</custom_instructions>")


async def test_custom_instructions_are_the_whole_block_verbatim() -> None:
    """They replace the default principles, and as a rendered value they are never scanned for placeholders."""
    _, fake = await _extract(custom_instructions="Mention every {{conversation}} literally.")

    system, user = _sent(fake)
    assert _custom_instructions_block(system) == "Mention every {{conversation}} literally."
    assert "Mention every" not in user


@pytest.mark.parametrize("custom_instructions", [None, "", "  \n"])
async def test_no_or_blank_custom_instructions_send_the_default_principles(custom_instructions: str | None) -> None:
    """The reference run's five principles fill the block, as that run filled its slot when the caller gave none."""
    _, with_none = await _extract()
    _, with_value = await _extract(custom_instructions=custom_instructions)

    assert _sent(with_value) == _sent(with_none)
    assert _custom_instructions_block(_sent(with_value)[0]) == DEFAULT_CUSTOM_INSTRUCTIONS


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
    """Pins the host's call (no language, no instructions): the rules and default principles, then the data."""
    _, fake = await _extract()

    assert _sent(fake) == (
        assemble_system_prompt(custom_instructions=None, language_rule=PARTICIPANT_LANGUAGE_RULE),
        _MEMCELL_USER_MESSAGE,
    )


# ===========================================================================
# Language — the named language or the participant rule; no declaration
# ===========================================================================


async def test_no_language_named_sends_the_participant_rule() -> None:
    """The fallback every conversation reader uses: write in the language the participants use."""
    _, fake = await _extract()

    system, user = _sent(fake)
    output_language = system.split("<output_language>\n", 1)[1].split("\n</output_language>", 1)[0]
    assert output_language == PARTICIPANT_LANGUAGE_RULE
    assert system.count(PARTICIPANT_LANGUAGE_RULE) == 1
    assert PARTICIPANT_LANGUAGE_RULE not in user


async def test_rendering_injects_the_named_language_rule() -> None:
    _, fake = await _extract(output_language=OutputLanguage.GERMAN)

    system, _ = _sent(fake)
    output_language = system.split("<output_language>\n", 1)[1].split("\n</output_language>", 1)[0]
    assert output_language == build_language_rule(OutputLanguage.GERMAN)
    assert system.count(output_language) == 1
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

    with pytest.raises(ValueError, match="names no supported language"):
        await EpisodeExtractor(llm=fake).aextract(_memcell(), sender_id=None, output_language="Klingon")

    assert fake.call_count == 0


def test_prompt_text_never_says_user_language() -> None:
    """No prompt text names ``user_language``: the model declares no language under any name."""
    for name, value in vars(prompts).items():
        if name.isupper() and isinstance(value, str):
            assert "user_language" not in value, name


# ===========================================================================
# Rendering — the start time and the reference run's message blocks
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

    assert rendered.count('"timestamp": ') == 1
    assert '"speaker": Alice,\n                    "content": hi\n' in rendered
    assert "Bob" not in rendered


@pytest.mark.parametrize(
    ("timestamp", "written"),
    [
        (1_710_428_400_000, "2024-03-14 15:00 UTC (Thursday)"),
        (_TS_FRIDAY_MS, "2026-05-29 12:25 UTC (Friday)"),
    ],
)
def test_render_conversation_writes_times_in_the_shared_shape_with_the_weekday(timestamp: int, written: str) -> None:
    """``YYYY-MM-DD HH:MM UTC (Weekday)``, 24-hour, for the start time and each message (``format_time``)."""
    cell = MemCell(
        items=[
            ChatMessage(
                id="m1", role="user", content="hello", timestamp=timestamp, sender_id="u_alice", sender_name="Alice"
            )
        ],
        timestamp=timestamp,
    )

    rendered = _render_conversation(cell)

    assert rendered.startswith(f"<conversation_start_time>\n{written}\n</conversation_start_time>\n")
    assert f'\n                    "timestamp": {written},\n' in rendered


def test_render_conversation_falls_back_to_the_sender_id() -> None:
    cell = MemCell(
        items=[ChatMessage(id="m1", role="user", content="hi", timestamp=_TS_MS, sender_id="u_alice")],
        timestamp=_TS_MS,
    )

    rendered = _render_conversation(cell)

    assert '\n                    "speaker": u_alice,\n' in rendered


def test_render_conversation_rejects_a_message_without_a_timestamp() -> None:
    """Every line carries its time, so a timestamp of 0 is an input error, named by id, never by text or speaker."""
    cell = MemCell(
        items=[
            ChatMessage(id="m1", role="user", content="hi", timestamp=_TS_MS, sender_id="u_alice"),
            ChatMessage(id="m2", role="user", content="secret plan", timestamp=0, sender_id="u_bob", sender_name="Bob"),
        ],
        timestamp=_TS_MS,
    )

    with pytest.raises(ValueError, match=r"^chat message 'm2' has no timestamp: ") as raised:
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


async def test_start_time_is_the_first_message_s_and_each_message_keeps_its_own() -> None:
    """The start time is when the first message was sent, never the memcell's closing time."""
    _, fake = await _extract(memcell=_memcell_spanning_70_minutes())

    _, user = _sent(fake)
    assert user.startswith("<conversation_start_time>\n2026-05-29 12:25 UTC (Friday)\n</conversation_start_time>\n\n")
    assert user.count("2026-05-29 13:35 UTC (Friday)") == 1
    assert '"timestamp": 2026-05-29 13:35 UTC (Friday),' in user


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
# Stored record — body verbatim, times from the model, summary the content's first 200 characters
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


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("Alice met Bob.", id="short"),
        pytest.param("Alice asked about async retries. " * 20, id="long"),
        pytest.param("他周六早上出发去看日出。" * 30, id="cjk"),
    ],
)
async def test_summary_is_the_content_s_first_200_characters(content: str) -> None:
    """As the reference run filled it: a plain slice counted in characters, no sentence cut, no ellipsis, one call."""
    episode, fake = await _extract(_reply(content=content))

    assert episode.summary == content[:200]
    assert fake.call_count == 1


# ===========================================================================
# Log lines — one extraction reads as one story
# ===========================================================================


async def test_extraction_logs_entry_and_exit(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger=_LOGGER):
        await _extract(_reply(content="Alice met Bob."))

    logged = [record.message for record in caplog.records]
    assert any(m.startswith("extracting episode: 2 items, ") and m.endswith("output_language=None") for m in logged)
    assert "episode extracted: content 14 units, summary 14 units" in logged
