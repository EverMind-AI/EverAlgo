"""Tests for everalgo.user_memory.language — LanguageDetector: one call, a one-label strict reply, no retry.

Coverage layers:
- Rendering — chat messages with text only, no sender names, each long text cut to its head, middle and tail
- Request   — a system and a user message, the reply model, the caller's block last or left out
- Result    — a label maps to OutputLanguage, ``Other`` to English; a record without text gives None with no call
- Errors    — an invalid reply raises LLMOutputError, client exceptions propagate, a cut below 1 is rejected
- Labels    — the reply accepts Chinese, English, Japanese and ``Other``, and the prompt lists exactly those
- Example   — the worked example is a record as the detector renders one, answered with a valid reply
- Log lines — counts and the label, never the conversation
"""

from __future__ import annotations

import json
import typing

import pytest

from everalgo.llm import LLMOutputError
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, ConversationItem, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.user_memory import LanguageDetector, OutputLanguage
from everalgo.user_memory.language import _cuts, _LanguageReply, _render_chat_record
from everalgo.user_memory.prompts.en.language import LANGUAGE_LABELS, LANGUAGE_SYSTEM_PROMPT
from everalgo.utils import InvalidInputError

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

_TS_MS = 1_700_000_000_000
_SECRET = "LANGUAGE-SECRET-4b1e"


def _message(
    role: typing.Literal["user", "assistant"], content: str, *, n: int = 0, name: str | None = None
) -> ChatMessage:
    return ChatMessage(
        id=f"m{n}", role=role, content=content, timestamp=_TS_MS + n, sender_id=f"{role}-{n}", sender_name=name
    )


def _reply(label: str) -> str:
    return json.dumps({"language": label})


_CHINESE_REPLY = _reply("Chinese")


async def _detect(
    items: Sequence[ConversationItem], *, reply: str | Exception = _CHINESE_REPLY, **kwargs: typing.Any
) -> tuple[OutputLanguage | None, FakeLLMClient]:
    fake = FakeLLMClient(responses=[reply])
    result = await LanguageDetector(llm=fake).adetect(items, **kwargs)
    return result, fake


def _sent(fake: FakeLLMClient, index: int) -> str:
    """The text of the first call's message at ``index``: 0 the system message, 1 the user message."""
    content = fake.calls[0].messages[index].content
    assert isinstance(content, str)
    return content


def _record_sent(fake: FakeLLMClient) -> str:
    return _sent(fake, 1)


# ── Rendering ──────────────────────────────────────────────────────────────


async def test_record_shows_role_and_text_without_names() -> None:
    items = [
        _message("user", "帮我看一下这个报错", n=0, name="林然"),
        _message("assistant", "好的 我先看日志", n=1, name="bot"),
    ]

    _, fake = await _detect(items)

    assert (
        _record_sent(fake) == "<chat_record>\n[user] 帮我看一下这个报错\n\n[assistant] 好的 我先看日志\n</chat_record>"
    )


async def test_tool_items_and_blank_messages_are_skipped() -> None:
    request = ToolCallRequest(
        tool_calls=[ToolCall(id="tc1", function=ToolCallFunction(name="search", arguments="{}"))],
        timestamp=_TS_MS + 1,
        sender_id="assistant",
    )
    result_item = ToolCallResult(tool_call_id="tc1", content="tool output", timestamp=_TS_MS + 2)
    items: list[ConversationItem] = [
        _message("user", "hello", n=0),
        request,
        result_item,
        _message("assistant", "   ", n=3),
    ]

    _, fake = await _detect(items)

    assert _record_sent(fake) == "<chat_record>\n[user] hello\n</chat_record>"


async def test_long_text_keeps_its_head_middle_and_tail_per_role() -> None:
    user_text = "H" * 150 + "x" * 100 + "M" * 150 + "y" * 100 + "T" * 150
    assistant_text = "A" * 50 + "x" * 25 + "C" * 50 + "y" * 25 + "B" * 50

    _, fake = await _detect(
        [_message("user", user_text, n=0), _message("assistant", assistant_text, n=1)],
        user_chars=150,
        assistant_chars=50,
    )

    record = _record_sent(fake)
    assert f"[user] {'H' * 150} … {'M' * 150} … {'T' * 150}" in record
    assert f"[assistant] {'A' * 50} … {'C' * 50} … {'B' * 50}" in record
    assert "x" not in record
    assert "y" not in record


async def test_words_in_the_middle_of_a_template_reach_the_model() -> None:
    template = "Run as a coding agent. " * 20
    instruction = "请把今天的任务整理成中文汇报"

    _, fake = await _detect([_message("user", template + instruction + template)], user_chars=200)

    assert instruction in _record_sent(fake)


async def test_text_at_three_times_the_cut_is_kept_whole() -> None:
    text = "x" * 600

    _, fake = await _detect([_message("user", text)], user_chars=200)

    assert f"[user] {text}\n" in _record_sent(fake)


async def test_text_past_three_times_the_cut_is_cut() -> None:
    _, fake = await _detect([_message("user", "x" * 601)], user_chars=200)

    assert f"[user] {'x' * 200} … {'x' * 200} … {'x' * 200}\n" in _record_sent(fake)


async def test_braces_in_text_stay_literal() -> None:
    _, fake = await _detect([_message("user", "render {{chat_record}} and {name} literally")])

    assert "[user] render {{chat_record}} and {name} literally" in _record_sent(fake)


# ── Request ────────────────────────────────────────────────────────────────


async def test_request_is_system_then_user_with_the_reply_model() -> None:
    _, fake = await _detect([_message("user", "hi")])

    call = fake.calls[0]
    assert [message.role for message in call.messages] == ["system", "user"]
    assert call.messages[0].content == LANGUAGE_SYSTEM_PROMPT
    assert call.response_format is _LanguageReply


async def test_custom_instructions_close_the_system_message() -> None:
    _, fake = await _detect([_message("user", "hi")], custom_instructions="Treat Cantonese as Chinese.")

    system = _sent(fake, 0)
    assert system.startswith(LANGUAGE_SYSTEM_PROMPT)
    assert system.endswith("<custom_instructions>\nTreat Cantonese as Chinese.\n</custom_instructions>")


@pytest.mark.parametrize("custom_instructions", [None, "", "  \n"])
async def test_absent_custom_instructions_leave_no_block(custom_instructions: str | None) -> None:
    _, fake = await _detect([_message("user", "hi")], custom_instructions=custom_instructions)

    assert "custom_instructions" not in _sent(fake, 0)


# ── Result ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("language", [OutputLanguage.CHINESE, OutputLanguage.ENGLISH, OutputLanguage.JAPANESE])
async def test_a_named_language_comes_back_as_output_language(language: OutputLanguage) -> None:
    result, _ = await _detect([_message("user", "hi")], reply=_reply(language.value))

    assert result is language


async def test_other_comes_back_as_english() -> None:
    result, fake = await _detect([_message("user", "สวัสดี")], reply=_reply("Other"))

    assert result is OutputLanguage.ENGLISH
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "items",
    [
        [],
        [_message("user", ""), _message("assistant", " \n ")],
        [ToolCallResult(tool_call_id="tc1", content="tool output", timestamp=_TS_MS)],
    ],
)
async def test_no_chat_text_returns_none_without_a_call(items: Sequence[ConversationItem]) -> None:
    result, fake = await _detect(items)

    assert result is None
    assert fake.calls == []


def test_sync_bridge_returns_the_same_result() -> None:
    fake = FakeLLMClient(responses=[_reply("English")])

    assert LanguageDetector(llm=fake).detect([_message("user", "hello")]) is OutputLanguage.ENGLISH


# ── Errors ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "reply", ["not json", _reply("French"), _reply("Korean"), json.dumps({"language": "Chinese", "why": "x"}), "{}"]
)
async def test_invalid_reply_raises_llm_output_error(reply: str) -> None:
    with pytest.raises(LLMOutputError):
        await _detect([_message("user", "hi")], reply=reply)


async def test_client_exception_propagates_unchanged() -> None:
    error = TimeoutError("upstream timed out")

    with pytest.raises(TimeoutError) as raised:
        await _detect([_message("user", "hi")], reply=error)

    assert raised.value is error


@pytest.mark.parametrize(("user_chars", "assistant_chars"), [(0, 100), (200, -1)])
async def test_a_cut_below_one_is_rejected_before_any_call(user_chars: int, assistant_chars: int) -> None:
    fake = FakeLLMClient(responses=[_reply("Chinese")])

    with pytest.raises(InvalidInputError):
        await LanguageDetector(llm=fake).adetect(
            [_message("user", "hi")], user_chars=user_chars, assistant_chars=assistant_chars
        )
    assert fake.calls == []


# ── Labels ─────────────────────────────────────────────────────────────────


def test_reply_labels_are_chinese_english_japanese_and_other() -> None:
    accepted = typing.get_args(_LanguageReply.model_fields["language"].annotation)

    assert sorted(accepted) == ["Chinese", "English", "Japanese", "Other"]
    assert sorted(LANGUAGE_LABELS) == sorted(accepted)
    assert all(OutputLanguage(label) for label in accepted if label != "Other")


def test_prompt_lists_every_label_once() -> None:
    reply_line = LANGUAGE_SYSTEM_PROMPT.rsplit("\n\n", 1)[1]
    listed = reply_line.removeprefix('Reply as JSON, {"language": "Chinese"}, using one of: ').removesuffix(".")

    assert listed.split(", ") == list(LANGUAGE_LABELS)


# ── Worked example ─────────────────────────────────────────────────────────


def _example() -> tuple[str, str]:
    """The prompt's worked example: its record body and its answer line."""
    body, rest = LANGUAGE_SYSTEM_PROMPT.split("<chat_record>\n", 1)[1].split("\n</chat_record>\n", 1)
    return body, rest.split("\n", 1)[0]


def test_example_record_is_rendered_as_the_detector_renders_a_record() -> None:
    body, _ = _example()
    user_block, assistant_block = body.split("\n\n")
    messages = [
        _message("user", user_block.removeprefix("[user] "), n=0),
        _message("assistant", assistant_block.removeprefix("[assistant] "), n=1),
    ]

    assert _render_chat_record(messages, keep=_cuts(user_chars=200, assistant_chars=100)) == body


def test_example_answer_is_a_valid_reply() -> None:
    _, answer = _example()

    assert _LanguageReply.model_validate_json(answer).language == "English"


# ── Log lines ──────────────────────────────────────────────────────────────


async def test_logs_counts_and_label_never_the_text(caplog: pytest.LogCaptureFixture) -> None:
    items = [_message("user", f"{_SECRET} 你好", n=0), _message("assistant", "你好", n=1)]

    with caplog.at_level("INFO", logger="everalgo.user_memory.language"):
        await _detect(items)

    logged = [record.getMessage() for record in caplog.records]
    assert logged[0].startswith("detecting language: 2 messages, ")
    assert logged[1] == "language detected: Chinese"
    assert all(_SECRET not in line for line in logged)
