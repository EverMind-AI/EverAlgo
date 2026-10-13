"""Tests for everalgo.agent_memory.case — AgentCaseExtractor.

Uses:
- :class:`everalgo.types.ConversationItem` discriminated union (ChatMessage / ToolCallRequest / ToolCallResult)
- :class:`everalgo.testing.fake_llm.FakeLLMClient` for deterministic LLM replays
- :mod:`everalgo.utils.tokens` shared tokenizer (no DI mock needed)

All private pipeline helpers operate directly on :class:`ConversationItem` objects; OpenAI-format dict
conversion happens only at LLM-prompt boundaries via :func:`_to_openai_dicts` / :func:`_dump_messages`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from everalgo.agent_memory import CaseSkipReason
from everalgo.agent_memory._replies import _CaseCompressReply, _CaseFilterReply, _ToolPreCompressReply
from everalgo.agent_memory.case import (
    AgentCaseExtractor,
    _apply_truncation,
    _calc_group_size,
    _calc_tool_content_size,
    _clamp_quality_score,
    _collect_tool_call_groups,
    _compress_experience,
    _compress_tool_chunk,
    _count_tool_call_rounds,
    _count_user_messages,
    _dump_messages,
    _has_tool_calls,
    _heuristic_trim,
    _is_worth_extracting,
    _pre_compress_to_list,
    _should_skip,
    _strip_before_first_user,
    _to_openai_dict,
    _to_openai_dicts,
)
from everalgo.agent_memory.prompts.case_compress import AGENT_CASE_COMPRESS_SYSTEM_PROMPT
from everalgo.agent_memory.prompts.case_filter import AGENT_CASE_FILTER_SYSTEM_PROMPT
from everalgo.agent_memory.prompts.tool_pre_compress import AGENT_TOOL_PRE_COMPRESS_SYSTEM_PROMPT
from everalgo.llm import LLMOutputError
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import (
    ChatMessage,
    ConversationItem,
    MemCell,
    ToolCall,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)

# ── Fixture helpers — typed ConversationItem instances throughout ────────────────────────────────────


def _long_text(token_count: int) -> str:
    """Generate text approximately ``token_count`` tokens long (``word_0 word_1 ...``)."""
    return " ".join(f"word_{i}" for i in range(token_count))


def _user_msg(content: str, ts: int = 1700000000000) -> ChatMessage:
    return ChatMessage(id="u", role="user", content=content, timestamp=ts, sender_id="user")


def _assistant_msg(content: str, ts: int = 1700000001000) -> ChatMessage:
    return ChatMessage(id="a", role="assistant", content=content, timestamp=ts, sender_id="assistant")


def _tool_call_msg(
    content: str | None = None,
    arguments: str = "{}",
    name: str = "search",
    call_id: str = "call_1",
    ts: int = 1700000000500,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_calls=[ToolCall(id=call_id, function=ToolCallFunction(name=name, arguments=arguments))],
        content=content,
        timestamp=ts,
        sender_id="assistant",
    )


def _tool_response_msg(content: str = "result", call_id: str = "call_1", ts: int = 1700000000600) -> ToolCallResult:
    return ToolCallResult(tool_call_id=call_id, content=content, timestamp=ts)


def _high_volume_tool_msgs(n: int = 21, base_ts: int = 1700000000100) -> list[ConversationItem]:
    """Build ``n`` paired (ToolCallRequest, ToolCallResult) messages — enough to trigger fast-pass.

    Each ToolCallRequest counts as one round, so ``n > the default complex-task threshold`` flags
    the trajectory as a complex task and skips the LLM filter (see case.py step 7).
    """
    msgs: list[ConversationItem] = []
    for i in range(n):
        call_id = f"call_{i}"
        msgs.append(_tool_call_msg(content="thinking", call_id=call_id, ts=base_ts + i * 2))
        msgs.append(_tool_response_msg("ok", call_id=call_id, ts=base_ts + i * 2 + 1))
    return msgs


# ── _should_skip ────────────────────────────────────────────────────────────────────────────────────


class TestShouldSkip:
    def test_empty_returns_skip_reason(self) -> None:
        assert _should_skip([]) is not None

    def test_no_user_messages(self) -> None:
        assert _should_skip([_assistant_msg("hello")]) is not None

    def test_no_assistant_messages(self) -> None:
        assert _should_skip([_user_msg("hi")]) is not None

    def test_incomplete_last_is_tool_call(self) -> None:
        msgs: list[ConversationItem] = [_user_msg("do X"), _tool_call_msg("thinking")]
        skip = _should_skip(msgs)
        assert skip is not None
        assert skip[0] is CaseSkipReason.TRAJECTORY_NOT_CLOSED
        assert skip[1] == {"last_item_kind": "tool_call"}

    def test_incomplete_last_is_tool_response(self) -> None:
        msgs: list[ConversationItem] = [_user_msg("do X"), _tool_call_msg(), _tool_response_msg()]
        assert _should_skip(msgs) is not None

    def test_single_turn_no_tools_skipped(self) -> None:
        msgs: list[ConversationItem] = [_user_msg("hi"), _assistant_msg("hello")]
        assert _should_skip(msgs) is not None

    def test_multi_turn_no_tools_passes(self) -> None:
        """> FILTER_NO_TOOL_MAX_MESSAGES (4) + assistant tokens >= FILTER_NO_TOOL_MIN_ASSISTANT (200)."""
        long_response = (
            "The TypeError on line 5 is caused by passing a string where an integer is expected. "
            "Wrap the input with int() before passing it to calculate(). Specifically change "
            "calculate(user_input) to calculate(int(user_input)). This happens because input() always "
            "returns a string in Python 3, even when the user types a number. The int() call converts "
            "it properly. You should also add error handling with a try/except ValueError block around "
            "the int() conversion to handle malformed user input gracefully."
        )
        msgs: list[ConversationItem] = [
            _user_msg("help debug"),
            _assistant_msg(long_response),
            _user_msg("TypeError on line 5"),
            _assistant_msg(long_response),
            _user_msg("thanks"),
            _assistant_msg("glad it worked"),
        ]
        assert _should_skip(msgs) is None

    def test_single_round_tool_passes(self) -> None:
        msgs: list[ConversationItem] = [
            _user_msg("search docs"),
            _tool_call_msg("let me search"),
            _tool_response_msg("found docs"),
            _assistant_msg("here are the results"),
        ]
        assert _should_skip(msgs) is None


# ── _strip_before_first_user ────────────────────────────────────────────────────────────────────────


class TestStripBeforeFirstUser:
    def test_already_user_first(self) -> None:
        msgs: list[ConversationItem] = [_user_msg("hi"), _assistant_msg("hello")]
        assert len(_strip_before_first_user(msgs)) == 2

    def test_drops_leading_non_user(self) -> None:
        """Leading assistant messages (e.g. system framing relayed as assistant) are dropped."""
        msgs: list[ConversationItem] = [_assistant_msg("setup"), _user_msg("hi"), _assistant_msg("hello")]
        result = _strip_before_first_user(msgs)
        assert len(result) == 2
        assert isinstance(result[0], ChatMessage) and result[0].role == "user"

    def test_no_user_returns_empty(self) -> None:
        msgs: list[ConversationItem] = [_assistant_msg("hi")]
        assert _strip_before_first_user(msgs) == []


# ── _has_tool_calls / _count_tool_call_rounds ───────────────────────────────────────────────────────


class TestToolCallHelpers:
    def test_has_tool_calls_via_assistant(self) -> None:
        msgs: list[ConversationItem] = [_user_msg("x"), _tool_call_msg(), _tool_response_msg(), _assistant_msg("done")]
        assert _has_tool_calls(msgs) is True

    def test_has_tool_calls_via_tool_response_only(self) -> None:
        msgs: list[ConversationItem] = [_user_msg("x"), _tool_response_msg("ok"), _assistant_msg("done")]
        assert _has_tool_calls(msgs) is True

    def test_has_tool_calls_false_plain_chat(self) -> None:
        assert _has_tool_calls([_user_msg("hi"), _assistant_msg("hello")]) is False

    def test_has_tool_calls_empty(self) -> None:
        assert _has_tool_calls([]) is False

    def test_count_tool_call_rounds_two(self) -> None:
        msgs: list[ConversationItem] = [
            _user_msg("x"),
            _tool_call_msg(),
            _tool_response_msg(),
            _tool_call_msg(call_id="call_2", ts=1700000000700),
            _tool_response_msg(call_id="call_2", ts=1700000000800),
            _assistant_msg("done"),
        ]
        assert _count_tool_call_rounds(msgs) == 2

    def test_count_tool_call_rounds_zero(self) -> None:
        assert _count_tool_call_rounds([_user_msg("hi"), _assistant_msg("hi")]) == 0

    def test_count_tool_call_rounds_parallel_calls_count_as_one_round(self) -> None:
        """A single ToolCallRequest with multiple parallel tool_calls is still one round."""
        msg = ToolCallRequest(
            tool_calls=[
                ToolCall(id="c1", function=ToolCallFunction(name="f1", arguments="{}")),
                ToolCall(id="c2", function=ToolCallFunction(name="f2", arguments="{}")),
                ToolCall(id="c3", function=ToolCallFunction(name="f3", arguments="{}")),
            ],
            content=None,
            timestamp=1,
            sender_id="assistant",
        )
        assert _count_tool_call_rounds([_user_msg("x"), msg, _assistant_msg("done")]) == 1

    def test_count_user_messages_basic(self) -> None:
        msgs: list[ConversationItem] = [
            _user_msg("first", ts=1),
            _assistant_msg("a", ts=2),
            _user_msg("second", ts=3),
            _assistant_msg("b", ts=4),
        ]
        assert _count_user_messages(msgs) == 2

    def test_count_user_messages_ignores_tool_messages(self) -> None:
        """ToolCallRequest / ToolCallResult must not be counted as user messages."""
        msgs: list[ConversationItem] = [
            _user_msg("only user", ts=1),
            _tool_call_msg(ts=2),
            _tool_response_msg(ts=3),
            _assistant_msg("done", ts=4),
        ]
        assert _count_user_messages(msgs) == 1

    def test_count_user_messages_empty(self) -> None:
        assert _count_user_messages([]) == 0


# ── _calc_tool_content_size ─────────────────────────────────────────────────────────────────────────


class TestCalcToolContentSize:
    def test_tool_message_counts_content(self) -> None:
        assert _calc_tool_content_size(_tool_response_msg("some tool output here")) > 0

    def test_assistant_with_tool_calls_counts_arguments(self) -> None:
        assert _calc_tool_content_size(_tool_call_msg(arguments='{"query": "x"}')) > 0

    def test_user_message_returns_zero(self) -> None:
        assert _calc_tool_content_size(_user_msg("some text")) == 0

    def test_plain_assistant_returns_zero(self) -> None:
        assert _calc_tool_content_size(_assistant_msg("text")) == 0

    def test_multiple_tool_calls_sum(self) -> None:
        msg = ToolCallRequest(
            tool_calls=[
                ToolCall(id="c1", function=ToolCallFunction(name="f1", arguments='{"a": 1}')),
                ToolCall(id="c2", function=ToolCallFunction(name="f2", arguments='{"b": 2}')),
            ],
            content="",
            timestamp=1,
            sender_id="assistant",
        )
        single = _calc_tool_content_size(_tool_call_msg(arguments='{"a": 1}'))
        assert _calc_tool_content_size(msg) > single


# ── _collect_tool_call_groups ───────────────────────────────────────────────────────────────────────


class TestCollectToolCallGroups:
    def test_single_group(self) -> None:
        items: list[ConversationItem] = [
            _user_msg("hi"),
            _tool_call_msg(),
            _tool_response_msg(),
            _assistant_msg("done"),
        ]
        assert _collect_tool_call_groups(items) == [[1, 2]]

    def test_multiple_groups(self) -> None:
        items: list[ConversationItem] = [
            _user_msg("hi"),
            _tool_call_msg(),
            _tool_response_msg(),
            _tool_call_msg(call_id="call_2", ts=1700000000700),
            _tool_response_msg(call_id="call_2", ts=1700000000800),
            _tool_response_msg(call_id="call_2", ts=1700000000900),
            _assistant_msg("done"),
        ]
        assert _collect_tool_call_groups(items) == [[1, 2], [3, 4, 5]]

    def test_no_tool_calls(self) -> None:
        assert _collect_tool_call_groups([_user_msg("hi"), _assistant_msg("hi")]) == []

    def test_empty(self) -> None:
        assert _collect_tool_call_groups([]) == []

    def test_tool_call_without_response(self) -> None:
        items: list[ConversationItem] = [_user_msg("hi"), _tool_call_msg(), _assistant_msg("done")]
        assert _collect_tool_call_groups(items) == [[1]]


# ── _calc_group_size ────────────────────────────────────────────────────────────────────────────────


class TestCalcGroupSize:
    def test_group_size_is_sum(self) -> None:
        items: list[ConversationItem] = [
            _tool_call_msg(arguments='{"query": "test search"}'),
            _tool_response_msg("tool result here"),
        ]
        expected = sum(_calc_tool_content_size(items[i]) for i in (0, 1))
        assert _calc_group_size(items, [0, 1]) == expected

    def test_empty_group_is_zero(self) -> None:
        assert _calc_group_size([_tool_call_msg(), _tool_response_msg()], []) == 0


# ── _clamp_quality_score ────────────────────────────────────────────────────────────────────────────


class TestClampQualityScore:
    def test_valid_float(self) -> None:
        assert _clamp_quality_score(0.5) == 0.5

    def test_clamps_above_one(self) -> None:
        assert _clamp_quality_score(1.5) == 1.0

    def test_clamps_below_zero(self) -> None:
        assert _clamp_quality_score(-0.3) == 0.0

    def test_bounds_are_kept(self) -> None:
        assert (_clamp_quality_score(0.0), _clamp_quality_score(1.0)) == (0.0, 1.0)


# ── _heuristic_trim ─────────────────────────────────────────────────────────────────────────────────


class TestHeuristicTrim:
    def test_short_content_unchanged(self) -> None:
        msgs: list[ConversationItem] = [
            _user_msg("hi"),
            _tool_call_msg("thinking", arguments='{"q": "x"}'),
            _tool_response_msg("short result"),
            _assistant_msg("done"),
        ]
        trimmed, _ = _heuristic_trim(msgs)
        assert trimmed[2].content == "short result"  # type: ignore[union-attr]

    def test_long_tool_output_trimmed_when_scale_kicks_in(self) -> None:
        """When total tokens exceed scale_trigger, oversized tool outputs are head+tail truncated."""
        long = _long_text(200_000)
        msgs: list[ConversationItem] = [
            _user_msg("search"),
            _tool_call_msg(),
            _tool_response_msg(long),
            _assistant_msg("done"),
        ]
        trimmed, total = _heuristic_trim(msgs)
        assert total > 100_000  # confirms scale_trigger should have fired
        assert trimmed[2].content is not None
        assert "[... trimmed" in trimmed[2].content  # type: ignore[union-attr]
        assert len(trimmed[2].content) < len(long)  # type: ignore[union-attr, arg-type]

    def test_returns_original_total_tokens(self) -> None:
        msgs: list[ConversationItem] = [_user_msg("hi"), _assistant_msg("hello")]
        _, total = _heuristic_trim(msgs)
        # total is the pre-trim token count of the JSON-dumped messages
        baseline = sum(len(m.content or "") for m in msgs)
        assert total > 0
        assert total >= baseline // 4  # tokens ~ chars/4 rough sanity


# ── _to_openai_dict / _to_openai_dicts ──────────────────────────────────────────────────────────────


class TestToOpenaiDicts:
    def test_basic_message_serialisation(self) -> None:
        msgs: list[ConversationItem] = [_user_msg("hi"), _assistant_msg("hello")]
        dicts = _to_openai_dicts(msgs)
        assert len(dicts) == 2
        assert dicts[0]["role"] == "user"
        assert dicts[0]["content"] == "hi"
        assert isinstance(dicts[0]["role"], str)

    def test_assistant_with_tool_calls_drops_none_content(self) -> None:
        msgs: list[ConversationItem] = [_tool_call_msg(content=None, arguments='{"q": "x"}')]
        dicts = _to_openai_dicts(msgs)
        assert "content" not in dicts[0]  # None content omitted (matches OpenAI wire format)
        assert dicts[0]["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'

    def test_tool_response_carries_tool_call_id(self) -> None:
        msgs: list[ConversationItem] = [_tool_response_msg("result")]
        dicts = _to_openai_dicts(msgs)
        assert dicts[0]["role"] == "tool"
        assert dicts[0]["tool_call_id"] == "call_1"

    def test_drops_everalgo_private_fields(self) -> None:
        """``timestamp`` / ``sender_id`` / ``sender_name`` must not appear in the dump."""
        msg = ChatMessage(
            id="msg_1",
            role="user",
            content="hi",
            timestamp=1700000000000,
            sender_id="u123",
            sender_name="Alice",
        )
        d = _to_openai_dict(msg)
        assert d == {"role": "user", "content": "hi"}
        assert "timestamp" not in d
        assert "sender_id" not in d
        assert "sender_name" not in d

    def test_json_dumpable(self) -> None:
        """Output must be json.dumps-able directly (no extra default= needed)."""
        msgs: list[ConversationItem] = [_user_msg("hi"), _tool_call_msg(arguments='{"a": 1}'), _tool_response_msg("ok")]
        dicts = _to_openai_dicts(msgs)
        json.dumps(dicts)  # should not raise

    def test_dump_messages_uses_only_openai_fields(self) -> None:
        """The canonical dump is what reaches the LLM — must not contain EverAlgo-private fields verbatim."""
        msg = ChatMessage(
            id="msg_2",
            role="user",
            content="hi",
            timestamp=1700000000000,
            sender_id="u123",
            sender_name="Alice",
        )
        dumped = _dump_messages([msg])
        assert '"timestamp"' not in dumped
        assert '"sender_id"' not in dumped
        assert '"sender_name"' not in dumped


# ── _is_worth_extracting (LLM) ──────────────────────────────────────────────────────────────────────


class TestIsWorthExtracting:
    async def test_exploration_signal_extracts(self) -> None:
        fake = FakeLLMClient(responses=['{"has_exploration": true, "has_user_correction": false, "reason": "r"}'])
        assert (await _is_worth_extracting("[]", 2, fake))[0] is True

    async def test_user_correction_signal_extracts_with_enough_user_messages(self) -> None:
        fake = FakeLLMClient(responses=['{"has_exploration": false, "has_user_correction": true, "reason": "r"}'])
        assert (await _is_worth_extracting("[]", 2, fake))[0] is True

    async def test_user_correction_rejected_on_single_user_message(self) -> None:
        """``has_user_correction=True`` with only 1 user message → treat as hallucination → False."""
        fake = FakeLLMClient(responses=['{"has_exploration": false, "has_user_correction": true, "reason": "r"}'])
        assert (await _is_worth_extracting("[]", 1, fake))[0] is False

    async def test_user_correction_rejected_still_lets_exploration_pass(self) -> None:
        """User-correction validation does not affect the exploration signal."""
        fake = FakeLLMClient(responses=['{"has_exploration": true, "has_user_correction": true, "reason": "r"}'])
        assert (await _is_worth_extracting("[]", 1, fake))[0] is True

    async def test_all_signals_false_filtered_out(self) -> None:
        fake = FakeLLMClient(
            responses=['{"has_exploration": false, "has_user_correction": false, "reason": "trivial"}']
        )
        assert (await _is_worth_extracting("[]", 2, fake))[0] is False

    async def test_llm_reason_is_returned_with_the_verdict(self) -> None:
        fake = FakeLLMClient(
            responses=['{"has_exploration": false, "has_user_correction": false, "reason": "linear lookup"}']
        )
        assert await _is_worth_extracting("[]", 2, fake) == (False, "linear lookup")

    async def test_requests_the_filter_reply_model(self) -> None:
        fake = FakeLLMClient(responses=['{"has_exploration": true, "has_user_correction": false, "reason": "r"}'])
        await _is_worth_extracting("[]", 2, fake)
        assert fake.calls[0].response_format is _CaseFilterReply

    @pytest.mark.parametrize(
        "reply",
        [
            "not valid json {{",
            '{"has_exploration": false, "reason": "r"}',
            '{"has_exploration": true, "has_user_correction": false, "reason": "r", "extra": 1}',
            '{"has_exploration": "yes", "has_user_correction": false, "reason": "r"}',
        ],
        ids=["not-json", "missing-field", "extra-field", "string-for-bool"],
    )
    async def test_invalid_reply_raises_llm_output_error(self, reply: str) -> None:
        """A missing signal is no longer read as False: the reply is rejected and the host decides on a retry."""
        fake = FakeLLMClient(responses=[reply])
        with pytest.raises(LLMOutputError, match="agent case filter reply is not a valid _CaseFilterReply"):
            await _is_worth_extracting("[]", 2, fake)


# ── End-to-end AgentCaseExtractor.aextract ──────────────────────────────────────────────────────────


class TestAgentCaseExtractorAExtract:
    @staticmethod
    def _memcell(messages: list[Any]) -> MemCell:
        ts = messages[-1].timestamp if messages else 1700000000000
        return MemCell(items=messages, timestamp=ts)

    async def test_empty_messages_returns_empty_list(self) -> None:
        fake = FakeLLMClient(responses=[])  # no LLM call expected
        cases = await AgentCaseExtractor(llm=fake).aextract(MemCell(items=[], timestamp=1700000000000))
        assert cases == []

    async def test_single_turn_no_tool_skipped_before_llm(self) -> None:
        """Pre-filter (no tool + ≤4 messages) skips without any LLM call."""
        fake = FakeLLMClient(responses=[])
        msgs: list[ConversationItem] = [_user_msg("hi"), _assistant_msg("hello")]
        cases = await AgentCaseExtractor(llm=fake).aextract(self._memcell(msgs))
        assert cases == []
        assert fake.call_count == 0

    async def test_high_tool_call_volume_fast_pass_to_compress(self) -> None:
        """tool_calls > the default complex-task threshold → fast-pass as complex task; only the compress LLM call."""
        compress_response = json.dumps(
            {
                "task_intent": "Build API",
                "approach": "1. design 2. implement 3. test",
                "quality_score": 0.85,
                "key_insight": "Use caching",
            }
        )
        fake = FakeLLMClient(responses=[compress_response])
        msgs: list[ConversationItem] = [_user_msg("build an API"), *_high_volume_tool_msgs()]
        msgs.append(_assistant_msg("Done! I built the API.", ts=1700000099999))
        cases = await AgentCaseExtractor(llm=fake).aextract(self._memcell(msgs))
        assert len(cases) == 1
        case = cases[0]
        assert case.task_intent == "Build API"
        assert case.approach.startswith("1. design")
        assert case.quality_score == 0.85
        assert case.key_insight == "Use caching"
        # Caller embeds — vector must remain unset (DESIGN.md §5.2)
        assert case.model_dump().get("vector") is None
        # parent_id / parent_type removed from AgentCase schema
        assert not hasattr(case, "parent_id") or case.model_dump().get("parent_id") is None
        # Exactly one LLM call — the compress; filter was bypassed via fast-pass
        assert fake.call_count == 1

    async def test_single_round_tool_runs_filter_then_compress(self) -> None:
        """Low tool-call volume triggers filter; any True signal then runs compress."""
        compress_response = json.dumps(
            {
                "task_intent": "search",
                "approach": "1. search 2. answer",
                "key_insight": "use proper keywords",
                "quality_score": 0.7,
            }
        )
        fake = FakeLLMClient(
            responses=[
                '{"has_exploration": true, "has_user_correction": false, "reason": "r"}',
                compress_response,
            ]
        )
        msgs: list[ConversationItem] = [
            _user_msg("look up X"),
            _tool_call_msg(content="searching"),
            _tool_response_msg("hit"),
            _assistant_msg("Here is X."),
        ]
        cases = await AgentCaseExtractor(llm=fake, min_tool_call_rounds=0).aextract(self._memcell(msgs))
        assert len(cases) == 1
        assert fake.call_count == 2

    async def test_filter_rejects_short_circuits_extraction(self) -> None:
        """All three signals False → no compress call, no AgentCase."""
        fake = FakeLLMClient(
            responses=['{"has_exploration": false, "has_user_correction": false, "reason": "trivial"}']
        )
        msgs: list[ConversationItem] = [
            _user_msg("hi"),
            _tool_call_msg(content="search"),
            _tool_response_msg("nothing useful"),
            _assistant_msg("got nothing"),
        ]
        cases = await AgentCaseExtractor(llm=fake, min_tool_call_rounds=0).aextract(self._memcell(msgs))
        assert cases == []
        assert fake.call_count == 1  # only the filter call

    async def test_compress_empty_task_intent_yields_no_case(self) -> None:
        """Empty task_intent → return [] regardless."""
        compress_response = json.dumps({"task_intent": "", "approach": "x", "key_insight": "", "quality_score": 0.5})
        fake = FakeLLMClient(
            responses=[
                '{"has_exploration": true, "has_user_correction": false, "reason": "r"}',
                compress_response,
            ]
        )
        msgs: list[ConversationItem] = [
            _user_msg("hi"),
            _tool_call_msg(content="search"),
            _tool_response_msg("hit"),
            _assistant_msg("done"),
        ]
        cases = await AgentCaseExtractor(llm=fake).aextract(self._memcell(msgs))
        assert cases == []

    async def test_filter_and_compress_send_static_system_and_data_only_user(self) -> None:
        """Each call is ``[system, user]``: the system is the shipped constant, the user holds only the data."""
        compress_response = json.dumps({"task_intent": "T", "approach": "A", "key_insight": "k", "quality_score": 0.6})
        fake = FakeLLMClient(
            responses=['{"has_exploration": true, "has_user_correction": false, "reason": "r"}', compress_response]
        )
        msgs: list[ConversationItem] = [
            _user_msg("look up X"),
            _tool_call_msg(content="searching"),
            _tool_response_msg("hit"),
            _assistant_msg("Here is X."),
        ]
        await AgentCaseExtractor(llm=fake, min_tool_call_rounds=0).aextract(self._memcell(msgs))
        filter_call, compress_call = fake.calls
        assert [m.role for m in filter_call.messages] == ["system", "user"]
        assert filter_call.messages[0].content == AGENT_CASE_FILTER_SYSTEM_PROMPT
        assert isinstance(filter_call.messages[1].content, str)
        assert filter_call.messages[1].content.startswith("<conversation>\n[")
        assert filter_call.response_format is _CaseFilterReply
        assert [m.role for m in compress_call.messages] == ["system", "user"]
        assert compress_call.messages[0].content == AGENT_CASE_COMPRESS_SYSTEM_PROMPT
        assert isinstance(compress_call.messages[1].content, str)
        assert compress_call.messages[1].content.startswith("<trajectory>\n[")
        assert compress_call.response_format is _CaseCompressReply

    async def test_monkey_patch_system_prompt_override_still_works(self) -> None:
        """The re-exported system constants can be monkey-patched at startup to replace the rules."""
        import everalgo.agent_memory.case as case_mod

        original = case_mod.AGENT_CASE_COMPRESS_SYSTEM_PROMPT
        case_mod.AGENT_CASE_COMPRESS_SYSTEM_PROMPT = "MONKEY_PATCH_COMPRESS_RULES"
        try:
            compress_response = json.dumps(
                {"task_intent": "T", "approach": "A", "key_insight": "key", "quality_score": 0.6}
            )
            fake = FakeLLMClient(responses=[compress_response])
            msgs: list[ConversationItem] = [_user_msg("solve X"), *_high_volume_tool_msgs()]
            msgs.append(_assistant_msg("Done.", ts=1700000099999))
            await AgentCaseExtractor(llm=fake).aextract(self._memcell(msgs))
            assert fake.calls[0].messages[0].content == "MONKEY_PATCH_COMPRESS_RULES"
        finally:
            case_mod.AGENT_CASE_COMPRESS_SYSTEM_PROMPT = original

    async def test_prompt_payload_strips_everalgo_private_fields(self) -> None:
        """LLM prompt must not contain ``timestamp`` / ``sender_id`` / ``sender_name`` (OpenAI fields only)."""
        compress_response = json.dumps(
            {"task_intent": "T", "approach": "A", "key_insight": "key", "quality_score": 0.6}
        )
        fake = FakeLLMClient(responses=[compress_response])
        msgs: list[ConversationItem] = [
            ChatMessage(
                id="msg_u",
                role="user",
                content="solve X",
                timestamp=1700000000000,
                sender_id="user_42",
                sender_name="Alice",
            ),
            *_high_volume_tool_msgs(),
            _assistant_msg("Done.", ts=1700000099999),
        ]
        await AgentCaseExtractor(llm=fake).aextract(self._memcell(msgs))
        prompt_text = fake.calls[0].messages[1].content
        assert "user_42" not in prompt_text
        assert "Alice" not in prompt_text
        assert "sender_id" not in prompt_text
        assert "sender_name" not in prompt_text
        assert "timestamp" not in prompt_text


# ── _should_skip — missing branches: ≤4 msgs no-tool, brief assistant tokens ─────────────────────────


class TestShouldSkipNoToolBranches:
    def test_no_tool_too_few_messages_skipped(self) -> None:
        """≥ 2 users + no tools + len(messages) <= FILTER_NO_TOOL_MAX_MESSAGES (4) → skip with msg-count reason."""
        # 4 messages, 2 users, no tools — hits the "<= 4 messages" branch
        msgs: list[ConversationItem] = [
            _user_msg("hi"),
            _assistant_msg("hello"),
            _user_msg("again"),
            _assistant_msg("yes"),
        ]
        skip = _should_skip(msgs)
        assert skip is not None
        assert skip[0] is CaseSkipReason.NO_TOOL_TOO_FEW_MESSAGES
        assert skip[1] == {"messages": 4, "min": 5}

    def test_no_tool_brief_assistant_tokens_skipped(self) -> None:
        """> 4 msgs + no tools + assistant tokens < FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS (200) → skip."""
        msgs: list[ConversationItem] = [
            _user_msg("hi 1"),
            _assistant_msg("ok"),  # short
            _user_msg("hi 2"),
            _assistant_msg("yes"),  # short
            _user_msg("hi 3"),
            _assistant_msg("done"),  # short — total tokens << 200
        ]
        skip = _should_skip(msgs)
        assert skip is not None
        assert skip[0] is CaseSkipReason.NO_TOOL_ASSISTANT_TOO_SHORT
        assert skip[1]["min"] == 200


# ── _apply_truncation — edge cases on tool_calls / assistant content ─────────────────────────────────


class TestApplyTruncationToolCallEdgeCases:
    def test_tool_call_with_empty_arguments_skipped(self) -> None:
        """Per case.py ``if args:`` False branch — empty arguments string skips truncation."""
        msgs: list[ConversationItem] = [
            ToolCallRequest(
                tool_calls=[ToolCall(id="call_1", function=ToolCallFunction(name="f", arguments=""))],
                content=None,
                timestamp=1,
                sender_id="assistant",
            ),
        ]
        result = _apply_truncation(msgs, max_tool_output=1000, max_tool_args=100, max_assistant=3000)
        assert isinstance(result[0], ToolCallRequest)
        assert result[0].tool_calls[0].function.arguments == ""

    def test_tool_call_with_long_arguments_trimmed(self) -> None:
        """Long arguments string is truncated to max_tool_args tokens."""
        msgs: list[ConversationItem] = [
            ToolCallRequest(
                tool_calls=[ToolCall(id="call_1", function=ToolCallFunction(name="x", arguments="a" * 5000))],
                content=None,
                timestamp=1,
                sender_id="assistant",
            ),
        ]
        result = _apply_truncation(msgs, max_tool_output=1000, max_tool_args=100, max_assistant=3000)
        assert isinstance(result[0], ToolCallRequest)
        assert len(result[0].tool_calls[0].function.arguments) < 5000  # trimmed


# ── _pre_compress_to_list — full body coverage ───────────────────────────────────────────────────────


def _huge_text(tokens: int) -> str:
    """Generate text ≈ ``tokens`` tokens long (each `word_N ` is roughly 2 tokens)."""
    return " ".join(f"word_{i}" for i in range(tokens))


class TestPreCompressToList:
    async def test_no_tool_call_groups_returns_unchanged(self) -> None:
        """Plain conversation (no tool-call groups) returns input unchanged + makes no LLM call."""
        fake = FakeLLMClient(responses=[])
        msgs: list[ConversationItem] = [_user_msg("hi"), _assistant_msg("hello")]
        result = await _pre_compress_to_list(msgs, fake)
        assert [m.content for m in result] == [m.content for m in msgs]
        assert fake.call_count == 0

    async def test_under_chunk_size_returns_unchanged(self) -> None:
        """Total tool content <= chunk size → return unchanged, no LLM call (early-return path)."""
        fake = FakeLLMClient(responses=[])
        msgs: list[ConversationItem] = [
            _user_msg("hi"),
            _tool_call_msg("thinking", arguments='{"q": "x"}'),
            _tool_response_msg("short result"),
            _assistant_msg("done"),
        ]
        result = await _pre_compress_to_list(msgs, fake)
        assert [m.content for m in result] == [m.content for m in msgs]
        assert fake.call_count == 0

    async def test_oversized_triggers_selective_compression(self) -> None:
        """Two large groups + sum > chunk size → both get selectively compressed via LLM."""
        big_response_1 = _huge_text(60_000)
        big_response_2 = _huge_text(60_000)

        def _make_compressed_payload(call_id: str = "call_1") -> str:
            return json.dumps(
                {
                    "compressed_messages": [
                        {
                            "role": "assistant",
                            "content": "compressed thinking",
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": "search", "arguments": '{"q":"x"}'},
                                }
                            ],
                        },
                        {"role": "tool", "content": "[compressed]", "tool_call_id": call_id},
                    ]
                }
            )

        # Each group is ~60k tokens → two separate chunks → two LLM calls needed.
        fake = FakeLLMClient(
            responses=[
                _make_compressed_payload("call_1"),
                _make_compressed_payload("call_2"),
            ]
        )
        msgs: list[ConversationItem] = [
            _user_msg("hi"),
            _tool_call_msg("thinking 1", arguments='{"q":"a"}', ts=1700000000100),
            _tool_response_msg(big_response_1, ts=1700000000200),
            _tool_call_msg("thinking 2", arguments='{"q":"b"}', call_id="call_2", ts=1700000000300),
            _tool_response_msg(big_response_2, call_id="call_2", ts=1700000000400),
            _assistant_msg("done", ts=1700000000500),
        ]
        result = await _pre_compress_to_list(msgs, fake)
        assert fake.call_count >= 1
        all_contents = " ".join(str(m.content or "") for m in result)
        assert "[compressed]" in all_contents

    async def test_compression_raises_propagates_error(self) -> None:
        """LLM raises during chunk compression → error propagates from _pre_compress_to_list."""

        def raise_llm_error(*_args: object, **_kwargs: object) -> str:
            raise RuntimeError("simulated LLM error during chunk compression")

        fake = FakeLLMClient(handler=raise_llm_error)
        big = _huge_text(60_000)
        msgs: list[ConversationItem] = [
            _user_msg("hi"),
            _tool_call_msg("thinking 1", arguments='{"q":"a"}', ts=1700000000100),
            _tool_response_msg(big, ts=1700000000200),
            _tool_call_msg("thinking 2", arguments='{"q":"b"}', call_id="call_2", ts=1700000000300),
            _tool_response_msg(big, call_id="call_2", ts=1700000000400),
            _assistant_msg("done", ts=1700000000500),
        ]
        with pytest.raises(RuntimeError, match="simulated LLM error"):
            await _pre_compress_to_list(msgs, fake)

    async def test_malformed_chunk_reply_raises_llm_output_error(self) -> None:
        """A chunk reply that is not JSON → LLMOutputError (handler always bad so asyncio.gather covers 2 groups)."""

        def _bad_handler(messages: object, **_: object) -> str:
            return "not valid json {{"

        fake = FakeLLMClient(handler=_bad_handler)
        big = _huge_text(60_000)
        msgs: list[ConversationItem] = [
            _user_msg("hi"),
            _tool_call_msg("thinking 1", arguments='{"q":"a"}', ts=1700000000100),
            _tool_response_msg(big, ts=1700000000200),
            _tool_call_msg("thinking 2", arguments='{"q":"b"}', call_id="call_2", ts=1700000000300),
            _tool_response_msg(big, call_id="call_2", ts=1700000000400),
            _assistant_msg("done", ts=1700000000500),
        ]
        with pytest.raises(LLMOutputError):
            await _pre_compress_to_list(msgs, fake)

    async def test_compressed_count_mismatch_raises_llm_output_error(self) -> None:
        """LLM returns a list of wrong length → the business check rejects the chunk reply.

        Uses a single large tool group so there's exactly one chunk → one LLM call.
        """
        # One large group (2 messages) but wrong_count_payload returns 1 item → shape mismatch.
        wrong_count_payload = json.dumps(
            {"compressed_messages": [{"role": "tool", "content": "just one", "tool_call_id": "call_1"}]}
        )
        fake = FakeLLMClient(responses=[wrong_count_payload])
        big = _huge_text(60_000)
        # Single group: [tool_call, tool_response] (2 messages); wrong_count has 1 → mismatch.
        msgs: list[ConversationItem] = [
            _user_msg("hi"),
            _tool_call_msg("thinking 1", arguments='{"q":"a"}', ts=1700000000100),
            _tool_response_msg(big, ts=1700000000200),
            _assistant_msg("done", ts=1700000000500),
        ]
        with pytest.raises(LLMOutputError, match="holds 1 messages for a chunk of 2"):
            await _pre_compress_to_list(msgs, fake)


# ── _compress_tool_chunk — direct unit coverage ──────────────────────────────────────────────────────


class TestCompressToolChunk:
    async def test_success_returns_compressed_list(self) -> None:
        msgs: list[ConversationItem] = [
            _tool_call_msg("thinking", arguments='{"q":"x"}'),
            _tool_response_msg("hit"),
        ]
        payload = json.dumps(
            {
                "compressed_messages": [
                    {
                        "role": "assistant",
                        "content": "compressed",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q":"x"}'},
                            }
                        ],
                    },
                    {"role": "tool", "content": "[compressed]", "tool_call_id": "call_1"},
                ]
            }
        )
        fake = FakeLLMClient(responses=[payload])
        result = await _compress_tool_chunk(msgs, fake)
        assert fake.calls[0].response_format is _ToolPreCompressReply
        assert len(result) == 2
        assert isinstance(result[0], ToolCallRequest)
        assert isinstance(result[1], ToolCallResult)
        assert result[1].content == "[compressed]"
        # Timestamps from originals are preserved across the compressed rebuild
        assert result[0].timestamp == msgs[0].timestamp
        assert result[1].timestamp == msgs[1].timestamp

    async def test_compressed_messages_may_omit_keys_their_role_does_not_carry(self) -> None:
        """A prompt-only reply copies the input shape, so a message leaves out the keys its role has no use for."""
        msgs: list[ConversationItem] = [_tool_call_msg(), _tool_response_msg()]
        payload = json.dumps(
            {
                "compressed_messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"id": "call_1", "type": "function", "function": {"name": "s", "arguments": "{}"}}
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "[x]"},
                ]
            }
        )
        result = await _compress_tool_chunk(msgs, FakeLLMClient(responses=[payload]))
        assert isinstance(result[0], ToolCallRequest)
        assert result[0].content is None

    async def test_schema_constrained_reply_with_explicit_nulls_is_accepted(self) -> None:
        """A schema-constrained reply writes every key, ``null`` where the role has nothing."""
        msgs: list[ConversationItem] = [_tool_response_msg("x")]
        payload = json.dumps(
            {"compressed_messages": [{"role": "tool", "content": "[x]", "tool_calls": None, "tool_call_id": "call_1"}]}
        )
        result = await _compress_tool_chunk(msgs, FakeLLMClient(responses=[payload]))
        assert isinstance(result[0], ToolCallResult)
        assert result[0].content == "[x]"

    @pytest.mark.parametrize(
        "reply",
        [
            "not json {{",
            '{"compressed_messages": [{"role": "tool", "content": "x", "tool_call_id": "call_1", "name": "n"}]}',
            '{"compressed_messages": [{"role": "system", "content": "x"}]}',
            '{"messages": []}',
        ],
        ids=["not-json", "extra-field", "unknown-role", "wrong-key"],
    )
    async def test_invalid_reply_raises_llm_output_error(self, reply: str) -> None:
        msgs: list[ConversationItem] = [_tool_response_msg("x")]
        with pytest.raises(LLMOutputError, match="agent tool pre-compress reply is not a valid _ToolPreCompressReply"):
            await _compress_tool_chunk(msgs, FakeLLMClient(responses=[reply]))

    async def test_compressed_list_wrong_length_raises(self) -> None:
        """Response has a list but wrong length → LLMOutputError."""
        msgs: list[ConversationItem] = [
            _tool_call_msg(),
            _tool_response_msg(),
            _tool_response_msg(call_id="call_2"),
        ]  # 3 input msgs
        payload = json.dumps(
            {
                "compressed_messages": [
                    {"role": "tool", "content": "x", "tool_call_id": "call_1"},
                    {"role": "tool", "content": "y", "tool_call_id": "call_2"},
                ]
            }
        )
        fake = FakeLLMClient(responses=[payload])
        with pytest.raises(LLMOutputError, match="holds 2 messages for a chunk of 3"):
            await _compress_tool_chunk(msgs, fake)

    async def test_tool_message_without_tool_call_id_raises(self) -> None:
        """A compressed tool message with no tool_call_id answers no tool call → LLMOutputError."""
        msgs: list[ConversationItem] = [_tool_response_msg("x")]
        payload = json.dumps({"compressed_messages": [{"role": "tool", "content": "x"}]})
        fake = FakeLLMClient(responses=[payload])
        with pytest.raises(LLMOutputError, match="tool message with no tool_call_id"):
            await _compress_tool_chunk(msgs, fake)

    async def test_chunk_sends_static_system_and_counted_messages_user(self) -> None:
        msgs: list[ConversationItem] = [_tool_response_msg("x")]
        payload = json.dumps({"compressed_messages": [{"role": "tool", "content": "[x]", "tool_call_id": "call_1"}]})
        fake = FakeLLMClient(responses=[payload])
        await _compress_tool_chunk(msgs, fake)
        system, user = fake.calls[0].messages
        assert (system.role, system.content) == ("system", AGENT_TOOL_PRE_COMPRESS_SYSTEM_PROMPT)
        assert user.role == "user"
        assert user.content == f'<messages count="1">\n{_dump_messages(msgs)}\n</messages>'


# ── _compress_experience — missing/empty field branches ──────────────────────────────────────────────


class TestCompressExperience:
    async def test_empty_task_intent_returns_none(self) -> None:
        """Response 'task_intent' is empty string → None."""
        fake = FakeLLMClient(
            responses=['{"task_intent": "", "approach": "x", "key_insight": "", "quality_score": 0.5}']
        )
        assert await _compress_experience("[]", fake) == (None, CaseSkipReason.COMPRESS_EMPTY_INTENT)

    async def test_empty_approach_returns_none(self) -> None:
        """Response 'approach' is empty → None."""
        fake = FakeLLMClient(
            responses=['{"task_intent": "t", "approach": "", "key_insight": "", "quality_score": 0.5}']
        )
        assert await _compress_experience("[]", fake) == (None, CaseSkipReason.COMPRESS_EMPTY_APPROACH)

    async def test_valid_reply_is_returned_with_no_reason(self) -> None:
        reply = '{"task_intent": "t", "approach": "a", "key_insight": "k", "quality_score": 1}'
        fake = FakeLLMClient(responses=[reply])
        assert await _compress_experience("[]", fake) == (
            _CaseCompressReply(task_intent="t", approach="a", key_insight="k", quality_score=1.0),
            None,
        )
        assert fake.calls[0].response_format is _CaseCompressReply

    @pytest.mark.parametrize(
        "reply",
        [
            "not json {{",
            '{"task_intent": "t", "approach": "a", "quality_score": 0.5}',
            '{"task_intent": "t", "approach": "a", "key_insight": "k", "quality_score": 0.5, "extra": 1}',
            '{"task_intent": "t", "approach": "a", "key_insight": "k", "quality_score": "high"}',
        ],
        ids=["not-json", "missing-field", "extra-field", "string-for-score"],
    )
    async def test_invalid_reply_raises_llm_output_error(self, reply: str) -> None:
        fake = FakeLLMClient(responses=[reply])
        with pytest.raises(LLMOutputError, match="agent case compress reply is not a valid _CaseCompressReply"):
            await _compress_experience("[]", fake)


# ── AgentCaseExtractor end-to-end: trim → log → return path ──────────────────────────────────────────


class TestAgentCaseExtractorTruncationLogging:
    async def test_task_intent_truncation_logged(self) -> None:
        """LLM-emitted long task_intent gets head-truncated to MAX_TASK_INTENT_TOKENS."""
        big_intent = " ".join(f"word_{i}" for i in range(400))  # > 300 tokens
        compress_response = json.dumps(
            {"task_intent": big_intent, "approach": "1. do thing", "key_insight": "key", "quality_score": 0.7}
        )
        fake = FakeLLMClient(responses=[compress_response])
        msgs: list[ConversationItem] = [_user_msg("solve"), *_high_volume_tool_msgs()]
        msgs.append(_assistant_msg("Done.", ts=1700000099999))
        cases = await AgentCaseExtractor(llm=fake).aextract(MemCell(items=msgs, timestamp=1700000099999))
        assert len(cases) == 1
        assert len(cases[0].task_intent) < len(big_intent)


# ── _apply_truncation — assistant content trimming ──────────────────────────────────────────────────


class TestApplyTruncationAssistantContent:
    def test_long_assistant_content_trimmed(self) -> None:
        """Assistant message with content longer than max_assistant gets trimmed."""
        long_content = _long_text(5000)
        msgs: list[ConversationItem] = [
            _user_msg("hi"),
            _assistant_msg(long_content),
        ]
        result = _apply_truncation(msgs, max_tool_output=1000, max_tool_args=800, max_assistant=200)
        assert isinstance(result[1], ChatMessage)
        assert result[1].content != long_content
        assert result[1].content is not None
        assert "[... trimmed" in result[1].content


# ── _pre_compress_to_list — break-out path on small subsequent groups ────────────────────────────────


class TestPreCompressToListBreakOut:
    async def test_break_when_estimated_drops_below_threshold(self) -> None:
        """Once estimated_total falls below chunk_size, the selection loop breaks.

        Build 1 huge group (drops estimated below threshold after one iteration) + several small groups
        whose token counts are tiny — they should NOT be selected for compression.
        """
        huge_response = _huge_text(70_000)  # ≈ 140K tokens, > PRE_COMPRESS_CHUNK_SIZE
        compressed_payload = json.dumps(
            {
                "compressed_messages": [
                    {
                        "role": "assistant",
                        "content": "compressed",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "search", "arguments": '{"q":"x"}'},
                            }
                        ],
                    },
                    {"role": "tool", "content": "[compressed]", "tool_call_id": "call_1"},
                ]
            }
        )
        fake = FakeLLMClient(responses=[compressed_payload])
        msgs: list[ConversationItem] = [
            _user_msg("hi"),
            # The huge group (gets selected for compression)
            _tool_call_msg("thinking huge", arguments='{"q":"a"}', ts=1700000000100),
            _tool_response_msg(huge_response, ts=1700000000200),
            # 3 small groups — should NOT be selected (break fires after huge is enough)
            _tool_call_msg("tiny 1", arguments='{"q":"b"}', call_id="call_2", ts=1700000000300),
            _tool_response_msg("tiny", call_id="call_2", ts=1700000000400),
            _tool_call_msg("tiny 2", arguments='{"q":"c"}', call_id="call_3", ts=1700000000500),
            _tool_response_msg("tiny", call_id="call_3", ts=1700000000600),
            _tool_call_msg("tiny 3", arguments='{"q":"d"}', call_id="call_4", ts=1700000000700),
            _tool_response_msg("tiny", call_id="call_4", ts=1700000000800),
            _assistant_msg("done", ts=1700000000900),
        ]
        await _pre_compress_to_list(msgs, fake)
        # Exactly one LLM call (the huge group); the small ones were left alone via break
        assert fake.call_count == 1


class TestAgentCaseExtractorBetweenThresholds:
    async def test_total_between_chunk_size_and_2x_does_not_bail(self) -> None:
        """Between-threshold path: total > chunk_size but trimmed <= 2x chunk_size."""
        user_chunk = _huge_text(3000)  # ≈ 11K tokens each
        msgs: list[Any] = []
        for i in range(12):
            msgs.append(_user_msg(user_chunk, ts=1700000000000 + i * 10))
            msgs.append(_assistant_msg("ack", ts=1700000000000 + i * 10 + 1))
        # High tool-call volume → fast-pass, skip LLM filter (only compress is called).
        msgs.extend(_high_volume_tool_msgs(base_ts=1700000099000))
        msgs.append(_assistant_msg("Done.", ts=1700000099999))

        compress_response = json.dumps(
            {"task_intent": "T", "approach": "A", "key_insight": "key", "quality_score": 0.7}
        )
        fake = FakeLLMClient(responses=[compress_response])
        cases = await AgentCaseExtractor(llm=fake).aextract(MemCell(items=msgs, timestamp=1700000099999))
        assert len(cases) == 1
        assert fake.call_count >= 1


class TestAgentCaseExtractorPostTrimBail:
    async def test_extremely_oversized_trajectory_bails_after_trim(self) -> None:
        """Post-trim oversize bail: trimmed total > 2x chunk_size returns []."""
        user_huge = _huge_text(3000)  # ≈ 6000 tokens each
        msgs: list[Any] = []
        for i in range(100):
            msgs.append(_user_msg(user_huge, ts=1700000000000 + i * 10))
            msgs.append(_assistant_msg("noted", ts=1700000000000 + i * 10 + 1))
        msgs.append(_tool_call_msg(content="thinking", ts=1700000099000))
        msgs.append(_tool_response_msg("result", ts=1700000099001))
        msgs.append(_assistant_msg("Final answer.", ts=1700000099999))
        fake = FakeLLMClient(responses=[])  # must NEVER be called — bail happens before any LLM
        cases = await AgentCaseExtractor(llm=fake).aextract(MemCell(items=msgs, timestamp=1700000099999))
        assert cases == []
        assert fake.call_count == 0


# ── Instance-level llm= binding ─────────────────────────────────────────────────────────────────────


def _minimal_multi_round_memcell() -> MemCell:
    """Minimal agent trajectory eligible for fast-pass (>the default complex-task threshold tool calls)."""
    msgs: list[Any] = [_user_msg("do the task"), *_high_volume_tool_msgs()]
    msgs.append(_assistant_msg("Done.", ts=1700000099999))
    return MemCell(items=msgs, timestamp=1700000099999)


async def test_aextract_uses_instance_llm_when_per_call_omitted() -> None:
    """Instance-level llm= is used when aextract() is called without a per-call llm= argument."""
    compress_response = json.dumps(
        {"task_intent": "Instance task", "approach": "approach A", "key_insight": "key", "quality_score": 0.8}
    )
    instance_fake = FakeLLMClient(responses=[compress_response])
    extractor = AgentCaseExtractor(llm=instance_fake)
    cases = await extractor.aextract(_minimal_multi_round_memcell())
    assert len(cases) == 1
    assert cases[0].task_intent == "Instance task"
    assert instance_fake.call_count == 1


# ── min_tool_call_rounds gate ───────────────────────────────────────────────────────────────────────


class TestMinToolCallRoundsGate:
    async def test_default_three_short_circuits_single_round(self) -> None:
        """Default ``min_tool_call_rounds=3`` rejects a single-round trajectory before any LLM call."""
        fake = FakeLLMClient(responses=[])  # must NEVER be called
        msgs: list[ConversationItem] = [
            _user_msg("look up X"),
            _tool_call_msg(content="searching"),
            _tool_response_msg("hit"),
            _assistant_msg("Here is X."),
        ]
        cases = await AgentCaseExtractor(llm=fake).aextract(MemCell(items=msgs, timestamp=1700000000500))
        assert cases == []
        assert fake.call_count == 0

    async def test_explicit_zero_disables_gate(self) -> None:
        """``min_tool_call_rounds=0`` lets a single-round trajectory through to the LLM filter."""
        fake = FakeLLMClient(responses=['{"has_exploration": false, "has_user_correction": false, "reason": "r"}'])
        msgs: list[ConversationItem] = [
            _user_msg("look up X"),
            _tool_call_msg(content="searching"),
            _tool_response_msg("hit"),
            _assistant_msg("Here is X."),
        ]
        cases = await AgentCaseExtractor(llm=fake, min_tool_call_rounds=0).aextract(
            MemCell(items=msgs, timestamp=1700000000500)
        )
        # filter rejects (all signals false), but the LLM call happened — gate did not short-circuit
        assert cases == []
        assert fake.call_count == 1

    async def test_below_threshold_returns_empty_without_llm_call(self) -> None:
        """``min_tool_call_rounds=3`` on a 1-round trajectory short-circuits before any LLM call."""
        fake = FakeLLMClient(responses=[])  # must NEVER be called
        msgs: list[ConversationItem] = [
            _user_msg("look up X"),
            _tool_call_msg(content="searching"),
            _tool_response_msg("hit"),
            _assistant_msg("Here is X."),
        ]
        extractor = AgentCaseExtractor(llm=fake, min_tool_call_rounds=3)
        cases = await extractor.aextract(MemCell(items=msgs, timestamp=1700000000500))
        assert cases == []
        assert fake.call_count == 0

    async def test_at_threshold_passes_gate(self) -> None:
        """``rounds == min_tool_call_rounds`` is allowed (strict ``<`` comparison)."""
        compress_response = json.dumps({"task_intent": "T", "approach": "A", "key_insight": "k", "quality_score": 0.7})
        fake = FakeLLMClient(
            responses=[
                '{"has_exploration": true, "has_user_correction": false, "reason": "r"}',
                compress_response,
            ]
        )
        # 2 rounds, threshold 2 → 2 < 2 is False → gate passes → filter + compress run.
        msgs: list[ConversationItem] = [
            _user_msg("solve"),
            _tool_call_msg(content="step 1", ts=1700000000100),
            _tool_response_msg("ok 1", ts=1700000000200),
            _tool_call_msg(content="step 2", call_id="call_2", ts=1700000000300),
            _tool_response_msg("ok 2", call_id="call_2", ts=1700000000400),
            _assistant_msg("Done.", ts=1700000000500),
        ]
        extractor = AgentCaseExtractor(llm=fake, min_tool_call_rounds=2)
        cases = await extractor.aextract(MemCell(items=msgs, timestamp=1700000000500))
        assert len(cases) == 1
        assert fake.call_count == 2


# ── complex_task_tool_call_round_threshold parameter ────────────────────────────────────────────────


class TestComplexTaskRoundThresholdParam:
    async def test_custom_threshold_triggers_fast_pass_below_default(self) -> None:
        """Custom threshold ``= 2`` fast-passes a 3-round trajectory that would normally run the filter."""
        compress_response = json.dumps({"task_intent": "T", "approach": "A", "key_insight": "k", "quality_score": 0.7})
        fake = FakeLLMClient(responses=[compress_response])
        # 3 rounds > custom threshold 2 → fast-pass, skip filter, only compress is called.
        msgs: list[ConversationItem] = [
            _user_msg("solve"),
            _tool_call_msg(content="step 1", ts=1700000000100),
            _tool_response_msg("ok 1", ts=1700000000200),
            _tool_call_msg(content="step 2", call_id="call_2", ts=1700000000300),
            _tool_response_msg("ok 2", call_id="call_2", ts=1700000000400),
            _tool_call_msg(content="step 3", call_id="call_3", ts=1700000000500),
            _tool_response_msg("ok 3", call_id="call_3", ts=1700000000600),
            _assistant_msg("Done.", ts=1700000000700),
        ]
        extractor = AgentCaseExtractor(llm=fake, complex_task_tool_call_round_threshold=2)
        cases = await extractor.aextract(MemCell(items=msgs, timestamp=1700000000700))
        assert len(cases) == 1
        assert fake.call_count == 1  # compress only, filter bypassed

    async def test_high_custom_threshold_forces_filter(self) -> None:
        """High threshold prevents fast-pass; even a 21-round trajectory runs the LLM filter."""
        fake = FakeLLMClient(responses=['{"has_exploration": false, "has_user_correction": false, "reason": "r"}'])
        msgs: list[ConversationItem] = [_user_msg("solve"), *_high_volume_tool_msgs()]
        msgs.append(_assistant_msg("Done.", ts=1700000099999))
        extractor = AgentCaseExtractor(llm=fake, complex_task_tool_call_round_threshold=100)
        cases = await extractor.aextract(MemCell(items=msgs, timestamp=1700000099999))
        # Filter rejects (all signals false) — but the LLM filter call happened because rounds < 100.
        assert cases == []
        assert fake.call_count == 1
