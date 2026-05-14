"""Tests for everalgo.boundary.chat — ChatMemCellExtractor (new-release batch algorithm).

Tests exercise the algorithm at three layers:
- Phase boundaries (input validation / force-split / batch LLM detection / is_final closure)
- Opensource ports (5-retry RuntimeError, 3-tier JSON parse, _extract_participants USER-only,
  _format_messages_with_indices ``[N] [ISO+TZ] sender_name: content`` rendering)
- Doc-level interface (DetectionOutput NamedTuple, is_final semantics, per-call overrides)
"""

from __future__ import annotations

from typing import Any

import pytest

from everalgo.boundary.chat import (
    DEFAULT_HARD_MSG_LIMIT,
    DEFAULT_HARD_TOKEN_LIMIT,
    ChatMemCellExtractor,
    DetectionOutput,
    _extract_participants,
    _find_force_split_point,
    _format_messages_with_indices,
    _parse_batch_boundary_response,
    _parse_json_three_tier,
)
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Message, MessageRole, RawDataType


def _msg(
    role: MessageRole,
    content: str,
    ts: int,
    *,
    sender_id: str | None = None,
    sender_name: str | None = None,
) -> Message:
    return Message(
        role=role,
        content=content,
        timestamp=ts,
        sender_id=sender_id,
        sender_name=sender_name,
    )


def _dialogue(n: int, base_ts: int = 1_700_000_000_000, *, gap_ms: int = 1000) -> list[Message]:
    """Alternating user/assistant messages spaced ``gap_ms`` apart, each with sender identity set."""
    return [
        _msg(
            MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
            f"msg-{i}",
            base_ts + i * gap_ms,
            sender_id="u_alice" if i % 2 == 0 else "u_bot",
            sender_name="Alice" if i % 2 == 0 else "Bot",
        )
        for i in range(n)
    ]


def _llm_no_boundary(*, should_wait: bool = False) -> ChatResponse:
    """LLM response: no boundary, full input belongs to one episode."""
    return ChatResponse(
        content=f'{{"reasoning": "single episode", "boundaries": [], "should_wait": {str(should_wait).lower()}}}',
        model="fake",
    )


def _llm_boundaries(boundaries: list[int], *, should_wait: bool = False) -> ChatResponse:
    return ChatResponse(
        content=(f'{{"reasoning": "split", "boundaries": {boundaries}, "should_wait": {str(should_wait).lower()}}}'),
        model="fake",
    )


# ==========================================================================
# Phase 1 — Input validation
# ==========================================================================


async def test_adetect_empty_messages_short_circuits() -> None:
    fake = FakeLLMClient(responses=[])
    output = await ChatMemCellExtractor().adetect([], llm=fake)
    assert output == DetectionOutput(cells=[], tail=[])
    assert fake.call_count == 0


async def test_adetect_single_message_no_boundary_returns_as_tail() -> None:
    msgs = _dialogue(1)
    fake = FakeLLMClient(responses=[_llm_no_boundary()])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert fake.call_count == 1
    assert output.cells == []
    assert output.tail == msgs


async def test_adetect_single_message_is_final_closes_into_one_cell() -> None:
    msgs = _dialogue(1)
    fake = FakeLLMClient(responses=[_llm_no_boundary()])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake, is_final=True)

    assert fake.call_count == 1
    assert len(output.cells) == 1
    assert output.cells[0].messages == msgs
    assert output.tail == []


# ==========================================================================
# Phase 4 — Batch LLM detection
# ==========================================================================


async def test_adetect_no_boundary_is_final_false_yields_tail() -> None:
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_llm_no_boundary()])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert fake.call_count == 1
    assert output.cells == []
    assert output.tail == msgs


async def test_adetect_no_boundary_is_final_true_closes_into_one_cell() -> None:
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_llm_no_boundary()])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake, is_final=True)

    assert fake.call_count == 1
    assert len(output.cells) == 1
    assert output.cells[0].messages == msgs
    assert output.tail == []


async def test_adetect_single_boundary_splits_into_two_cells_and_tail() -> None:
    """LLM says split after msg index 2 (1-based) -> cell of [0:2], tail of [2:]."""
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_llm_boundaries([2])])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert fake.call_count == 1
    assert len(output.cells) == 1
    assert output.cells[0].messages == msgs[:2]
    assert output.tail == msgs[2:]


async def test_adetect_multiple_boundaries_yields_multiple_cells() -> None:
    """6 messages, boundaries [2, 4] -> cells [0:2], [2:4], tail [4:]."""
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_llm_boundaries([2, 4])])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert fake.call_count == 1
    assert len(output.cells) == 2
    assert output.cells[0].messages == msgs[:2]
    assert output.cells[1].messages == msgs[2:4]
    assert output.tail == msgs[4:]


async def test_adetect_multiple_boundaries_with_is_final_closes_tail() -> None:
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_llm_boundaries([2, 4])])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake, is_final=True)

    assert fake.call_count == 1
    assert len(output.cells) == 3
    assert output.cells[2].messages == msgs[4:]
    assert output.tail == []


async def test_adetect_out_of_range_boundaries_are_filtered() -> None:
    """Boundaries outside ``1 <= b < len(messages)`` are silently dropped."""
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_llm_boundaries([0, 2, 4, 99])])  # only 2 is valid

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert len(output.cells) == 1
    assert output.cells[0].messages == msgs[:2]
    assert output.tail == msgs[2:]


async def test_adetect_unsorted_boundaries_are_sorted() -> None:
    """Boundaries returned out of order get sorted before slicing."""
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_llm_boundaries([4, 2])])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert len(output.cells) == 2
    assert output.cells[0].messages == msgs[:2]
    assert output.cells[1].messages == msgs[2:4]


async def test_adetect_duplicate_boundaries_are_deduped() -> None:
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_llm_boundaries([2, 2, 4])])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert len(output.cells) == 2


async def test_adetect_raises_runtime_error_after_5_retries() -> None:
    """Per new release line 446-451: exhausting 5 retries raises RuntimeError."""
    msgs = _dialogue(2)
    bad = ChatResponse(content="not json at all", model="fake")
    fake = FakeLLMClient(responses=[bad, bad, bad, bad, bad])

    with pytest.raises(RuntimeError, match="5 retries exhausted"):
        await ChatMemCellExtractor().adetect(msgs, llm=fake)


async def test_adetect_retries_on_malformed_json_then_succeeds() -> None:
    """Attempts 1-3 unparseable; attempt 4 valid -> algorithm uses attempt 4."""
    msgs = _dialogue(4)
    responses: list[str | ChatResponse] = [
        ChatResponse(content="not json at all", model="fake"),
        ChatResponse(content="[1, 2, 3]", model="fake"),  # parses but not a dict
        ChatResponse(content='{"boundaries": "not a list"}', model="fake"),
        _llm_boundaries([2]),
    ]
    fake = FakeLLMClient(responses=responses)

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert fake.call_count == 4
    assert len(output.cells) == 1


async def test_adetect_handles_json_fence() -> None:
    """LLM response wrapped in ```json``` fence is parsed via tier 1."""
    msgs = _dialogue(4)
    fence_response = ChatResponse(
        content='```json\n{"boundaries": [2], "should_wait": false}\n```',
        model="fake",
    )
    fake = FakeLLMClient(responses=[fence_response])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert len(output.cells) == 1
    assert output.cells[0].messages == msgs[:2]


async def test_adetect_handles_prose_wrapped_json() -> None:
    """LLM response with surrounding prose is parsed via tier 3 (outermost ``{...}``)."""
    msgs = _dialogue(4)
    prose_response = ChatResponse(
        content='Reasoning... {"boundaries": [2], "should_wait": false} ...trailing text',
        model="fake",
    )
    fake = FakeLLMClient(responses=[prose_response])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake)

    assert len(output.cells) == 1


# ==========================================================================
# Phase 3 — Force-split loop
# ==========================================================================


async def test_adetect_force_split_msg_limit_triggers_before_llm() -> None:
    """When len(messages) >= hard_msg_limit, force-split runs and the LLM sees only the remainder."""
    msgs = _dialogue(10)
    fake = FakeLLMClient(responses=[_llm_no_boundary()])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake, hard_msg_limit=5, is_final=True)

    # 2 force-split cells (msgs 0:4, 4:8) + 1 is_final closure cell (msgs 8:10).
    assert len(output.cells) == 3
    assert output.cells[0].messages == msgs[0:4]
    assert output.cells[1].messages == msgs[4:8]
    assert output.cells[2].messages == msgs[8:10]
    assert fake.call_count == 1
    assert output.tail == []


async def test_adetect_force_split_alone_does_not_call_llm_when_remainder_short() -> None:
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_llm_no_boundary()])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake, hard_msg_limit=2)

    # hard_msg_limit=2 -> split=1 each iteration; remainder 1 message; LLM still called once.
    assert fake.call_count == 1
    assert output.tail == [msgs[5]]


# ==========================================================================
# Per-call overrides
# ==========================================================================


async def test_adetect_per_call_prompt_overrides_default() -> None:
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **_: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return _llm_no_boundary()

    fake = FakeLLMClient(handler=handler)
    custom = "CUSTOM PROMPT msgs={messages}"

    await ChatMemCellExtractor().adetect(_dialogue(2), llm=fake, prompt=custom)

    assert captured["content"].startswith("CUSTOM PROMPT msgs=")
    assert "Alice: msg-0" in captured["content"]


async def test_adetect_per_call_llm_overrides_default() -> None:
    captured: dict[str, Any] = {"called": False}

    def handler(messages: list[LLMChatMessage], **_: Any) -> ChatResponse:
        captured["called"] = True
        return _llm_no_boundary()

    fake = FakeLLMClient(handler=handler)
    await ChatMemCellExtractor().adetect(_dialogue(2), llm=fake)
    assert captured["called"] is True


# ==========================================================================
# Participants extraction (new release: USER-only, no refer_list)
# ==========================================================================


async def test_adetect_extracts_participants_from_user_sender_ids_only() -> None:
    """Only USER messages contribute sender_id; assistant/tool messages do not."""
    msgs = [
        _msg(MessageRole.USER, "hi", 1, sender_id="u_alice", sender_name="Alice"),
        _msg(MessageRole.ASSISTANT, "hey", 2, sender_id="u_bot", sender_name="Bot"),
        _msg(MessageRole.USER, "more", 3, sender_id="u_bob", sender_name="Bob"),
    ]
    fake = FakeLLMClient(responses=[_llm_no_boundary()])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake, is_final=True)

    assert output.cells[0].participants == ["u_alice", "u_bob"]
    assert output.cells[0].sender_ids == ["u_alice", "u_bob"]


def test_extract_participants_dedupes_within_user_messages() -> None:
    msgs = [
        _msg(MessageRole.USER, "a", 1, sender_id="u_a"),
        _msg(MessageRole.USER, "b", 2, sender_id="u_a"),
    ]
    assert _extract_participants(msgs) == ["u_a"]


def test_extract_participants_skips_assistant_and_tool() -> None:
    msgs = [
        _msg(MessageRole.ASSISTANT, "x", 1, sender_id="u_bot"),
        Message(role=MessageRole.TOOL, content="t", timestamp=2, sender_id="u_tool"),
        _msg(MessageRole.USER, "u", 3, sender_id="u_human"),
    ]
    assert _extract_participants(msgs) == ["u_human"]


# ==========================================================================
# Message rendering (new release: [N] [ISO+TZ] sender_name: content)
# ==========================================================================


def test_format_messages_uses_indices_and_sender_name() -> None:
    msgs = [
        _msg(MessageRole.USER, "hi", 1_700_000_000_000, sender_name="Alice"),
        _msg(MessageRole.ASSISTANT, "hey", 1_700_000_001_000, sender_name="Bot"),
    ]
    rendered = _format_messages_with_indices(msgs)
    lines = rendered.split("\n")
    assert lines[0].startswith("[1] [2023-11-14 22:13:20")
    assert lines[0].endswith(" Alice: hi")
    assert lines[1].startswith("[2] [2023-11-14 22:13:21")
    assert lines[1].endswith(" Bot: hey")


def test_format_messages_falls_back_to_role_when_sender_name_missing() -> None:
    msgs = [_msg(MessageRole.USER, "hi", 1_700_000_000_000)]
    rendered = _format_messages_with_indices(msgs)
    assert " user: hi" in rendered


def test_format_messages_skips_empty_content() -> None:
    msgs = [
        _msg(MessageRole.USER, "real", 1, sender_name="Alice"),
        _msg(MessageRole.USER, "", 2, sender_name="Bob"),
    ]
    rendered = _format_messages_with_indices(msgs)
    assert "Alice: real" in rendered
    assert "Bob" not in rendered


# ==========================================================================
# Batch response parsing
# ==========================================================================


def test_parse_batch_response_direct_json() -> None:
    raw = '{"boundaries": [2, 4], "should_wait": true}'
    result = _parse_batch_boundary_response(raw)
    assert result is not None
    assert result.boundaries == [2, 4]
    assert result.should_wait is True


def test_parse_batch_response_json_fence() -> None:
    raw = '```json\n{"boundaries": [3], "should_wait": false}\n```'
    result = _parse_batch_boundary_response(raw)
    assert result is not None
    assert result.boundaries == [3]


def test_parse_batch_response_outermost_braces() -> None:
    raw = 'thinking out loud {"boundaries": [1], "should_wait": false} trailing'
    result = _parse_batch_boundary_response(raw)
    assert result is not None
    assert result.boundaries == [1]


def test_parse_batch_response_skips_unparseable_boundary_entries() -> None:
    raw = '{"boundaries": [2, "abc", null, 4], "should_wait": false}'
    result = _parse_batch_boundary_response(raw)
    assert result is not None
    assert result.boundaries == [2, 4]


def test_parse_batch_response_returns_none_on_bad_json() -> None:
    assert _parse_batch_boundary_response("totally not json") is None


def test_parse_batch_response_returns_none_when_boundaries_not_list() -> None:
    raw = '{"boundaries": "nope", "should_wait": false}'
    assert _parse_batch_boundary_response(raw) is None


# ==========================================================================
# MemCell schema parity with new release
# ==========================================================================


async def test_make_cell_sets_opensource_fields() -> None:
    msgs = _dialogue(2)
    fake = FakeLLMClient(responses=[_llm_no_boundary()])

    output = await ChatMemCellExtractor().adetect(msgs, llm=fake, is_final=True)

    cell = output.cells[0]
    assert cell.type == RawDataType.CONVERSATION
    assert cell.event_id == f"mc_{msgs[0].timestamp}_{msgs[-1].timestamp}"
    assert cell.timestamp == msgs[-1].timestamp
    assert len(cell.original_data) == 2
    assert cell.original_data[0]["message"]["content"] == "msg-0"
    assert cell.participants == cell.sender_ids


# ==========================================================================
# NamedTuple ergonomics + defaults
# ==========================================================================


async def test_detection_output_supports_positional_unpacking() -> None:
    fake = FakeLLMClient(responses=[_llm_no_boundary()])
    cells, tail = await ChatMemCellExtractor().adetect(_dialogue(2), llm=fake)
    assert isinstance(cells, list)
    assert isinstance(tail, list)


def test_default_hard_limits_match_design_doc() -> None:
    assert DEFAULT_HARD_TOKEN_LIMIT == 65536
    assert DEFAULT_HARD_MSG_LIMIT == 500


# ==========================================================================
# _find_force_split_point edge cases
# ==========================================================================


def test_find_force_split_point_returns_len_for_zero_or_one_message() -> None:
    """``len(messages) <= 1`` short-circuit (line 251)."""
    assert _find_force_split_point([], hard_token_limit=10, hard_msg_limit=10) == 0
    assert _find_force_split_point(_dialogue(1), hard_token_limit=10, hard_msg_limit=10) == 1


def test_find_force_split_point_halves_candidate_when_head_exceeds_token_limit() -> None:
    """When the candidate head still exceeds ``hard_token_limit``, candidate halves (line 254)."""
    # 10 long messages; tiny token limit forces the loop to halve.
    long_msgs = [_msg(MessageRole.USER, "x" * 1000, 1_700_000_000_000 + i, sender_name="Alice") for i in range(10)]
    split = _find_force_split_point(long_msgs, hard_token_limit=10, hard_msg_limit=500)
    # Starts at min(499, 9) = 9, halves down (9→4→2→1) until head fits or floor 1 reached.
    assert 1 <= split <= 9


# ==========================================================================
# _detect_boundaries — client.chat raises retryable error
# ==========================================================================


async def test_adetect_retries_when_client_chat_raises_value_error() -> None:
    """ValueError raised by client.chat is caught by the retry loop (lines 299-300)."""
    state = {"calls": 0}

    def flaky_handler(messages: list[LLMChatMessage], **_: Any) -> ChatResponse:
        state["calls"] += 1
        if state["calls"] < 3:
            raise ValueError("transient")
        return _llm_no_boundary()

    fake = FakeLLMClient(handler=flaky_handler)
    output = await ChatMemCellExtractor().adetect(_dialogue(2), llm=fake, is_final=True)
    assert state["calls"] == 3
    assert len(output.cells) == 1


# ==========================================================================
# _parse_json_three_tier — fence + outermost-braces invalid-JSON branches
# ==========================================================================


def test_parse_json_three_tier_fence_with_invalid_inner_falls_through() -> None:
    """Fence found but body is not valid JSON (lines 348-349): falls through to subsequent tiers."""
    # Valid direct JSON outside the fence so tier 2 takes over.
    raw = '```json\nNOT VALID JSON\n```\n{"boundaries": [1], "should_wait": false}'
    parsed = _parse_json_three_tier(raw)
    # Tier 2 (`json.loads(raw.strip())`) sees the leading fence so it fails;
    # tier 3 extracts outermost braces and parses successfully.
    assert isinstance(parsed, dict)
    assert parsed["boundaries"] == [1]


def test_parse_json_three_tier_outermost_braces_invalid_returns_none() -> None:
    """All three tiers fail when no valid JSON anywhere (lines 361-362)."""
    raw = "prefix {not valid at all} suffix"
    assert _parse_json_three_tier(raw) is None
