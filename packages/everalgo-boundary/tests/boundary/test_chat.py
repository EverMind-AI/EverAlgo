"""Tests for everalgo.boundary.chat — detect_boundaries (single-call LLM, no retry).

Coverage layers:
- Phase 1  — empty input short-circuits; missing llm raises ValueError
- Phase 3  — force-split loop (hard_msg_limit / hard_token_limit)
- Phase 4  — LLM batch detection: happy path, boundary filtering, is_final closure
- Error path — invalid LLM JSON raises ValueError; non-list boundaries raises TypeError
- Helpers   — _format_messages_with_indices, _find_force_split_point, _parse_batch_boundary_response
- DetectionResult — NamedTuple unpacking, named-field access
"""

from __future__ import annotations

from typing import Any

import pytest

from everalgo.boundary.chat import (
    DEFAULT_HARD_MSG_LIMIT,
    DEFAULT_HARD_TOKEN_LIMIT,
    DetectionResult,
    _find_force_split_point,
    _format_messages_with_indices,
    detect_boundaries,
)
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_BASE_TS_MS = 1_700_000_000_000  # 2023-11-14T22:13:20Z


def _msg(
    role: str,
    content: str,
    ts: int,
    *,
    sender_id: str = "u_alice",
    sender_name: str | None = None,
) -> ChatMessage:
    """Build a minimal ChatMessage with required fields."""
    return ChatMessage(
        id=f"msg_{ts}",
        role=role,  # type: ignore[arg-type]
        content=content,
        timestamp=ts,
        sender_id=sender_id,
        sender_name=sender_name,
    )


def _dialogue(n: int, *, base_ts: int = _BASE_TS_MS, gap_ms: int = 1000) -> list[ChatMessage]:
    """Alternating user/assistant messages, each with sender identity set."""
    return [
        _msg(
            "user" if i % 2 == 0 else "assistant",
            f"msg-{i}",
            base_ts + i * gap_ms,
            sender_id="u_alice" if i % 2 == 0 else "u_bot",
            sender_name="Alice" if i % 2 == 0 else "Bot",
        )
        for i in range(n)
    ]


def _resp_no_boundary(*, should_wait: bool = False) -> ChatResponse:
    """LLM response: full input belongs to one episode (no boundary)."""
    flag = str(should_wait).lower()
    return ChatResponse(
        content=f'{{"reasoning": "single episode", "boundaries": [], "should_wait": {flag}}}',
        model="fake",
    )


def _resp_boundaries(indices: list[int], *, should_wait: bool = False) -> ChatResponse:
    """LLM response: split at each index in ``indices``."""
    flag = str(should_wait).lower()
    return ChatResponse(
        content=f'{{"reasoning": "split", "boundaries": {indices}, "should_wait": {flag}}}',
        model="fake",
    )


# ===========================================================================
# Phase 1 — Input validation
# ===========================================================================


async def test_empty_messages_short_circuits_without_calling_llm() -> None:
    fake = FakeLLMClient(responses=[])
    result = await detect_boundaries([], llm=fake)
    assert result == DetectionResult(cells=[], tail=[])
    assert fake.call_count == 0


# NOTE: removed test_missing_llm_raises_value_error — `llm` is now a required
# keyword-only argument; static type checking catches missing-arg misuse.


# ===========================================================================
# Phase 4 — LLM batch detection: happy path
# ===========================================================================


async def test_no_boundary_is_final_false_returns_tail() -> None:
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_resp_no_boundary()])

    result = await detect_boundaries(msgs, llm=fake)

    assert fake.call_count == 1
    assert result.cells == []
    assert result.tail == msgs


async def test_no_boundary_is_final_true_closes_into_one_cell() -> None:
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_resp_no_boundary()])

    result = await detect_boundaries(msgs, llm=fake, is_final=True)

    assert fake.call_count == 1
    assert len(result.cells) == 1
    assert result.cells[0].items == msgs
    assert result.tail == []


async def test_single_message_no_boundary_returns_as_tail() -> None:
    msgs = _dialogue(1)
    fake = FakeLLMClient(responses=[_resp_no_boundary()])

    result = await detect_boundaries(msgs, llm=fake)

    assert fake.call_count == 1
    assert result.cells == []
    assert result.tail == msgs


async def test_single_message_is_final_closes_into_one_cell() -> None:
    msgs = _dialogue(1)
    fake = FakeLLMClient(responses=[_resp_no_boundary()])

    result = await detect_boundaries(msgs, llm=fake, is_final=True)

    assert fake.call_count == 1
    assert len(result.cells) == 1
    assert result.cells[0].items == msgs
    assert result.tail == []


async def test_single_boundary_splits_into_one_closed_cell_and_tail() -> None:
    """LLM returns boundary at index 2 — messages[:2] are closed, messages[2:] become tail."""
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_resp_boundaries([2])])

    result = await detect_boundaries(msgs, llm=fake)

    assert fake.call_count == 1
    assert len(result.cells) == 1
    assert result.cells[0].items == msgs[:2]
    assert result.tail == msgs[2:]


async def test_multiple_boundaries_yield_multiple_closed_cells() -> None:
    """boundaries=[2, 4] on 6 messages → cells [0:2], [2:4]; tail [4:]."""
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_resp_boundaries([2, 4])])

    result = await detect_boundaries(msgs, llm=fake)

    assert fake.call_count == 1
    assert len(result.cells) == 2
    assert result.cells[0].items == msgs[:2]
    assert result.cells[1].items == msgs[2:4]
    assert result.tail == msgs[4:]


async def test_multiple_boundaries_with_is_final_closes_tail_as_final_cell() -> None:
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_resp_boundaries([2, 4])])

    result = await detect_boundaries(msgs, llm=fake, is_final=True)

    assert len(result.cells) == 3
    assert result.cells[2].items == msgs[4:]
    assert result.tail == []


# ===========================================================================
# Phase 4 — Boundary index filtering
# ===========================================================================


async def test_out_of_range_boundary_indices_are_filtered() -> None:
    """Indices outside ``1 <= b < len(messages)`` are silently dropped; only valid index survives."""
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_resp_boundaries([0, 2, 4, 99])])  # only 2 is valid (1<=2<4)

    result = await detect_boundaries(msgs, llm=fake)

    assert len(result.cells) == 1
    assert result.cells[0].items == msgs[:2]
    assert result.tail == msgs[2:]


async def test_unsorted_boundary_indices_are_sorted_before_slicing() -> None:
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_resp_boundaries([4, 2])])

    result = await detect_boundaries(msgs, llm=fake)

    assert len(result.cells) == 2
    assert result.cells[0].items == msgs[:2]
    assert result.cells[1].items == msgs[2:4]


async def test_duplicate_boundary_indices_are_deduplicated() -> None:
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_resp_boundaries([2, 2, 4])])

    result = await detect_boundaries(msgs, llm=fake)

    assert len(result.cells) == 2


async def test_empty_boundaries_list_yields_no_closed_cells() -> None:
    msgs = _dialogue(3)
    fake = FakeLLMClient(responses=[_resp_boundaries([])])

    result = await detect_boundaries(msgs, llm=fake)

    assert result.cells == []
    assert result.tail == msgs


# ===========================================================================
# Phase 4 — Error path (single call, no retry)
# ===========================================================================


async def test_invalid_json_response_raises_value_error() -> None:
    """Non-JSON response → ValueError immediately (fail-loud, no retry)."""
    msgs = _dialogue(2)
    bad = ChatResponse(content="not json at all", model="fake")
    fake = FakeLLMClient(responses=[bad])

    with pytest.raises(ValueError, match="No JSON object found"):
        await detect_boundaries(msgs, llm=fake)

    assert fake.call_count == 1


async def test_boundaries_not_list_in_json_raises_value_error() -> None:
    """``boundaries`` field present but not a list → ValueError immediately (fail-loud, no retry)."""
    msgs = _dialogue(2)
    bad = ChatResponse(content='{"boundaries": "not-a-list", "should_wait": false}', model="fake")
    fake = FakeLLMClient(responses=[bad])

    with pytest.raises(ValueError, match="boundaries must be a list"):
        await detect_boundaries(msgs, llm=fake)

    assert fake.call_count == 1


# ===========================================================================
# Phase 3 — Force-split loop
# ===========================================================================


async def test_force_split_on_msg_limit_splits_before_llm_call() -> None:
    """hard_msg_limit=5 on 10 messages: 2 force-split cells produced before LLM sees remainder."""
    msgs = _dialogue(10)
    fake = FakeLLMClient(responses=[_resp_no_boundary()])

    result = await detect_boundaries(msgs, llm=fake, hard_msg_limit=5, is_final=True)

    # Force-split loop splits [0:4] and [4:8] (hard_msg_limit-1 = 4 each); LLM sees [8:10].
    # is_final=True closes the LLM's remainder → 3 total cells.
    assert len(result.cells) == 3
    assert result.cells[0].items == msgs[0:4]
    assert result.cells[1].items == msgs[4:8]
    assert result.cells[2].items == msgs[8:10]
    assert fake.call_count == 1
    assert result.tail == []


async def test_no_force_split_when_within_limits() -> None:
    """Message count and token count both under defaults — force-split loop never fires."""
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_resp_no_boundary()])

    result = await detect_boundaries(msgs, llm=fake)

    # A single LLM call covers all 4 messages — no force-split cells prepended.
    assert fake.call_count == 1
    assert result.cells == []


# ===========================================================================
# Per-call overrides
# ===========================================================================


async def test_per_call_prompt_override_is_rendered_with_messages() -> None:
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **_: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return _resp_no_boundary()

    fake = FakeLLMClient(handler=handler)
    custom = "CUSTOM PROMPT msgs={messages}"

    await detect_boundaries(_dialogue(2), llm=fake, prompt=custom)

    assert captured["content"].startswith("CUSTOM PROMPT msgs=")
    assert "Alice: msg-0" in captured["content"]


async def test_default_prompt_template_is_used_when_no_override() -> None:
    """Without a ``prompt=`` override, the bundled template is rendered into the LLM call."""
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **_: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return _resp_no_boundary()

    fake = FakeLLMClient(handler=handler)
    await detect_boundaries(_dialogue(2), llm=fake)

    # The bundled prompt always contains the message block — sender name "Alice" appears.
    assert "Alice" in captured["content"]


# ===========================================================================
# MemCell shape
# ===========================================================================


async def test_make_cell_timestamp_is_last_message_timestamp() -> None:
    msgs = _dialogue(3)
    fake = FakeLLMClient(responses=[_resp_boundaries([2])])

    result = await detect_boundaries(msgs, llm=fake)

    cell = result.cells[0]
    assert isinstance(cell, MemCell)
    assert cell.timestamp == msgs[1].timestamp  # closing message of the slice [0:2] is index 1
    assert cell.items == msgs[:2]


async def test_tail_cell_timestamp_when_is_final_matches_last_message() -> None:
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_resp_boundaries([2])])

    result = await detect_boundaries(msgs, llm=fake, is_final=True)

    last_cell = result.cells[-1]
    assert last_cell.timestamp == msgs[-1].timestamp
    assert last_cell.items == msgs[2:]


# ===========================================================================
# DetectionResult NamedTuple ergonomics
# ===========================================================================


async def test_detection_result_supports_positional_unpacking() -> None:
    fake = FakeLLMClient(responses=[_resp_no_boundary()])
    cells, tail = await detect_boundaries(_dialogue(2), llm=fake)
    assert isinstance(cells, list)
    assert isinstance(tail, list)


async def test_detection_result_named_field_access() -> None:
    fake = FakeLLMClient(responses=[_resp_no_boundary()])
    result = await detect_boundaries(_dialogue(2), llm=fake)
    assert result.cells == []
    assert isinstance(result.tail, list)


def test_detection_result_index_access() -> None:
    r = DetectionResult(cells=[], tail=[])
    assert r[0] == []
    assert r[1] == []


def test_default_hard_limits_match_design_spec() -> None:
    assert DEFAULT_HARD_TOKEN_LIMIT == 8192
    assert DEFAULT_HARD_MSG_LIMIT == 50


# ===========================================================================
# Helper: _format_messages_with_indices
# ===========================================================================


def test_format_messages_renders_index_timestamp_and_sender_name() -> None:
    msgs = [
        _msg("user", "hi", _BASE_TS_MS, sender_name="Alice"),
        _msg("assistant", "hey", _BASE_TS_MS + 1000, sender_id="u_bot", sender_name="Bot"),
    ]
    rendered = _format_messages_with_indices(msgs)
    lines = rendered.split("\n")
    assert lines[0].startswith("[1] [2023-11-14 22:13:20")
    assert lines[0].endswith(" Alice: hi")
    assert lines[1].startswith("[2] [2023-11-14 22:13:21")
    assert lines[1].endswith(" Bot: hey")


def test_format_messages_falls_back_to_sender_id_when_sender_name_absent() -> None:
    """When sender_name is missing, render sender_id (preserves multi-user group distinguishability).

    Falling back to ``role`` would collapse multiple distinct users into a single ``user:`` label
    and break boundary detection for groupchat — see boundary docstring for rationale.
    """
    msgs = [_msg("user", "hi", _BASE_TS_MS, sender_id="u_alice")]  # no sender_name
    rendered = _format_messages_with_indices(msgs)
    assert " u_alice: hi" in rendered


def test_format_messages_skips_empty_content_entries() -> None:
    msgs = [
        _msg("user", "real content", _BASE_TS_MS, sender_name="Alice"),
        _msg("user", "", _BASE_TS_MS + 1000, sender_id="u_bob", sender_name="Bob"),
    ]
    rendered = _format_messages_with_indices(msgs)
    assert "Alice: real content" in rendered
    assert "Bob" not in rendered


def test_format_messages_empty_list_returns_empty_string() -> None:
    assert _format_messages_with_indices([]) == ""


def test_format_messages_indices_are_one_based() -> None:
    msgs = _dialogue(3)
    rendered = _format_messages_with_indices(msgs)
    assert "[1] " in rendered
    assert "[2] " in rendered
    assert "[3] " in rendered
    assert "[0] " not in rendered


# ===========================================================================
# Helper: _find_force_split_point
# ===========================================================================


def test_find_force_split_point_returns_zero_for_empty_list() -> None:
    assert _find_force_split_point([], hard_token_limit=10, hard_msg_limit=10) == 0


def test_find_force_split_point_returns_one_for_single_message() -> None:
    assert _find_force_split_point(_dialogue(1), hard_token_limit=10, hard_msg_limit=10) == 1


def test_find_force_split_point_candidate_below_msg_limit() -> None:
    """Candidate is capped at hard_msg_limit - 1."""
    msgs = _dialogue(10)
    split = _find_force_split_point(msgs, hard_token_limit=DEFAULT_HARD_TOKEN_LIMIT, hard_msg_limit=5)
    assert split <= 4  # hard_msg_limit - 1


def test_find_force_split_point_halves_when_head_exceeds_token_limit() -> None:
    """Tiny token limit forces the halving loop to reduce the candidate."""
    long_msgs = [_msg("user", "x" * 500, _BASE_TS_MS + i, sender_name="Alice") for i in range(10)]
    split = _find_force_split_point(long_msgs, hard_token_limit=1, hard_msg_limit=500)
    # Halving loop bottoms out at 1 (floor guard).
    assert 1 <= split <= 9


# NOTE: The three-tier JSON parsing logic that used to live in _parse_json_three_tier
# has been extracted to everalgo.llm.parse.parse_llm_json_object and is unit-tested
# in packages/everalgo-core/tests/llm/test_parse.py.
