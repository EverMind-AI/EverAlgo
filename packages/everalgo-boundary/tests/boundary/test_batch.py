"""Tests for the batch path — adetect_boundaries over chat-only input (single-call LLM, no retry).

Mixed chat + tool input is covered in ``test_mixed_items.py``; the hard-limit defaults in ``test_hard_limits.py``.

Coverage layers:
- Phase 1  — empty input short-circuits; missing llm raises ValueError
- Phase 3  — force-split loop (hard_msg_limit / hard_token_limit)
- Phase 4  — LLM batch detection: happy path, boundary filtering, is_final closure
- Error path — a reply that fails ``_BatchReply`` raises LLMOutputError; client exceptions propagate unchanged
- Contract  — the call requests ``_BatchReply``; the reply models obey docs/concepts/llm-contract.md §2.2
- Helpers   — _format_messages_with_indices, _find_force_split_point
- DetectionResult — NamedTuple unpacking, named-field access
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from pydantic import ValidationError

from everalgo.boundary import DetectionResult, adetect_boundaries
from everalgo.boundary._render import _format_messages_with_indices
from everalgo.boundary._replies import _BatchReply, _StepReply
from everalgo.boundary._segments import _find_force_split_point
from everalgo.boundary.prompts.en.chat import BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT
from everalgo.llm import LLMMessage, LLMOutputError, StrictOutputModel, to_openai_response_format
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.utils import InvalidInputError

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


def _resp_no_boundary(*, tail_complete: bool = True) -> str:
    """LLM reply: full input belongs to one episode (no boundary)."""
    return json.dumps({"reasoning": "single episode", "boundaries": [], "tail_complete": tail_complete})


def _resp_boundaries(indices: list[int], *, tail_complete: bool = True) -> str:
    """LLM reply: split at each index in ``indices``."""
    return json.dumps({"reasoning": "split", "boundaries": indices, "tail_complete": tail_complete})


# ===========================================================================
# Phase 1 — Input validation
# ===========================================================================


async def test_empty_messages_short_circuits_without_calling_llm() -> None:
    fake = FakeLLMClient(responses=[])
    result = await adetect_boundaries([], llm=fake)
    assert result == DetectionResult(cells=[], tail=[], tail_complete=None)
    assert fake.call_count == 0


# NOTE: removed test_missing_llm_raises_value_error — `llm` is now a required
# keyword-only argument; static type checking catches missing-arg misuse.


# ===========================================================================
# Phase 4 — LLM batch detection: happy path
# ===========================================================================


async def test_no_boundary_is_final_false_returns_tail() -> None:
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_resp_no_boundary()])

    result = await adetect_boundaries(msgs, llm=fake)

    assert fake.call_count == 1
    assert result.cells == []
    assert result.tail == msgs


async def test_no_boundary_is_final_true_closes_into_one_cell() -> None:
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_resp_no_boundary()])

    result = await adetect_boundaries(msgs, llm=fake, is_final=True)

    assert fake.call_count == 1
    assert len(result.cells) == 1
    assert result.cells[0].items == msgs
    assert result.tail == []


async def test_single_message_no_boundary_returns_as_tail() -> None:
    msgs = _dialogue(1)
    fake = FakeLLMClient(responses=[_resp_no_boundary()])

    result = await adetect_boundaries(msgs, llm=fake)

    assert fake.call_count == 1
    assert result.cells == []
    assert result.tail == msgs


async def test_single_message_is_final_closes_into_one_cell() -> None:
    msgs = _dialogue(1)
    fake = FakeLLMClient(responses=[_resp_no_boundary()])

    result = await adetect_boundaries(msgs, llm=fake, is_final=True)

    assert fake.call_count == 1
    assert len(result.cells) == 1
    assert result.cells[0].items == msgs
    assert result.tail == []


async def test_single_boundary_splits_into_one_closed_cell_and_tail() -> None:
    """LLM returns boundary at index 2 — messages[:2] are closed, messages[2:] become tail."""
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_resp_boundaries([2])])

    result = await adetect_boundaries(msgs, llm=fake)

    assert fake.call_count == 1
    assert len(result.cells) == 1
    assert result.cells[0].items == msgs[:2]
    assert result.tail == msgs[2:]


async def test_multiple_boundaries_yield_multiple_closed_cells() -> None:
    """boundaries=[2, 4] on 6 messages → cells [0:2], [2:4]; tail [4:]."""
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_resp_boundaries([2, 4])])

    result = await adetect_boundaries(msgs, llm=fake)

    assert fake.call_count == 1
    assert len(result.cells) == 2
    assert result.cells[0].items == msgs[:2]
    assert result.cells[1].items == msgs[2:4]
    assert result.tail == msgs[4:]


async def test_multiple_boundaries_with_is_final_closes_tail_as_final_cell() -> None:
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_resp_boundaries([2, 4])])

    result = await adetect_boundaries(msgs, llm=fake, is_final=True)

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

    result = await adetect_boundaries(msgs, llm=fake)

    assert len(result.cells) == 1
    assert result.cells[0].items == msgs[:2]
    assert result.tail == msgs[2:]


async def test_unsorted_boundary_indices_are_sorted_before_slicing() -> None:
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_resp_boundaries([4, 2])])

    result = await adetect_boundaries(msgs, llm=fake)

    assert len(result.cells) == 2
    assert result.cells[0].items == msgs[:2]
    assert result.cells[1].items == msgs[2:4]


async def test_duplicate_boundary_indices_are_deduplicated() -> None:
    msgs = _dialogue(6)
    fake = FakeLLMClient(responses=[_resp_boundaries([2, 2, 4])])

    result = await adetect_boundaries(msgs, llm=fake)

    assert len(result.cells) == 2


async def test_empty_boundaries_list_yields_no_closed_cells() -> None:
    msgs = _dialogue(3)
    fake = FakeLLMClient(responses=[_resp_boundaries([])])

    result = await adetect_boundaries(msgs, llm=fake)

    assert result.cells == []
    assert result.tail == msgs


# ===========================================================================
# Phase 4 — Error path (single call, no retry)
# ===========================================================================


_SECRET = "BOUNDARY-SECRET-5c1e"


class _ClientError(Exception):
    """A host's own call-failure type, which the operator must neither wrap nor retype."""


@pytest.mark.parametrize(
    ("reply", "failing_location"),
    [
        pytest.param(f"not json {_SECRET}", "<root>", id="not-json"),
        pytest.param(
            json.dumps({"reasoning": _SECRET, "boundaries": "not-a-list", "tail_complete": True}),
            "boundaries:",
            id="boundaries-not-a-list",
        ),
        pytest.param(
            json.dumps({"reasoning": _SECRET, "boundaries": [], "tail_complete": True, "note": "x"}),
            "note:",
            id="extra-field-forbidden",
        ),
        pytest.param(
            json.dumps({"reasoning": _SECRET, "boundaries": []}), "tail_complete:", id="tail-complete-missing"
        ),
        pytest.param(
            json.dumps({"reasoning": _SECRET, "boundaries": ["2"], "tail_complete": True}),
            "boundaries.0:",
            id="string-index-not-coerced",
        ),
        pytest.param(
            json.dumps({"reasoning": _SECRET, "boundaries": [None, 2], "tail_complete": True}),
            "boundaries.0:",
            id="null-index-not-skipped",
        ),
    ],
)
async def test_reply_failing_batch_model_raises_llm_output_error(reply: str, failing_location: str) -> None:
    """A reply that fails ``_BatchReply`` is an LLMOutputError at once — no retry, no reply text in the message."""
    fake = FakeLLMClient(responses=[reply])

    with pytest.raises(LLMOutputError) as excinfo:
        await adetect_boundaries(_dialogue(2), llm=fake)

    assert excinfo.type is LLMOutputError
    assert isinstance(excinfo.value.__cause__, ValidationError)
    assert "boundary batch reply is not a valid _BatchReply" in str(excinfo.value)
    assert failing_location in str(excinfo.value)
    assert _SECRET not in str(excinfo.value)
    assert fake.call_count == 1


async def test_client_exception_propagates_unchanged() -> None:
    """The host recognises call failures by its own types, so the operator neither wraps nor retypes them."""
    error = _ClientError("upstream down")

    with pytest.raises(_ClientError) as excinfo:
        await adetect_boundaries(_dialogue(2), llm=FakeLLMClient(responses=[error]))

    assert excinfo.value is error


# ===========================================================================
# LLM contract — the operator-side obligations of docs/concepts/llm-contract.md §4.3
# ===========================================================================


_REPLY_MODELS: tuple[type[StrictOutputModel], ...] = (_BatchReply, _StepReply)


@pytest.mark.parametrize("model", _REPLY_MODELS, ids=lambda model: model.__name__)
def test_reply_models_are_strict_one_level_and_default_free(model: type[StrictOutputModel]) -> None:
    """Both reply models inherit StrictOutputModel directly, require every field and carry no validators (§2.2)."""
    assert model.__bases__ == (StrictOutputModel,)
    assert model.model_config.get("strict") is True
    assert model.model_config.get("extra") == "forbid"
    for name, field in model.model_fields.items():
        assert field.is_required(), f"{model.__name__}.{name} has a default"
    assert not model.__pydantic_decorators__.field_validators
    assert not model.__pydantic_decorators__.model_validators


@pytest.mark.parametrize("model", _REPLY_MODELS, ids=lambda model: model.__name__)
def test_reply_models_fit_the_strict_subset(model: type[StrictOutputModel]) -> None:
    """The request builders raise on anything outside the strict subset or a non-object top level (§8.3)."""
    response_format = to_openai_response_format(model)

    assert response_format is not None
    assert response_format["json_schema"]["schema"]["type"] == "object"


async def test_batch_call_requests_the_batch_reply_model() -> None:
    fake = FakeLLMClient(responses=[_resp_no_boundary()])

    await adetect_boundaries(_dialogue(2), llm=fake)

    assert [call.response_format for call in fake.calls] == [_BatchReply]


async def test_batch_reply_is_logged_without_content(caplog: pytest.LogCaptureFixture) -> None:
    """One INFO line per call counts what the reply held and what the range check dropped; the reasoning stays out."""
    reply = json.dumps({"reasoning": _SECRET, "boundaries": [0, 2, 2, 9], "tail_complete": False})

    with caplog.at_level(logging.INFO, logger="everalgo.boundary"):
        await adetect_boundaries(_dialogue(4), llm=FakeLLMClient(responses=[reply]))

    assert caplog.messages == [
        "boundary batch reply: messages=4 boundaries=1 tail_complete=False dropped_out_of_range=2"
    ]
    assert _SECRET not in caplog.text


# ===========================================================================
# Phase 3 — Force-split loop
# ===========================================================================


async def test_force_split_on_msg_limit_splits_before_llm_call() -> None:
    """hard_msg_limit=5 on 10 messages: 2 force-split cells produced before LLM sees remainder."""
    msgs = _dialogue(10)
    fake = FakeLLMClient(responses=[_resp_no_boundary()])

    result = await adetect_boundaries(msgs, llm=fake, hard_msg_limit=5, is_final=True)

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

    result = await adetect_boundaries(msgs, llm=fake)

    # A single LLM call covers all 4 messages — no force-split cells prepended.
    assert fake.call_count == 1
    assert result.cells == []


# ===========================================================================
# Per-call overrides
# ===========================================================================


async def test_per_call_custom_instructions_close_the_bundled_system_message() -> None:
    captured: list[LLMMessage] = []

    def handler(messages: list[LLMMessage], **_: Any) -> str:
        captured.extend(messages)
        return _resp_no_boundary()

    fake = FakeLLMClient(handler=handler)

    await adetect_boundaries(_dialogue(2), llm=fake, custom_instructions="CUSTOM RULES")

    assert [m.role for m in captured] == ["system", "user"]
    assert captured[0].content == (
        f"{BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT}\n\n<custom_instructions>\nCUSTOM RULES\n</custom_instructions>"
    )
    assert "Alice: msg-0" in str(captured[1].content)


async def test_default_system_prompt_is_used_without_custom_instructions() -> None:
    """Without ``custom_instructions``, the bundled rules are the system message and the data the user message."""
    captured: list[LLMMessage] = []

    def handler(messages: list[LLMMessage], **_: Any) -> str:
        captured.extend(messages)
        return _resp_no_boundary()

    fake = FakeLLMClient(handler=handler)
    await adetect_boundaries(_dialogue(2), llm=fake)

    assert [m.role for m in captured] == ["system", "user"]
    assert captured[0].content == BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT
    assert "Alice: msg-0" in str(captured[1].content)


# ===========================================================================
# MemCell shape
# ===========================================================================


async def test_make_cell_timestamp_is_last_message_timestamp() -> None:
    msgs = _dialogue(3)
    fake = FakeLLMClient(responses=[_resp_boundaries([2])])

    result = await adetect_boundaries(msgs, llm=fake)

    cell = result.cells[0]
    assert isinstance(cell, MemCell)
    assert cell.timestamp == msgs[1].timestamp  # closing message of the slice [0:2] is index 1
    assert cell.items == msgs[:2]


async def test_tail_cell_timestamp_when_is_final_matches_last_message() -> None:
    msgs = _dialogue(4)
    fake = FakeLLMClient(responses=[_resp_boundaries([2])])

    result = await adetect_boundaries(msgs, llm=fake, is_final=True)

    last_cell = result.cells[-1]
    assert last_cell.timestamp == msgs[-1].timestamp
    assert last_cell.items == msgs[2:]


# ===========================================================================
# DetectionResult NamedTuple ergonomics
# ===========================================================================


async def test_detection_result_supports_positional_unpacking() -> None:
    """Three fields now, so two-value unpacking no longer works — 0.3.0 states this as breaking."""
    fake = FakeLLMClient(responses=[_resp_no_boundary()])
    cells, tail, tail_complete = await adetect_boundaries(_dialogue(2), llm=fake)
    assert isinstance(cells, list)
    assert isinstance(tail, list)
    assert isinstance(tail_complete, bool)


async def test_detection_result_named_field_access() -> None:
    fake = FakeLLMClient(responses=[_resp_no_boundary()])
    result = await adetect_boundaries(_dialogue(2), llm=fake)
    assert result.cells == []
    assert isinstance(result.tail, list)


def test_detection_result_index_access() -> None:
    r = DetectionResult(cells=[], tail=[], tail_complete=None)
    assert r[0] == []
    assert r[1] == []
    assert r[2] is None


# ===========================================================================
# tail_complete — the LLM's verdict on the tail, previously parsed and then dropped
# ===========================================================================


async def test_tail_complete_false_is_surfaced_to_the_caller() -> None:
    """It was validated as a mandatory key and then discarded; nothing upstream could see it."""
    fake = FakeLLMClient(responses=[_resp_no_boundary(tail_complete=False)])

    result = await adetect_boundaries(_dialogue(2), llm=fake)

    assert result.tail_complete is False


async def test_tail_complete_true_is_surfaced_to_the_caller() -> None:
    fake = FakeLLMClient(responses=[_resp_no_boundary(tail_complete=True)])

    result = await adetect_boundaries(_dialogue(2), llm=fake)

    assert result.tail_complete is True


async def test_tail_complete_is_independent_of_whether_the_tail_is_empty() -> None:
    """A non-empty tail is the normal case; tail_complete is a narrower claim about that tail.

    If the two moved together the field would carry no information a caller could not already read off
    ``tail``, which is exactly the confusion that made it look droppable.
    """
    incomplete = await adetect_boundaries(
        _dialogue(2), llm=FakeLLMClient(responses=[_resp_no_boundary(tail_complete=False)])
    )
    complete = await adetect_boundaries(
        _dialogue(2), llm=FakeLLMClient(responses=[_resp_no_boundary(tail_complete=True)])
    )

    assert incomplete.tail == complete.tail
    assert incomplete.tail_complete != complete.tail_complete


async def test_tail_complete_survives_is_final_closure() -> None:
    """``is_final=True`` empties the tail, but the LLM's verdict is still what it was."""
    fake = FakeLLMClient(responses=[_resp_no_boundary(tail_complete=False)])

    result = await adetect_boundaries(_dialogue(2), llm=fake, is_final=True)

    assert result.tail == []
    assert result.tail_complete is False


async def test_empty_input_reports_no_verdict_rather_than_false() -> None:
    """No LLM call happened, so nothing judged the tail — ``None``, not a fabricated ``False``."""
    result = await adetect_boundaries([], llm=FakeLLMClient(responses=[]))

    assert result.tail_complete is None


# ===========================================================================
# Helper: _format_messages_with_indices
# ===========================================================================


def test_format_messages_renders_index_timestamp_and_sender_name() -> None:
    msgs = [
        _msg("user", "hi", _BASE_TS_MS, sender_name="Alice"),
        _msg("assistant", "hey", _BASE_TS_MS + 1000, sender_id="u_bot", sender_name="Bot"),
    ]
    rendered = _format_messages_with_indices(msgs)
    assert rendered.split("\n") == [
        "[1] [2023-11-14 22:13 UTC (Tuesday)] Alice: hi",
        "[2] [2023-11-14 22:13 UTC (Tuesday)] Bot: hey",
    ]


def test_format_messages_skips_a_text_less_message_before_reading_its_timestamp() -> None:
    """A dropped message keeps its number and, never rendered, may carry no timestamp."""
    msgs = [_msg("user", "", 0), _msg("user", "hi", _BASE_TS_MS, sender_name="Alice")]

    assert _format_messages_with_indices(msgs) == "[2] [2023-11-14 22:13 UTC (Tuesday)] Alice: hi"


def test_format_messages_rejects_a_message_with_text_but_no_timestamp() -> None:
    with pytest.raises(InvalidInputError, match=r"^chat message 'msg_0' has no timestamp: "):
        _format_messages_with_indices([_msg("user", "hi", 0)])


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
    split = _find_force_split_point(msgs, hard_token_limit=65_536, hard_msg_limit=5)
    assert split <= 4  # hard_msg_limit - 1


def test_find_force_split_point_halves_when_head_exceeds_token_limit() -> None:
    """Tiny token limit forces the halving loop to reduce the candidate."""
    long_msgs = [_msg("user", "x" * 500, _BASE_TS_MS + i, sender_name="Alice") for i in range(10)]
    split = _find_force_split_point(long_msgs, hard_token_limit=1, hard_msg_limit=500)
    # Halving loop bottoms out at 1 (floor guard).
    assert 1 <= split <= 9
