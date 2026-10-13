"""Tests for everalgo.user_memory.foresight — ForesightExtractor and the foresight reply contract.

One call, no internal retry — exceptions propagate directly. An empty list is a valid successful return. The prompt's
system / user split is pinned in ``test_foresight_prompt_split.py``, the conversation line format in
``test_foresight_render.py``.
"""

from __future__ import annotations

import inspect
import json
from typing import Any

import pytest

from everalgo.llm import LLMOutputError, StrictOutputModel, to_openai_response_format
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, Foresight, MemCell, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.user_memory import OutputLanguage
from everalgo.user_memory._language import PARTICIPANT_LANGUAGE_RULE
from everalgo.user_memory._replies import _ForesightItem, _ForesightsReply
from everalgo.user_memory.foresight import (
    ForesightExtractor,
    _calculate_duration_days,
    _calculate_end_time_from_duration,
    _clean_date_string,
    _resolve_user_name,
)

_MEMCELL_DATE = "2023-11-14"  # 1700000000000 ms, UTC


def _memcell() -> MemCell:
    return MemCell(
        items=[
            ChatMessage(
                id="m1",
                role="user",
                content="I'll follow up with Bob next week.",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            )
        ],
        timestamp=1700000000000,
    )


def _item(**overrides: object) -> dict[str, object]:
    item: dict[str, object] = {
        "content": "c",
        "evidence": "e",
        "start_time": "2024-01-01",
        "end_time": "2024-01-08",
        "duration_days": 7,
    }
    return item | overrides


def _reply(*items: dict[str, object]) -> str:
    return json.dumps({"foresights": list(items)})


async def _extract(*replies: str, **kwargs: Any) -> list[Foresight]:
    fake = FakeLLMClient(responses=list(replies))
    return await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice", **kwargs)


async def _sent(**kwargs: Any) -> tuple[str, str]:
    """The system and user message contents of one extraction."""
    fake = FakeLLMClient(responses=[_reply()])
    await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice", **kwargs)
    system, user = fake.calls[0].messages
    assert (system.role, user.role) == ("system", "user")
    return str(system.content), str(user.content)


class _ClientError(Exception):
    """A host-side client failure the operator must not wrap or retype."""


# ==========================================================================
# Language rules — mixed-input clauses (mirrors test_user_memory_atomic_fact.py)
# ==========================================================================


async def test_system_carries_the_participant_rule_once_when_no_language_is_named() -> None:
    system, user = await _sent()

    assert system.count("CRITICAL LANGUAGE RULE") == 1
    assert PARTICIPANT_LANGUAGE_RULE in system
    assert "{{language_rule}}" not in system + user


async def test_system_carries_the_named_language_once() -> None:
    system, _ = await _sent(output_language=OutputLanguage.GERMAN)

    assert (
        system.count(
            "Write every output field in German, whatever language the input is in, including anything already stored."
        )
        == 1
    )
    assert "CRITICAL LANGUAGE RULE" not in system
    assert "the language the participants use" not in system


async def test_the_language_rule_closes_the_system_message_inside_its_tag() -> None:
    system, _ = await _sent(output_language="german")

    assert system.endswith("</output_language>")
    assert system.index("<reply_format>") < system.index("<output_language>")


async def test_an_unknown_language_raises_before_any_call() -> None:
    fake = FakeLLMClient(responses=[_reply()])

    with pytest.raises(ValueError, match="unsupported output_language"):
        await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice", output_language="Klingon")

    assert fake.call_count == 0


# ==========================================================================
# What the model is sent
# ==========================================================================


async def test_user_message_carries_the_three_data_elements_in_order() -> None:
    _, user = await _sent()

    assert user == (
        "<user_id>\nu_alice\n</user_id>\n\n"
        "<user_name>\nAlice\n</user_name>\n\n"
        "<conversation>\n[2023-11-14 22:13 UTC (Tuesday)] Alice: I'll follow up with Bob next week.\n</conversation>"
    )


async def test_system_message_names_no_request_data() -> None:
    system, _ = await _sent()

    for datum in ("u_alice", "Alice", "follow up with Bob"):
        assert datum not in system


async def test_one_call_requests_the_reply_model_at_the_foresight_temperature() -> None:
    fake = FakeLLMClient(responses=[_reply()])

    await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert fake.call_count == 1
    assert fake.calls[0].response_format is _ForesightsReply
    assert fake.calls[0].temperature == 0.3


def test_aextract_takes_no_prompt_override() -> None:
    assert "prompt" not in inspect.signature(ForesightExtractor.aextract).parameters


# ==========================================================================
# Reply contract
# ==========================================================================


def test_reply_models_are_strict_with_no_defaults() -> None:
    for model in (_ForesightsReply, _ForesightItem):
        assert issubclass(model, StrictOutputModel)
        assert all(field.is_required() for field in model.model_fields.values())


def test_reply_model_is_inside_the_strict_schema_subset() -> None:
    assert to_openai_response_format(_ForesightsReply)


def test_reply_model_fields_follow_the_prompt_skeleton() -> None:
    assert list(_ForesightItem.model_fields) == ["content", "evidence", "start_time", "end_time", "duration_days"]
    assert list(_ForesightsReply.model_fields) == ["foresights"]


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param("not json", id="not-json"),
        pytest.param('{"foresights": [', id="truncated"),
        pytest.param("[]", id="top-level-array"),
        pytest.param("{}", id="missing-foresights"),
        pytest.param('{"foresights": {}}', id="foresights-not-a-list"),
        pytest.param(json.dumps({"foresights": [], "extra": "x"}), id="extra-top-level-field"),
        pytest.param(
            json.dumps({"foresights": [{k: v for k, v in _item().items() if k != "evidence"}]}), id="missing-field"
        ),
        pytest.param(_reply(_item(note="x")), id="extra-item-field"),
        pytest.param(_reply(_item(duration_days="7")), id="duration-as-string"),
        pytest.param(_reply(_item(content=None)), id="null-content"),
    ],
)
async def test_a_bad_reply_raises_llm_output_error_without_retry(bad: str) -> None:
    fake = FakeLLMClient(responses=[bad])

    with pytest.raises(LLMOutputError, match=r"^foresight reply is not a valid _ForesightsReply: "):
        await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert fake.call_count == 1


async def test_the_error_message_never_carries_the_reply() -> None:
    fake = FakeLLMClient(responses=[_reply(_item(secret_payload="HUSH-HUSH"))])

    with pytest.raises(LLMOutputError) as caught:
        await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert "HUSH-HUSH" not in str(caught.value)


async def test_a_client_exception_propagates_unchanged() -> None:
    fake = FakeLLMClient(responses=[_ClientError("upstream down")])

    with pytest.raises(_ClientError, match="upstream down"):
        await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")


# ==========================================================================
# Reply → Foresight
# ==========================================================================


async def test_aextract_parses_wrapped_foresight_payload() -> None:
    """Wrapped JSON object with foresights array → list[Foresight]."""
    foresights = await _extract(
        _reply(
            _item(
                content="Alice will follow up with Bob next week",
                evidence="Alice said she will follow up",
            )
        )
    )

    assert len(foresights) == 1
    fs = foresights[0]
    assert fs.foresight == "Alice will follow up with Bob next week"
    assert fs.evidence == "Alice said she will follow up"
    assert fs.start_time == "2024-01-01"
    assert fs.end_time == "2024-01-08"
    assert fs.duration_days == 7
    assert fs.owner_id == "u_alice"
    assert fs.timestamp == 1700000000000


@pytest.mark.parametrize("start_time", ["", None, "someday", "2024-02-30"])
async def test_aextract_fallback_start_time_to_memcell_date(start_time: str | None) -> None:
    """When start_time is null/invalid, falls back to memcell.items[0].timestamp's date."""
    foresights = await _extract(_reply(_item(content="X", start_time=start_time, end_time=None, duration_days=None)))

    assert len(foresights) == 1
    assert foresights[0].foresight == "X"
    assert foresights[0].start_time == _MEMCELL_DATE


async def test_aextract_returns_empty_list_when_llm_returns_empty_foresights() -> None:
    """Empty foresights array from LLM → return [] immediately (no retry)."""
    fake = FakeLLMClient(responses=[_reply()])

    result = await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert result == []
    assert fake.call_count == 1


async def test_aextract_skips_items_with_blank_content_and_strips_the_rest() -> None:
    foresights = await _extract(_reply(_item(content="  valid  "), _item(content=""), _item(content="  ")))

    assert [fs.foresight for fs in foresights] == ["valid"]


async def test_aextract_truncates_when_more_than_10_foresights() -> None:
    """LLM returns 12 foresights → truncated to 10."""
    foresights = await _extract(_reply(*(_item(content=f"f-{i}") for i in range(12))))

    assert [fs.foresight for fs in foresights] == [f"f-{i}" for i in range(10)]


# ==========================================================================
# _resolve_user_name
# ==========================================================================


def test_resolve_user_name_returns_sender_name_when_matched() -> None:
    """Returns sender_name for messages with matching sender_id."""
    cell = MemCell(
        items=[ChatMessage(id="m1", role="user", content="x", timestamp=1, sender_id="u_alice", sender_name="Alice")],
        timestamp=1,
    )
    assert _resolve_user_name(cell, "u_alice") == "Alice"


def test_resolve_user_name_falls_back_to_sender_id_when_no_match() -> None:
    """No matching sender_name → fall back to sender_id literal."""
    cell = MemCell(
        items=[ChatMessage(id="m1", role="user", content="x", timestamp=1, sender_id="u_alice")],
        timestamp=1,
    )
    assert _resolve_user_name(cell, "u_alice") == "u_alice"


# ==========================================================================
# _clean_date_string
# ==========================================================================


def test_clean_date_string_returns_none_on_bad_regex() -> None:
    """Non-matching pattern (e.g. ``2024/01/05``) → None."""
    assert _clean_date_string("2024/01/05") is None
    assert _clean_date_string("abc") is None
    assert _clean_date_string("") is None
    assert _clean_date_string(None) is None


def test_clean_date_string_returns_none_when_datetime_invalid() -> None:
    """``YYYY-MM-DD`` shape but no such date (e.g. Feb 30) → None."""
    assert _clean_date_string("2024-02-30") is None
    assert _clean_date_string("2024-13-01") is None


def test_clean_date_string_returns_normalised_string_on_success() -> None:
    """Stripped-and-validated date → ``YYYY-MM-DD``."""
    assert _clean_date_string("2024-03-14") == "2024-03-14"
    assert _clean_date_string(" 2024-03-14 ") == "2024-03-14"


# ==========================================================================
# _calculate_end_time_from_duration / _calculate_duration_days
# ==========================================================================


def test_calculate_end_time_from_duration_returns_iso_date() -> None:
    """Happy path: start + N days → end as ``YYYY-MM-DD``."""
    assert _calculate_end_time_from_duration("2024-03-14", 7) == "2024-03-21"


def test_calculate_end_time_from_duration_returns_none_on_bad_start() -> None:
    """Start is not a ``YYYY-MM-DD`` date → None."""
    assert _calculate_end_time_from_duration("not-a-date", 7) is None
    assert _calculate_end_time_from_duration("2024-03-14T00:00:00Z", 7) is None


def test_calculate_end_time_from_duration_returns_none_past_the_calendar() -> None:
    """An end beyond year 9999 is not computable → None rather than an ``OverflowError``."""
    assert _calculate_end_time_from_duration("9999-12-31", 1) is None


def test_calculate_duration_days_returns_int() -> None:
    """Happy path."""
    assert _calculate_duration_days("2024-03-14", "2024-03-21") == 7


def test_calculate_duration_days_returns_none_on_bad_input() -> None:
    """Either end is not a ``YYYY-MM-DD`` date → None."""
    assert _calculate_duration_days("not-a-date", "2024-03-21") is None
    assert _calculate_duration_days("2024-03-14", "not-a-date") is None


# ==========================================================================
# Time computation helpers
# ==========================================================================


async def test_aextract_computes_end_time_from_duration_when_only_duration_provided() -> None:
    """``start + duration`` → ``end_time`` computed."""
    foresights = await _extract(_reply(_item(start_time="2024-03-14", end_time=None, duration_days=7)))

    assert foresights[0].end_time == "2024-03-21"
    assert foresights[0].duration_days == 7


async def test_aextract_computes_duration_from_end_time_when_only_end_provided() -> None:
    """``start + end`` → ``duration_days`` computed."""
    foresights = await _extract(_reply(_item(start_time="2024-03-14", end_time="2024-03-21", duration_days=None)))

    assert foresights[0].duration_days == 7
    assert foresights[0].end_time == "2024-03-21"


async def test_aextract_computes_the_window_from_the_fallback_start_when_start_is_missing() -> None:
    foresights = await _extract(_reply(_item(start_time=None, end_time=None, duration_days=3)))

    assert (foresights[0].start_time, foresights[0].end_time) == (_MEMCELL_DATE, "2023-11-17")


async def test_aextract_drops_an_end_time_past_the_calendar_instead_of_raising() -> None:
    foresights = await _extract(_reply(_item(start_time="9999-12-31", end_time=None, duration_days=1)))

    assert foresights[0].end_time is None
    assert foresights[0].duration_days == 1


async def test_aextract_keeps_both_empty_when_the_model_gives_neither() -> None:
    foresights = await _extract(_reply(_item(end_time=None, duration_days=None)))

    assert (foresights[0].end_time, foresights[0].duration_days) == (None, None)


# ==========================================================================
# Silent-skip contract — agent → user-memory pipeline
# ==========================================================================


async def test_aextract_silently_skips_non_chat_items() -> None:
    """ForesightExtractor must silently skip ToolCallRequest / ToolCallResult items.

    Locks the agent → user-memory pipeline contract: a MemCell with mixed items (ChatMessage +
    tool calls) must produce the same Foresight list as a chat-only MemCell with the same ChatMessages.
    """
    reply = _reply(_item(content="Alice will follow up with Bob next week", evidence="stated directly"))

    chat_only_cell = MemCell(
        items=[
            ChatMessage(
                id="c1",
                role="user",
                content="I'll follow up with Bob next week.",
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
                content="I'll follow up with Bob next week.",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ToolCallRequest(
                tool_calls=[
                    ToolCall(
                        id="tc1", function=ToolCallFunction(name="calendar.remind", arguments='{"when": "next week"}')
                    )
                ],
                timestamp=1700000001000,
                sender_id="assistant",
            ),
            ToolCallResult(
                tool_call_id="tc1",
                content="Reminder set.",
                timestamp=1700000002000,
            ),
        ],
        timestamp=1700000002000,
    )

    fake_chat = FakeLLMClient(responses=[reply])
    fake_mixed = FakeLLMClient(responses=[reply])

    fs_chat = await ForesightExtractor(llm=fake_chat).aextract(chat_only_cell, sender_id="u_alice")
    fs_mixed = await ForesightExtractor(llm=fake_mixed).aextract(mixed_cell, sender_id="u_alice")

    assert len(fs_chat) == len(fs_mixed) == 1
    assert fs_chat[0].foresight == fs_mixed[0].foresight
    assert fs_chat[0].owner_id == fs_mixed[0].owner_id
    assert fake_chat.calls[0].messages == fake_mixed.calls[0].messages
