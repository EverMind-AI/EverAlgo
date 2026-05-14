"""Tests for everalgo.user_memory.foresight — ForesightExtractor (opensource port)."""

from __future__ import annotations

from typing import Any

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import MemCell, Message, MessageRole
from everalgo.user_memory.foresight import (
    ForesightExtractor,
    _calculate_duration_days,
    _calculate_end_time_from_duration,
    _clean_date_string,
    _derive_owner_id,
    _derive_user_name,
    _parse_llm_response,
    _render_conversation,
)


def _memcell() -> MemCell:
    msg = Message(
        role=MessageRole.USER,
        content="I'll follow up with Bob next week.",
        timestamp=1700000000000,
        sender_id="u_alice",
        sender_name="Alice",
    )
    return MemCell(
        event_id="mc_fs_001",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1700000000000,
        participants=["u_alice"],
        sender_ids=["u_alice"],
    )


async def test_aextract_parses_opensource_array_payload() -> None:
    """Top-level JSON array of {content, evidence, ...} → list[Foresight]."""
    llm_json = (
        "[{"
        '"content": "Alice will follow up with Bob next week",'
        '"evidence": "Alice said she will follow up",'
        '"start_time": "2024-01-01",'
        '"end_time": "2024-01-08",'
        '"duration_days": 7'
        "}]"
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    foresights = await ForesightExtractor().aextract(_memcell(), llm=fake)

    assert len(foresights) == 1
    fs = foresights[0]
    assert fs.foresight == "Alice will follow up with Bob next week"
    assert fs.evidence == "Alice said she will follow up"
    assert fs.start_time == "2024-01-01"
    assert fs.end_time == "2024-01-08"
    assert fs.duration_days == 7
    assert fs.owner_id == "u_alice"


async def test_aextract_parses_wrapped_foresights_payload() -> None:
    """{"foresights": [...]} wrapped form; start_time falls back to memcell.timestamp date."""
    llm_json = (
        '{"foresights": [{'
        '"content": "X", "evidence": "y", '
        '"start_time": null, "end_time": null, "duration_days": null'
        "}]}"
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    foresights = await ForesightExtractor().aextract(_memcell(), llm=fake)

    assert len(foresights) == 1
    assert foresights[0].foresight == "X"
    # start_time falls back to memcell.timestamp formatted as YYYY-MM-DD (matches opensource line 326-336)
    assert foresights[0].start_time == "2023-11-14"  # 1700000000000 ms = 2023-11-14 UTC


async def test_aextract_auto_fills_parent_id_from_memcell() -> None:
    fake = FakeLLMClient(responses=[ChatResponse(content='[{"content": "x", "evidence": "y"}]', model="fake")])
    mc = _memcell()

    foresights = await ForesightExtractor().aextract(mc, llm=fake)

    assert foresights[0].parent_id == mc.event_id
    assert foresights[0].parent_type == "memcell"


async def test_aextract_returns_empty_list_after_5_retries_on_empty_array() -> None:
    """Empty array → ValueError → retry 5 times → return [] (matches opensource line 138-162)."""
    empty = ChatResponse(content="[]", model="fake")
    fake = FakeLLMClient(responses=[empty, empty, empty, empty, empty])
    assert await ForesightExtractor().aextract(_memcell(), llm=fake) == []
    assert fake.call_count == 5


async def test_aextract_skips_invalid_items() -> None:
    """Items with empty content are filtered."""
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content='[{"content": "valid", "evidence": "y"}, {"content": "", "evidence": "y"}]',
                model="fake",
            )
        ]
    )
    foresights = await ForesightExtractor().aextract(_memcell(), llm=fake)
    assert len(foresights) == 1
    assert foresights[0].foresight == "valid"


async def test_aextract_per_call_prompt_overrides_default() -> None:
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content="[]", model="fake")

    fake = FakeLLMClient(handler=handler)
    custom = "CUSTOM FORESIGHT id={USER_ID} name={USER_NAME} conv={CONVERSATION_TEXT}"

    await ForesightExtractor().aextract(_memcell(), llm=fake, prompt=custom)

    assert captured["content"].startswith("CUSTOM FORESIGHT")
    assert "id=u_alice" in captured["content"]
    assert "name=Alice" in captured["content"]


# ==========================================================================
# Truncation at _FORESIGHT_MAX_COUNT (line 104)
# ==========================================================================


async def test_aextract_truncates_when_more_than_10_foresights() -> None:
    """LLM returns 12 foresights → truncated to 10 (line 104)."""
    import json

    items = [{"content": f"f-{i}", "evidence": "e"} for i in range(12)]
    fake = FakeLLMClient(responses=[ChatResponse(content=json.dumps(items), model="fake")])

    foresights = await ForesightExtractor().aextract(_memcell(), llm=fake)
    assert len(foresights) == 10


# ==========================================================================
# _render_conversation skips empty content (line 129)
# ==========================================================================


def test_render_conversation_skips_empty_content() -> None:
    """Messages with empty content are silently dropped (line 129)."""
    real = Message(role=MessageRole.USER, content="hi", timestamp=1, sender_name="Alice")
    empty = Message(role=MessageRole.USER, content="", timestamp=2, sender_name="Bob")
    cell = MemCell(
        event_id="mc_x",
        original_data=[
            {"message": real.model_dump(exclude_none=True)},
            {"message": empty.model_dump(exclude_none=True)},
        ],
        timestamp=2,
    )
    rendered = _render_conversation(cell)
    assert "Alice: hi" in rendered
    assert "Bob" not in rendered


# ==========================================================================
# _derive_owner_id + _derive_user_name fallbacks (lines 139-142, 150)
# ==========================================================================


def test_derive_owner_id_falls_back_to_message_sender_id() -> None:
    """No participants → first message with sender_id wins (lines 139-141)."""
    msg = Message(role=MessageRole.USER, content="x", timestamp=1, sender_id="u_from_msg")
    cell = MemCell(
        event_id="mc_x",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1,
    )
    assert _derive_owner_id(cell) == "u_from_msg"


def test_derive_owner_id_returns_u_default_when_nothing_identifies_user() -> None:
    """No participants, no sender_id → ``u_default`` (line 142)."""
    msg = Message(role=MessageRole.USER, content="x", timestamp=1)
    cell = MemCell(
        event_id="mc_x",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1,
    )
    assert _derive_owner_id(cell) == "u_default"


def test_derive_user_name_falls_back_to_owner_id_when_no_sender_name() -> None:
    """No matching sender_name in any message → owner_id used as user_name (line 150)."""
    msg = Message(role=MessageRole.USER, content="x", timestamp=1, sender_id="u_alice")
    cell = MemCell(
        event_id="mc_x",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1,
        participants=["u_alice"],
    )
    assert _derive_user_name(cell, "u_alice") == "u_alice"


# ==========================================================================
# _clean_date_string regex / datetime branches (lines 168, 172-173)
# ==========================================================================


def test_clean_date_string_returns_none_on_bad_regex() -> None:
    """Non-matching pattern (e.g. ``2024/01/05``) → None (line 168)."""
    assert _clean_date_string("2024/01/05") is None
    assert _clean_date_string("abc") is None
    assert _clean_date_string("") is None
    assert _clean_date_string(None) is None


def test_clean_date_string_returns_none_when_datetime_invalid() -> None:
    """Regex passes but datetime() raises (e.g. Feb 30) → None (lines 172-173)."""
    assert _clean_date_string("2024-02-30") is None
    assert _clean_date_string("2024-13-01") is None


def test_clean_date_string_returns_normalised_string_on_success() -> None:
    """Stripped-and-validated date → ``YYYY-MM-DD``."""
    assert _clean_date_string("2024-03-14") == "2024-03-14"


# ==========================================================================
# _calculate_end_time_from_duration / _calculate_duration_days (lines 179-194)
# ==========================================================================


def test_calculate_end_time_from_duration_returns_iso_date() -> None:
    """Happy path: start + N days → end as ``YYYY-MM-DD`` (line 184)."""
    assert _calculate_end_time_from_duration("2024-03-14", 7) == "2024-03-21"


def test_calculate_end_time_from_duration_returns_none_on_bad_start() -> None:
    """``strptime`` failure → None (lines 182-183)."""
    assert _calculate_end_time_from_duration("not-a-date", 7) is None


def test_calculate_duration_days_returns_int() -> None:
    """Happy path (line 194)."""
    assert _calculate_duration_days("2024-03-14", "2024-03-21") == 7


def test_calculate_duration_days_returns_none_on_bad_input() -> None:
    """``strptime`` failure → None (lines 192-193)."""
    assert _calculate_duration_days("not-a-date", "2024-03-21") is None
    assert _calculate_duration_days("2024-03-14", "not-a-date") is None


# ==========================================================================
# _parse_and_build_foresights branches (lines 219, 221, 226, 241, 243)
# ==========================================================================


async def test_aextract_returns_empty_when_wrapped_inner_not_a_list() -> None:
    """``{"foresights": "not a list"}`` → [] → 5 retries → return [] (line 219)."""
    bad = ChatResponse(content='{"foresights": "not a list"}', model="fake")
    fake = FakeLLMClient(responses=[bad, bad, bad, bad, bad])
    assert await ForesightExtractor().aextract(_memcell(), llm=fake) == []
    assert fake.call_count == 5


async def test_aextract_returns_empty_when_top_level_is_neither_list_nor_dict() -> None:
    """Top-level scalar (e.g. number) → [] → 5 retries → return [] (line 221)."""
    bad = ChatResponse(content="42", model="fake")
    fake = FakeLLMClient(responses=[bad, bad, bad, bad, bad])
    assert await ForesightExtractor().aextract(_memcell(), llm=fake) == []


async def test_aextract_skips_non_dict_items_in_array() -> None:
    """Mixed array with non-dict entries: non-dict items skipped (line 226)."""
    raw = '[{"content": "valid", "evidence": "e"}, "string-item", 42, null]'
    fake = FakeLLMClient(responses=[ChatResponse(content=raw, model="fake")])
    foresights = await ForesightExtractor().aextract(_memcell(), llm=fake)
    assert len(foresights) == 1
    assert foresights[0].foresight == "valid"


async def test_aextract_computes_end_time_from_duration_when_only_duration_provided() -> None:
    """``start + duration`` → ``end_time`` computed (line 241)."""
    raw = '[{"content": "c", "evidence": "e", "start_time": "2024-03-14", "end_time": null, "duration_days": 7}]'
    fake = FakeLLMClient(responses=[ChatResponse(content=raw, model="fake")])
    foresights = await ForesightExtractor().aextract(_memcell(), llm=fake)
    assert foresights[0].end_time == "2024-03-21"
    assert foresights[0].duration_days == 7


async def test_aextract_computes_duration_from_end_time_when_only_end_provided() -> None:
    """``start + end`` → ``duration_days`` computed (line 243)."""
    raw = (
        '[{"content": "c", "evidence": "e", '
        '"start_time": "2024-03-14", "end_time": "2024-03-21", "duration_days": null}]'
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=raw, model="fake")])
    foresights = await ForesightExtractor().aextract(_memcell(), llm=fake)
    assert foresights[0].duration_days == 7
    assert foresights[0].end_time == "2024-03-21"


# ==========================================================================
# _parse_llm_response — json fence path (lines 269-275)
# ==========================================================================


def test_parse_llm_response_handles_json_fence() -> None:
    """`````json ... ````` fenced response (lines 269-275)."""
    raw = '```json\n[{"content": "c", "evidence": "e"}]\n```'
    parsed = _parse_llm_response(raw)
    assert parsed == [{"content": "c", "evidence": "e"}]
