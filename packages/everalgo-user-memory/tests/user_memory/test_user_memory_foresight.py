"""Tests for everalgo.user_memory.foresight — ForesightExtractor.

No internal retry — exceptions propagate directly.  Empty list is a valid successful return.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.user_memory.foresight import (
    ForesightExtractor,
    _calculate_duration_days,
    _calculate_end_time_from_duration,
    _clean_date_string,
    _render_conversation,
    _resolve_user_name,
)


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


# ==========================================================================
# Language rules — mixed-input clauses (mirrors test_user_memory_episode.py)
# ==========================================================================

_MIXED_INPUT_CLAUSES_EN = (
    "themselves compose",  # judgement source restricted to participants' own writing
    "dominates the conversation by volume",  # long quoted material must not flip the judgement
    # An operational test for "is this pasted" — the negative instruction alone left the model unable to
    # recognise unmarked prose as pasted, which cost ~35% of judgements on that shape of input.
    "Apply this test to decide what is pasted",
    "whether or not it is wrapped in quotation marks or a code fence",
    "sentence structure",  # embedded foreign terms do not flip the judgement
    "keep their original form",  # proper nouns / technical terms stay untranslated
)


def test_en_prompt_has_language_rule() -> None:
    """The old rule enumerated only Chinese/English; the new rule is language-agnostic."""
    from everalgo.user_memory.prompts.en.foresight import FORESIGHT_GENERATION_PROMPT

    assert "CRITICAL LANGUAGE RULE" in FORESIGHT_GENERATION_PROMPT
    assert "in Chinese" not in FORESIGHT_GENERATION_PROMPT


def test_en_prompt_states_language_rule_at_both_ends() -> None:
    """Long prompts lose middle instructions; the rule is repeated at head and tail."""
    from everalgo.user_memory.prompts.en.foresight import FORESIGHT_GENERATION_PROMPT

    assert FORESIGHT_GENERATION_PROMPT.count("CRITICAL LANGUAGE RULE") == 2


@pytest.mark.parametrize("clause", _MIXED_INPUT_CLAUSES_EN)
def test_en_prompt_covers_mixed_input(clause: str) -> None:
    """Chinese question plus long English pasted material must still yield Chinese output."""
    from everalgo.user_memory.prompts.en.foresight import FORESIGHT_GENERATION_PROMPT

    assert clause in FORESIGHT_GENERATION_PROMPT


# zh must not rot relative to en — it is a public prompt selectable via `prompt=` (see README.md).
_MIXED_INPUT_CLAUSES_ZH = (
    "本人撰写的内容",  # judgement source restricted to participants' own writing
    "在篇幅上占据对话主体",  # long quoted material must not flip the judgement
    "判断何为粘贴材料时适用以下检验",  # operational test, mirrors the en clause above
    "也无论是否被引号或代码块包裹",
    "句子结构",  # embedded foreign terms do not flip the judgement
    "保留原文形式",  # proper nouns / technical terms stay untranslated
)


def test_zh_prompt_states_language_rule_at_both_ends() -> None:
    """The zh prompt previously had no standalone language rule at all."""
    from everalgo.user_memory.prompts.zh.foresight import FORESIGHT_GENERATION_PROMPT

    assert FORESIGHT_GENERATION_PROMPT.count("关键语言规则") == 2


@pytest.mark.parametrize("clause", _MIXED_INPUT_CLAUSES_ZH)
def test_zh_prompt_covers_mixed_input(clause: str) -> None:
    """Same mixed-input clauses as the en prompt."""
    from everalgo.user_memory.prompts.zh.foresight import FORESIGHT_GENERATION_PROMPT

    assert clause in FORESIGHT_GENERATION_PROMPT


def test_zh_no_longer_has_redundant_language_clause() -> None:
    """The closing-instruction clause is redundant once the head/tail rule is in place."""
    from everalgo.user_memory.prompts.zh.foresight import FORESIGHT_GENERATION_PROMPT

    assert "语言类型必须与" not in FORESIGHT_GENERATION_PROMPT
    assert "语言风格必须与事件场景匹配" in FORESIGHT_GENERATION_PROMPT


async def test_aextract_parses_wrapped_foresight_payload() -> None:
    """Wrapped JSON object with foresights array → list[Foresight]."""
    llm_json = (
        '{"foresights": [{'
        '"content": "Alice will follow up with Bob next week",'
        '"evidence": "Alice said she will follow up",'
        '"start_time": "2024-01-01",'
        '"end_time": "2024-01-08",'
        '"duration_days": 7'
        "}]}"
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    foresights = await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert len(foresights) == 1
    fs = foresights[0]
    assert fs.foresight == "Alice will follow up with Bob next week"
    assert fs.evidence == "Alice said she will follow up"
    assert fs.start_time == "2024-01-01"
    assert fs.end_time == "2024-01-08"
    assert fs.duration_days == 7
    assert fs.owner_id == "u_alice"


async def test_aextract_fallback_start_time_to_memcell_date() -> None:
    """When start_time is null/invalid, falls back to memcell.timestamp date."""
    llm_json = (
        '{"foresights": [{"content": "X", "evidence": "y", "start_time": "", "end_time": null, "duration_days": null}]}'
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    foresights = await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert len(foresights) == 1
    assert foresights[0].foresight == "X"
    # start_time falls back to memcell.timestamp formatted as YYYY-MM-DD
    assert foresights[0].start_time == "2023-11-14"  # 1700000000000 ms = 2023-11-14 UTC


async def test_aextract_owner_id_equals_sender_id() -> None:
    """``Foresight.owner_id`` must equal the ``sender_id`` argument."""
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content='{"foresights": [{"content": "x", "evidence": "y", "start_time": "2024-01-01"}]}', model="fake"
            )
        ]
    )

    foresights = await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert foresights[0].owner_id == "u_alice"


async def test_aextract_returns_empty_list_when_llm_returns_empty_foresights() -> None:
    """Empty foresights array from LLM → return [] immediately (no retry)."""
    fake = FakeLLMClient(responses=[ChatResponse(content='{"foresights": []}', model="fake")])

    result = await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert result == []
    assert fake.call_count == 1


async def test_aextract_raises_on_bad_json() -> None:
    """Unparseable JSON → ValueError on first attempt (no internal retry)."""
    bad_responses: list[str | ChatResponse] = [ChatResponse(content="not json", model="fake")]
    fake = FakeLLMClient(responses=bad_responses)

    with pytest.raises(ValueError):
        await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")

    assert fake.call_count == 1


async def test_aextract_skips_invalid_items() -> None:
    """Items with empty content are filtered."""
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content='{"foresights": [{"content": "valid", "evidence": "y", "start_time": "2024-01-01"}, {"content": "", "evidence": "y", "start_time": "2024-01-01"}]}',
                model="fake",
            )
        ]
    )
    foresights = await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")
    assert len(foresights) == 1
    assert foresights[0].foresight == "valid"


async def test_aextract_per_call_prompt_overrides_default() -> None:
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(content='{"foresights": []}', model="fake")

    fake = FakeLLMClient(handler=handler)
    custom = "CUSTOM FORESIGHT id={USER_ID} name={USER_NAME} conv={CONVERSATION_TEXT}"

    await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice", prompt=custom)

    assert captured["content"].startswith("CUSTOM FORESIGHT")
    assert "id=u_alice" in captured["content"]
    assert "name=Alice" in captured["content"]


# ==========================================================================
# Truncation at _FORESIGHT_MAX_COUNT
# ==========================================================================


async def test_aextract_truncates_when_more_than_10_foresights() -> None:
    """LLM returns 12 foresights → truncated to 10."""
    items = [{"content": f"f-{i}", "evidence": "e", "start_time": "2024-01-01"} for i in range(12)]
    fake = FakeLLMClient(responses=[ChatResponse(content=json.dumps({"foresights": items}), model="fake")])

    foresights = await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")
    assert len(foresights) == 10


# ==========================================================================
# _render_conversation skips empty content
# ==========================================================================


def test_render_conversation_skips_empty_content() -> None:
    """Messages with empty content are silently dropped."""
    cell = MemCell(
        items=[
            ChatMessage(id="m1", role="user", content="hi", timestamp=1, sender_id="u_alice", sender_name="Alice"),
            ChatMessage(id="m2", role="user", content="", timestamp=2, sender_id="u_bob", sender_name="Bob"),
        ],
        timestamp=2,
    )
    rendered = _render_conversation(cell)
    assert "Alice: hi" in rendered
    assert "Bob" not in rendered


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
# _clean_date_string regex / datetime branches
# ==========================================================================


def test_clean_date_string_returns_none_on_bad_regex() -> None:
    """Non-matching pattern (e.g. ``2024/01/05``) → None."""
    assert _clean_date_string("2024/01/05") is None
    assert _clean_date_string("abc") is None
    assert _clean_date_string("") is None
    assert _clean_date_string(None) is None


def test_clean_date_string_returns_none_when_datetime_invalid() -> None:
    """Regex passes but datetime() raises (e.g. Feb 30) → None."""
    assert _clean_date_string("2024-02-30") is None
    assert _clean_date_string("2024-13-01") is None


def test_clean_date_string_returns_normalised_string_on_success() -> None:
    """Stripped-and-validated date → ``YYYY-MM-DD``."""
    assert _clean_date_string("2024-03-14") == "2024-03-14"


# ==========================================================================
# _calculate_end_time_from_duration / _calculate_duration_days
# ==========================================================================


def test_calculate_end_time_from_duration_returns_iso_date() -> None:
    """Happy path: start + N days → end as ``YYYY-MM-DD``."""
    assert _calculate_end_time_from_duration("2024-03-14", 7) == "2024-03-21"


def test_calculate_end_time_from_duration_returns_none_on_bad_start() -> None:
    """``strptime`` failure → None."""
    assert _calculate_end_time_from_duration("not-a-date", 7) is None


def test_calculate_duration_days_returns_int() -> None:
    """Happy path."""
    assert _calculate_duration_days("2024-03-14", "2024-03-21") == 7


def test_calculate_duration_days_returns_none_on_bad_input() -> None:
    """``strptime`` failure → None."""
    assert _calculate_duration_days("not-a-date", "2024-03-21") is None
    assert _calculate_duration_days("2024-03-14", "not-a-date") is None


# ==========================================================================
# Time computation helpers
# ==========================================================================


async def test_aextract_computes_end_time_from_duration_when_only_duration_provided() -> None:
    """``start + duration`` → ``end_time`` computed."""
    raw = '{"foresights": [{"content": "c", "evidence": "e", "start_time": "2024-03-14", "end_time": null, "duration_days": 7}]}'
    fake = FakeLLMClient(responses=[ChatResponse(content=raw, model="fake")])
    foresights = await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")
    assert foresights[0].end_time == "2024-03-21"
    assert foresights[0].duration_days == 7


async def test_aextract_computes_duration_from_end_time_when_only_end_provided() -> None:
    """``start + end`` → ``duration_days`` computed."""
    raw = (
        '{"foresights": [{"content": "c", "evidence": "e", '
        '"start_time": "2024-03-14", "end_time": "2024-03-21", "duration_days": null}]}'
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=raw, model="fake")])
    foresights = await ForesightExtractor(llm=fake).aextract(_memcell(), sender_id="u_alice")
    assert foresights[0].duration_days == 7
    assert foresights[0].end_time == "2024-03-21"


# ==========================================================================
# Silent-skip contract — agent → user-memory pipeline
# ==========================================================================


async def test_aextract_silently_skips_non_chat_items() -> None:
    """ForesightExtractor must silently skip ToolCallRequest / ToolCallResult items.

    Locks the agent → user-memory pipeline contract: a MemCell with mixed items (ChatMessage +
    tool calls) must produce the same Foresight list as a chat-only MemCell with the same ChatMessages.
    """
    llm_json = '{"foresights": [{"content": "Alice will follow up with Bob next week", "evidence": "stated directly", "start_time": "2023-11-14"}]}'

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

    fake_chat = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])
    fake_mixed = FakeLLMClient(responses=[ChatResponse(content=llm_json, model="fake")])

    fs_chat = await ForesightExtractor(llm=fake_chat).aextract(chat_only_cell, sender_id="u_alice")
    fs_mixed = await ForesightExtractor(llm=fake_mixed).aextract(mixed_cell, sender_id="u_alice")

    assert len(fs_chat) == len(fs_mixed) == 1
    assert fs_chat[0].foresight == fs_mixed[0].foresight
    assert fs_chat[0].owner_id == fs_mixed[0].owner_id
