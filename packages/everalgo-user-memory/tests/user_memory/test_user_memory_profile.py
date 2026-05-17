"""Tests for everalgo.user_memory.profile — ProfileExtractor.

Uses a single LLM call against ``PROFILE_INITIAL_EXTRACTION_PROMPT`` returning
``{explicit_info, implicit_traits}``. No internal retry — exceptions propagate directly to the caller.

UPDATE-mode tests cover:
- update no compact: ops applied, LLM called once, result has merged items.
- update triggers compact: post-merge item count > threshold, second LLM call runs.
- update preserves new_explicit_info merge correctness: add + update + delete ops each applied correctly.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell, Profile, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.user_memory.profile import (
    _PROFILE_COMPACT_THRESHOLD,
    ProfileExtractor,
    _build_summary,
    _parse_profile_payload,
    _render_conversation,
)

_TS = 1700000010000
_TS_OLD_1 = 1690000000000
_TS_OLD_2 = 1695000000000

_msg_id = 0


def _msg(
    text: str,
    role: str = "user",
    ts: int = _TS,
    sender: str = "u_alice",
    sender_name: str | None = "Alice",
    msg_id: str | None = None,
) -> ChatMessage:
    global _msg_id
    _msg_id += 1
    return ChatMessage(
        id=msg_id if msg_id is not None else f"m{_msg_id}",
        role=role,  # type: ignore[arg-type]
        content=text,
        timestamp=ts,
        sender_id=sender,
        sender_name=sender_name,
    )


def _memcell(ts: int = _TS, content: str = "Default content") -> MemCell:
    return MemCell(
        items=[_msg(content, ts=ts)],
        timestamp=ts,
    )


def _cluster() -> list[MemCell]:
    return [
        MemCell(
            items=[_msg("User asked about Python async patterns.", ts=_TS_OLD_1)],
            timestamp=_TS_OLD_1,
        ),
        MemCell(
            items=[_msg("User mentioned they prefer ruff over black.", ts=_TS_OLD_2)],
            timestamp=_TS_OLD_2,
        ),
    ]


def _payload(explicit_info: list[dict[str, Any]], implicit_traits: list[dict[str, Any]]) -> str:
    return json.dumps({"explicit_info": explicit_info, "implicit_traits": implicit_traits})


async def test_aextract_builds_profile_from_explicit_info() -> None:
    """``summary`` derives from first explicit_info description; lists preserved as extras."""
    payload = _payload(
        explicit_info=[
            {
                "category": "Technical Skills",
                "description": "Alice is a Python developer focusing on async patterns.",
                "evidence": "Alice asked about async patterns.",
            },
        ],
        implicit_traits=[
            {
                "trait": "[Pragmatic]",
                "description": "Prefers tooling that minimises ceremony.",
                "basis": "Repeated preference for ruff over black.",
                "evidence": "Alice mentioned ruff preference.",
            },
        ],
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=payload, model="fake")])

    profile = await ProfileExtractor(llm=fake).aextract([*_cluster(), _memcell()], sender_id="u_alice")

    assert profile.owner_id == "u_alice"
    assert "Python developer" in profile.summary
    # Extras preserved via Profile.model_config extra="allow"
    assert profile.explicit_info[0]["category"] == "Technical Skills"  # type: ignore[attr-defined]
    assert profile.implicit_traits[0]["trait"] == "[Pragmatic]"  # type: ignore[attr-defined]


async def test_aextract_summary_falls_back_to_implicit_trait_when_explicit_empty() -> None:
    payload = _payload(
        explicit_info=[],
        implicit_traits=[
            {"trait": "[Pragmatic]", "description": "Prefers minimal-ceremony tooling.", "evidence": "x"},
        ],
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=payload, model="fake")])

    profile = await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice")

    assert profile.summary == "Prefers minimal-ceremony tooling."


async def test_aextract_raises_on_payload_missing_required_keys() -> None:
    """Payload without both explicit_info and implicit_traits → ValueError (no retry)."""
    bad = ChatResponse(content="{}", model="fake")
    fake = FakeLLMClient(responses=[bad])

    with pytest.raises(ValueError, match="missing both explicit_info and implicit_traits"):
        await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice")

    assert fake.call_count == 1


async def test_aextract_raises_on_bad_json() -> None:
    """Unparseable JSON → JSONDecodeError propagates immediately (no retry)."""
    bad = ChatResponse(content="not json", model="fake")
    fake = FakeLLMClient(responses=[bad])

    with pytest.raises((json.JSONDecodeError, ValueError)):
        await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice")

    assert fake.call_count == 1


async def test_aextract_owner_id_equals_sender_id() -> None:
    """``Profile.owner_id`` must equal the ``sender_id`` argument."""
    payload = _payload(
        explicit_info=[{"category": "x", "description": "y", "evidence": "z"}],
        implicit_traits=[],
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=payload, model="fake")])

    profile = await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_custom")

    assert profile.owner_id == "u_custom"


async def test_aextract_renders_cluster_into_conversation_text() -> None:
    """Cluster ChatMemCells are stitched into the {conversation_text} placeholder."""
    captured: dict[str, str] = {}
    payload = _payload(
        explicit_info=[{"category": "x", "description": "y", "evidence": "z"}],
        implicit_traits=[],
    )

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["prompt"] = messages[0].content
        return ChatResponse(content=payload, model="fake")

    fake = FakeLLMClient(handler=handler)

    await ProfileExtractor(llm=fake).aextract([*_cluster(), _memcell()], sender_id="u_alice")

    assert "Python async patterns" in captured["prompt"]


async def test_aextract_per_call_prompt_overrides_default() -> None:
    captured: dict[str, str] = {}
    payload = _payload(
        explicit_info=[{"category": "x", "description": "y", "evidence": "z"}],
        implicit_traits=[],
    )

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["prompt"] = messages[0].content
        return ChatResponse(content=payload, model="fake")

    fake = FakeLLMClient(handler=handler)
    custom = "CUSTOM PROFILE conv={conversation_text}"

    await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice", prompt=custom)

    assert captured["prompt"].startswith("CUSTOM PROFILE conv=")
    assert "Default content" in captured["prompt"]


# ==========================================================================
# Defensive type guards (lines 71, 73)
# ==========================================================================


async def test_aextract_coerces_non_list_explicit_info_to_empty() -> None:
    """When LLM returns a non-list ``explicit_info`` (e.g. dict), coerce to []."""
    payload = '{"explicit_info": {"not": "a list"}, "implicit_traits": []}'
    fake = FakeLLMClient(responses=[ChatResponse(content=payload, model="fake")])

    profile = await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice")

    assert profile.summary == "(no summary)"  # empty list → sentinel


async def test_aextract_coerces_non_list_implicit_traits_to_empty() -> None:
    """When LLM returns a non-list ``implicit_traits``, coerce to []."""
    payload = (
        '{"explicit_info": [{"category": "x", "description": "y", "evidence": "z"}], "implicit_traits": "not a list"}'
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=payload, model="fake")])

    profile = await ProfileExtractor(llm=fake).aextract([_memcell()], sender_id="u_alice")

    assert profile.summary == "y"  # explicit list still used


# ==========================================================================
# _render_conversation helper
# ==========================================================================


def test_render_conversation_skips_empty_content() -> None:
    """Messages with empty content are silently dropped."""
    cell = MemCell(
        items=[
            ChatMessage(
                id="m1", role="user", content="hello", timestamp=1700000000000, sender_id="u_alice", sender_name="Alice"
            ),
            ChatMessage(
                id="m2", role="user", content="", timestamp=1700000001000, sender_id="u_bob", sender_name="Bob"
            ),
        ],
        timestamp=1700000001000,
    )
    rendered = _render_conversation([cell])
    assert "Alice" in rendered
    assert "Bob" not in rendered


def test_render_conversation_uses_sentinel_when_all_inputs_empty() -> None:
    """Empty cluster + empty messages → sentinel line."""
    empty_cell = MemCell(
        items=[],
        timestamp=1700000000000,
    )
    rendered = _render_conversation([empty_cell])
    assert rendered == "(no prior MemCells in the cluster)"


# ==========================================================================
# _parse_profile_payload
# ==========================================================================


def test_parse_profile_payload_raises_on_non_object() -> None:
    """Top-level JSON that isn't an object raises ValueError."""
    with pytest.raises(ValueError, match="not a JSON object"):
        _parse_profile_payload("[1, 2, 3]")


# ==========================================================================
# _build_summary defensive branches
# ==========================================================================


def test_build_summary_skips_non_dict_items_in_explicit_info() -> None:
    """Non-dict entries inside explicit_info are skipped."""
    summary = _build_summary(
        explicit_info=["not a dict", 42, {"description": "real one"}],
        implicit_traits=[],
    )
    assert summary == "real one"


def test_build_summary_skips_non_dict_items_in_implicit_traits() -> None:
    """Non-dict entries inside implicit_traits are skipped."""
    summary = _build_summary(
        explicit_info=[],
        implicit_traits=["not a dict", {"description": "trait desc"}],
    )
    assert summary == "trait desc"


def test_build_summary_returns_sentinel_when_no_usable_description() -> None:
    """Empty / whitespace-only / non-string descriptions → sentinel."""
    summary = _build_summary(
        explicit_info=[{"description": ""}, {"description": "   "}, {"description": 42}],
        implicit_traits=[{"description": None}],
    )
    assert summary == "(no summary)"


# ==========================================================================
# Empty-memcells guard
# ==========================================================================


async def test_aextract_raises_on_empty_memcells() -> None:
    """Passing an empty sequence raises ValueError before any LLM call."""
    fake = FakeLLMClient(handler=lambda *_a, **_kw: ChatResponse(content="{}", model="fake"))
    with pytest.raises(ValueError, match="at least one"):
        await ProfileExtractor(llm=fake).aextract([], sender_id="u_alice")


# ==========================================================================
# Silent-skip contract — agent → user-memory pipeline
# ==========================================================================


async def test_aextract_silently_skips_non_chat_items() -> None:
    """ProfileExtractor must silently skip ToolCallRequest / ToolCallResult items.

    Locks the agent → user-memory pipeline contract: a sequence of MemCells with mixed items
    must render only the ChatMessage text into the profile prompt.
    """
    payload = json.dumps(
        {
            "explicit_info": [{"category": "Skills", "description": "Alice writes Python.", "evidence": "x"}],
            "implicit_traits": [],
        }
    )
    captured: dict[str, str] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["prompt"] = messages[0].content
        return ChatResponse(content=payload, model="fake")

    fake = FakeLLMClient(handler=handler)

    mixed_cell = MemCell(
        items=[
            ChatMessage(
                id="c1",
                role="user",
                content="I write Python for data pipelines.",
                timestamp=1700000000000,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ToolCallRequest(
                tool_calls=[
                    ToolCall(id="tc1", function=ToolCallFunction(name="search.docs", arguments='{"q": "python"}'))
                ],
                timestamp=1700000001000,
                sender_id="assistant",
            ),
            ToolCallResult(
                tool_call_id="tc1",
                content="Python docs returned.",
                timestamp=1700000002000,
            ),
        ],
        timestamp=1700000002000,
    )

    profile = await ProfileExtractor(llm=fake).aextract([mixed_cell], sender_id="u_alice")

    assert profile.owner_id == "u_alice"
    # ChatMessage content must be present in the rendered prompt
    assert "I write Python for data pipelines" in captured["prompt"]
    # Tool call content must NOT appear — ToolCallRequest / ToolCallResult were silently skipped
    assert "search.docs" not in captured["prompt"]
    assert "Python docs returned" not in captured["prompt"]


# ==========================================================================
# UPDATE mode
# ==========================================================================


def _old_profile(
    explicit_info: list[dict[str, Any]] | None = None,
    implicit_traits: list[dict[str, Any]] | None = None,
) -> Profile:
    ei: list[dict[str, Any]] = explicit_info or [
        {"category": "Skills", "description": "Alice writes Python.", "evidence": "x"}
    ]
    it: list[dict[str, Any]] = implicit_traits or [
        {"trait": "[Pragmatic]", "description": "Minimal ceremony.", "evidence": "y"}
    ]
    return Profile.model_validate(
        {
            "owner_id": "u_alice",
            "summary": "Alice writes Python.",
            "timestamp": _TS_OLD_1,
            "explicit_info": ei,
            "implicit_traits": it,
        }
    )


async def test_update_no_compact_applies_ops_and_returns_merged_profile() -> None:
    """UPDATE mode: ops applied, single LLM call, item count stays below compact threshold."""
    ops_json = json.dumps(
        {
            "operations": [
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": {"category": "Location", "description": "Lives in Berlin.", "evidence": "z"},
                },
            ],
            "update_note": "added 1",
        }
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=ops_json, model="fake")])

    profile = await ProfileExtractor(llm=fake).aextract(
        [_memcell()],
        sender_id="u_alice",
        old_profile=_old_profile(),
    )

    assert fake.call_count == 1
    assert profile.owner_id == "u_alice"
    ei: list[dict[str, Any]] = profile.explicit_info  # type: ignore[attr-defined]
    assert len(ei) == 2
    assert ei[1]["category"] == "Location"
    assert ei[1]["description"] == "Lives in Berlin."


async def test_update_triggers_compact_when_item_count_exceeds_threshold() -> None:
    """UPDATE mode: post-merge item count > _PROFILE_COMPACT_THRESHOLD triggers second LLM compact call."""
    # Build an old profile with enough items to push total over the threshold after one add.
    n = _PROFILE_COMPACT_THRESHOLD  # one add brings total to threshold + 1
    ei_items = [{"category": f"Cat{i}", "description": f"Desc{i}.", "evidence": "x"} for i in range(n)]
    old_p = _old_profile(explicit_info=ei_items, implicit_traits=[])

    ops_json = json.dumps(
        {
            "operations": [
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": {"category": "New", "description": "One more.", "evidence": "z"},
                }
            ],
            "update_note": "added 1",
        }
    )
    compact_json = json.dumps(
        {
            "explicit_info": [{"category": "Merged", "description": "Compacted profile.", "evidence": "all"}],
            "implicit_traits": [],
            "compact_note": "merged many into one",
        }
    )
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content=ops_json, model="fake"),
            ChatResponse(content=compact_json, model="fake"),
        ]
    )

    profile = await ProfileExtractor(llm=fake).aextract(
        [_memcell()],
        sender_id="u_alice",
        old_profile=old_p,
    )

    assert fake.call_count == 2  # update call + compact call
    ei: list[dict[str, Any]] = profile.explicit_info  # type: ignore[attr-defined]
    assert len(ei) == 1
    assert ei[0]["category"] == "Merged"


async def test_update_merge_correctness_add_update_delete() -> None:
    """UPDATE mode: add / update / delete ops each produce the correct merged result."""
    old_p = _old_profile(
        explicit_info=[
            {"category": "Skills", "description": "Writes Python.", "evidence": "x"},
            {"category": "Location", "description": "In Berlin.", "evidence": "y"},
        ],
        implicit_traits=[
            {"trait": "[Pragmatic]", "description": "Minimal ceremony.", "evidence": "z"},
        ],
    )
    ops_json = json.dumps(
        {
            "operations": [
                {
                    "action": "add",
                    "type": "explicit_info",
                    "data": {"category": "Hobby", "description": "Plays chess.", "evidence": "new"},
                },
                {
                    "action": "update",
                    "type": "explicit_info",
                    "index": 0,
                    "data": {"description": "Writes Python and Go."},
                },
                {"action": "delete", "type": "implicit_traits", "index": 0},
            ],
            "update_note": "add, update, delete",
        }
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=ops_json, model="fake")])

    profile = await ProfileExtractor(llm=fake).aextract(
        [_memcell()],
        sender_id="u_alice",
        old_profile=old_p,
    )

    ei: list[dict[str, Any]] = profile.explicit_info  # type: ignore[attr-defined]
    it: list[dict[str, Any]] = profile.implicit_traits  # type: ignore[attr-defined]
    # add: 3rd explicit_info item added
    assert len(ei) == 3
    assert ei[2]["category"] == "Hobby"
    # update: first item's description overwritten
    assert ei[0]["description"] == "Writes Python and Go."
    # existing field preserved by update merge
    assert ei[0]["category"] == "Skills"
    # Location (index 1) untouched
    assert ei[1]["category"] == "Location"
    # delete: implicit_traits[0] removed
    assert len(it) == 0
