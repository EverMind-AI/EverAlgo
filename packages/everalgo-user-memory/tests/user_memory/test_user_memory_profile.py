"""Tests for everalgo.user_memory.profile — ProfileExtractor (new-release single-call).

The new release replaces the prior 2-stage ``CONVERSATION_PROFILE_PART1 + PART2`` flow with a single
LLM call against ``PROFILE_INITIAL_EXTRACTION_PROMPT`` returning ``{explicit_info, implicit_traits}``.
"""

from __future__ import annotations

from typing import Any

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import MemCell, Message, MessageRole
from everalgo.user_memory.profile import (
    ProfileExtractor,
    _build_summary,
    _derive_owner_id,
    _parse_profile_payload,
    _render_conversation,
)


def _memcell(idx: str = "mc_now", ts: int = 1700000010000, content: str = "Default content") -> MemCell:
    msg = Message(
        role=MessageRole.USER,
        content=content,
        timestamp=ts,
        sender_id="u_alice",
        sender_name="Alice",
    )
    return MemCell(
        event_id=idx,
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=ts,
        participants=["u_alice"],
        sender_ids=["u_alice"],
    )


def _cluster() -> list[MemCell]:
    return [
        _memcell(idx="mc_old_1", ts=1690000000000, content="User asked about Python async patterns."),
        _memcell(idx="mc_old_2", ts=1695000000000, content="User mentioned they prefer ruff over black."),
    ]


def _payload(explicit_info: list[dict[str, Any]], implicit_traits: list[dict[str, Any]]) -> str:
    import json

    return json.dumps({"explicit_info": explicit_info, "implicit_traits": implicit_traits})


async def test_aextract_builds_profile_from_explicit_info() -> None:
    """``summary`` derives from first explicit_info description; lists preserved as extras."""
    payload = _payload(
        explicit_info=[
            {
                "category": "Technical Skills",
                "description": "Alice is a Python developer focusing on async patterns.",
                "evidence": "Alice asked about async patterns.",
                "sources": ["2026-05-13 10:00|mc_now"],
            },
        ],
        implicit_traits=[
            {
                "trait": "[Pragmatic]",
                "description": "Prefers tooling that minimises ceremony.",
                "basis": "Repeated preference for ruff over black.",
                "evidence": "Alice mentioned ruff preference.",
                "sources": ["2026-05-13 10:00|mc_old_2", "2026-05-13 10:01|mc_now"],
            },
        ],
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=payload, model="fake")])

    profile = await ProfileExtractor().aextract(_memcell(), cluster_episodes=_cluster(), llm=fake)

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

    profile = await ProfileExtractor().aextract(_memcell(), cluster_episodes=[], llm=fake)

    assert profile.summary == "Prefers minimal-ceremony tooling."


async def test_aextract_returns_fallback_when_payload_missing_required_keys() -> None:
    """Five retries of garbage → fallback Profile with sentinel summary."""
    bad = ChatResponse(content="{}", model="fake")
    fake = FakeLLMClient(responses=[bad, bad, bad, bad, bad])

    profile = await ProfileExtractor().aextract(_memcell(), cluster_episodes=[], llm=fake)

    assert profile.summary == "(no summary)"
    assert profile.owner_id == "u_alice"
    assert fake.call_count == 5


async def test_aextract_returns_fallback_after_5_retries_on_bad_json() -> None:
    bad = ChatResponse(content="not json", model="fake")
    fake = FakeLLMClient(responses=[bad, bad, bad, bad, bad])

    profile = await ProfileExtractor().aextract(_memcell(), cluster_episodes=[], llm=fake)

    assert profile.summary == "(no summary)"
    assert fake.call_count == 5


async def test_aextract_renders_cluster_into_conversation_text() -> None:
    """Cluster MemCells are stitched into the {conversation_text} placeholder."""
    captured: dict[str, str] = {}
    payload = _payload(
        explicit_info=[{"category": "x", "description": "y", "evidence": "z", "sources": ["s1"]}],
        implicit_traits=[],
    )

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["prompt"] = messages[0].content
        return ChatResponse(content=payload, model="fake")

    fake = FakeLLMClient(handler=handler)

    await ProfileExtractor().aextract(_memcell(), cluster_episodes=_cluster(), llm=fake)

    assert "mc_old_1" in captured["prompt"]
    assert "Python async patterns" in captured["prompt"]


async def test_aextract_per_call_prompt_overrides_default() -> None:
    captured: dict[str, str] = {}
    payload = _payload(
        explicit_info=[{"category": "x", "description": "y", "evidence": "z", "sources": ["s1"]}],
        implicit_traits=[],
    )

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        captured["prompt"] = messages[0].content
        return ChatResponse(content=payload, model="fake")

    fake = FakeLLMClient(handler=handler)
    custom = "CUSTOM PROFILE conv={conversation_text}"

    await ProfileExtractor().aextract(_memcell(), cluster_episodes=[], llm=fake, prompt=custom)

    assert captured["prompt"].startswith("CUSTOM PROFILE conv=")
    assert "Default content" in captured["prompt"]


# ==========================================================================
# Defensive type guards (lines 71, 73)
# ==========================================================================


async def test_aextract_coerces_non_list_explicit_info_to_empty() -> None:
    """When LLM returns a non-list ``explicit_info`` (e.g. dict), coerce to [] (line 71)."""
    payload = '{"explicit_info": {"not": "a list"}, "implicit_traits": []}'
    fake = FakeLLMClient(responses=[ChatResponse(content=payload, model="fake")])

    profile = await ProfileExtractor().aextract(_memcell(), cluster_episodes=[], llm=fake)

    assert profile.summary == "(no summary)"  # empty list → sentinel


async def test_aextract_coerces_non_list_implicit_traits_to_empty() -> None:
    """When LLM returns a non-list ``implicit_traits``, coerce to [] (line 73)."""
    payload = (
        '{"explicit_info": [{"category": "x", "description": "y", "evidence": "z", "sources": ["s"]}],'
        ' "implicit_traits": "not a list"}'
    )
    fake = FakeLLMClient(responses=[ChatResponse(content=payload, model="fake")])

    profile = await ProfileExtractor().aextract(_memcell(), cluster_episodes=[], llm=fake)

    assert profile.summary == "y"  # explicit list still used


# ==========================================================================
# _render_conversation helper (lines 106, 113)
# ==========================================================================


def test_render_conversation_skips_empty_content() -> None:
    """Messages with empty content are silently dropped (line 106)."""
    real = Message(role=MessageRole.USER, content="hello", timestamp=1700000000000, sender_name="Alice")
    empty = Message(role=MessageRole.USER, content="", timestamp=1700000001000, sender_name="Bob")
    cell = MemCell(
        event_id="mc_render",
        original_data=[
            {"message": real.model_dump(exclude_none=True)},
            {"message": empty.model_dump(exclude_none=True)},
        ],
        timestamp=1700000001000,
    )
    rendered = _render_conversation(cell, [])
    assert "Alice" in rendered
    assert "Bob" not in rendered


def test_render_conversation_uses_sentinel_when_all_inputs_empty() -> None:
    """Empty cluster + empty messages → sentinel line (line 113)."""
    empty_cell = MemCell(
        event_id="mc_empty",
        original_data=[],  # no messages
        timestamp=1700000000000,
    )
    rendered = _render_conversation(empty_cell, [])
    assert rendered == "(no prior MemCells in the cluster)"


# ==========================================================================
# _parse_profile_payload (line 142)
# ==========================================================================


def test_parse_profile_payload_raises_on_non_object() -> None:
    """Top-level JSON that isn't an object raises ValueError (line 142)."""
    import pytest

    with pytest.raises(ValueError, match="not a JSON object"):
        _parse_profile_payload("[1, 2, 3]")


# ==========================================================================
# _build_summary defensive branches (lines 157, 163, 167)
# ==========================================================================


def test_build_summary_skips_non_dict_items_in_explicit_info() -> None:
    """Non-dict entries inside explicit_info are skipped (line 157)."""
    summary = _build_summary(
        explicit_info=["not a dict", 42, {"description": "real one"}],
        implicit_traits=[],
    )
    assert summary == "real one"


def test_build_summary_skips_non_dict_items_in_implicit_traits() -> None:
    """Non-dict entries inside implicit_traits are skipped (line 163)."""
    summary = _build_summary(
        explicit_info=[],
        implicit_traits=["not a dict", {"description": "trait desc"}],
    )
    assert summary == "trait desc"


def test_build_summary_returns_sentinel_when_no_usable_description() -> None:
    """Empty / whitespace-only / non-string descriptions → sentinel (line 167)."""
    summary = _build_summary(
        explicit_info=[{"description": ""}, {"description": "   "}, {"description": 42}],
        implicit_traits=[{"description": None}],
    )
    assert summary == "(no summary)"


# ==========================================================================
# _derive_owner_id fallbacks (lines 187-190)
# ==========================================================================


def test_derive_owner_id_falls_back_to_message_sender_id() -> None:
    """No participants → first message with sender_id wins (line 188-189)."""
    msg = Message(role=MessageRole.USER, content="x", timestamp=1, sender_id="u_from_msg")
    cell = MemCell(
        event_id="mc_x",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1,
    )
    assert _derive_owner_id(cell) == "u_from_msg"


def test_derive_owner_id_falls_back_to_u_default_when_nothing_identifies_user() -> None:
    """No participants, no sender_id anywhere → ``u_default`` (line 190)."""
    msg = Message(role=MessageRole.USER, content="x", timestamp=1)
    cell = MemCell(
        event_id="mc_x",
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=1,
    )
    assert _derive_owner_id(cell) == "u_default"
