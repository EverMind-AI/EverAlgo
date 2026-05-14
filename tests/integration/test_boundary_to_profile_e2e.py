"""End-to-end pipeline test: messages → boundary → profile (with prior cluster).

LLM JSON follows new-release ``PROFILE_INITIAL_EXTRACTION_PROMPT`` schema:
``{"explicit_info": [...], "implicit_traits": [...]}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import everalgo.llm
from everalgo.boundary.chat import ChatMemCellExtractor
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.assertions import assert_profile_shape
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import MemCell, Message, MessageRole
from everalgo.user_memory.profile import ProfileExtractor

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def reset_everalgo_llm_state() -> Iterator[None]:
    """Reset everalgo.llm._default + _active per test."""
    saved_default = everalgo.llm._default
    token = everalgo.llm._active.set(None)
    try:
        everalgo.llm._default = None
        yield
    finally:
        everalgo.llm._default = saved_default
        everalgo.llm._active.reset(token)


def _prior_cluster() -> list[MemCell]:
    """Two pre-existing MemCells for the same user as cluster context."""
    msg1 = Message(
        role=MessageRole.USER,
        content="I've been digging into Python async patterns lately.",
        timestamp=1690000000000,
        sender_id="u_alice",
        sender_name="Alice",
    )
    msg2 = Message(
        role=MessageRole.USER,
        content="I prefer ruff over black for formatting.",
        timestamp=1695000000000,
        sender_id="u_alice",
        sender_name="Alice",
    )
    return [
        MemCell(
            event_id="mc_prior_001",
            original_data=[{"message": msg1.model_dump(exclude_none=True)}],
            timestamp=1690000000000,
            participants=["u_alice"],
            sender_ids=["u_alice"],
        ),
        MemCell(
            event_id="mc_prior_002",
            original_data=[{"message": msg2.model_dump(exclude_none=True)}],
            timestamp=1695000000000,
            participants=["u_alice"],
            sender_ids=["u_alice"],
        ),
    ]


async def test_boundary_to_profile_pipeline_e2e() -> None:
    """Boundary detects 1 MemCell, profile synthesises a snapshot via single-call initial extraction.

    Two LLM calls total:
    - Call 1 (boundary detect): new-release batch contract.
    - Call 2 (profile initial extraction): emits ``{explicit_info, implicit_traits}``.

    Verifies:
    1. Cluster MemCells land in the profile ``{conversation_text}`` placeholder.
    2. ``extra="allow"`` preserves explicit_info + implicit_traits as first-class attrs on Profile.
    3. ``assert_profile_shape`` passes.
    """
    call_count = 0
    captured_profile_prompt: dict[str, str] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ChatResponse(
                content='{"reasoning": "single coherent topic", "boundaries": [], "should_wait": false}',
                model="fake",
            )
        captured_profile_prompt["text"] = messages[0].content
        return ChatResponse(
            content=(
                '{"explicit_info": ['
                '{"category": "Technical Skills",'
                ' "description": "Alice is a Python developer focused on async patterns.",'
                ' "evidence": "Alice asked about async retry semantics.",'
                ' "sources": ["2024-03-10 10:00|mc_new"]}'
                "],"
                '"implicit_traits": ['
                '{"trait": "[Pragmatic]",'
                ' "description": "Prefers tooling that minimises ceremony.",'
                ' "basis": "Repeated preference for ruff over black.",'
                ' "evidence": "Mentioned ruff preference in prior conversation.",'
                ' "sources": ["2024-03-08 09:00|mc_prior_002", "2024-03-10 10:00|mc_new"]}'
                "]}"
            ),
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)
    new_msgs = [
        Message(
            role=MessageRole.USER,
            content="Quick question on async retry semantics in Python.",
            timestamp=1700000010000,
            sender_id="u_alice",
            sender_name="Alice",
        ),
        Message(
            role=MessageRole.ASSISTANT,
            content="Sure — what's the failure mode you're seeing?",
            timestamp=1700000011000,
        ),
    ]

    output = await ChatMemCellExtractor().adetect(new_msgs, llm=fake, is_final=True)
    assert output.tail == []
    assert len(output.cells) == 1
    mc = output.cells[0]

    profile = await ProfileExtractor().aextract(mc, cluster_episodes=_prior_cluster(), llm=fake)

    pf = assert_profile_shape(profile)
    assert pf.owner_id == "u_alice"
    assert "Python" in pf.summary
    # explicit_info / implicit_traits preserved via extra="allow"
    assert pf.explicit_info[0]["category"] == "Technical Skills"  # type: ignore[attr-defined]
    assert pf.implicit_traits[0]["trait"] == "[Pragmatic]"  # type: ignore[attr-defined]

    # Cluster rendering reached the profile prompt
    prompt_text = captured_profile_prompt["text"]
    assert "mc_prior_001" in prompt_text
    assert "Python async patterns" in prompt_text
    assert "ruff over black" in prompt_text
    assert call_count == 2  # 1 boundary + 1 profile
