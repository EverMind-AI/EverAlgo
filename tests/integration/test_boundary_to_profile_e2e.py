"""End-to-end pipeline test: messages → boundary → profile (with prior cluster).

Verifies that ProfileExtractor receives a freshly-detected MemCell plus a caller-fetched prior cluster and
emits a single Profile snapshot with the LLM-supplied summary + extra fields preserved via
``ConfigDict(extra='allow')``.

Profile differs from the other EPISODE-path extractors: it is a **user-level aggregate**, so the e2e flow
includes a pre-existing cluster (synthesized in-test) plus a fresh boundary detection.
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

pytestmark = pytest.mark.skip(reason="boundary.chat.adetect stub — full implementation pending")

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
    """Synthesize two pre-existing MemCells for the same user as cluster context."""
    return [
        MemCell(
            id="mc_prior_001",
            messages=[
                Message(
                    role=MessageRole.USER,
                    content="I've been digging into Python async patterns lately.",
                    timestamp=1690000000000,
                )
            ],
            timestamp=1690000000000,
        ),
        MemCell(
            id="mc_prior_002",
            messages=[
                Message(
                    role=MessageRole.USER,
                    content="I prefer ruff over black for formatting.",
                    timestamp=1695000000000,
                )
            ],
            timestamp=1695000000000,
        ),
    ]


async def test_boundary_to_profile_pipeline_e2e() -> None:
    """Boundary detects 1 MemCell, profile synthesizes a snapshot from it + prior cluster.

    Two LLM calls:
    - Call 1 (boundary detect): {"split_at": null}
    - Call 2 (profile extract): profile JSON with extras

    Verifies:
    1. Cluster summaries land in the profile prompt (so the LLM sees prior context).
    2. ``extra="allow"`` preserves LLM-emitted optional fields (interests / communication_style).
    3. ``assert_profile_shape`` passes.
    """
    call_count = 0
    captured_profile_prompt: dict[str, str] = {}

    def handler(messages: list[LLMChatMessage], **kwargs: Any) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ChatResponse(content='{"split_at": null}', model="fake")
        # call 2 — profile
        captured_profile_prompt["content"] = messages[0].content
        return ChatResponse(
            content=(
                '{"id": "pf_alice", "owner_id": "u_alice", '
                '"summary": "Alice is a Python developer who prefers ruff for linting and is comfortable with async patterns.", '
                '"timestamp": 1700000010000, '
                '"interests": ["python", "tooling"], '
                '"communication_style": "concise"}'
            ),
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)
    new_msgs = [
        Message(
            role=MessageRole.USER,
            content="Quick question on async retry semantics in Python.",
            timestamp=1700000010000,
        )
    ]

    memcells, _tail = await ChatMemCellExtractor().adetect(new_msgs, llm=fake, is_final=True)
    assert len(memcells) == 1
    mc = memcells[0]

    profile = await ProfileExtractor().aextract(mc, cluster_episodes=_prior_cluster(), llm=fake)

    pf = assert_profile_shape(profile)
    assert pf.owner_id == "u_alice"
    assert "Python" in pf.summary
    # extra="allow" surfaces optional fields as model attributes
    assert pf.interests == ["python", "tooling"]  # type: ignore[attr-defined]
    assert pf.communication_style == "concise"  # type: ignore[attr-defined]

    # Cluster rendering reached the LLM
    assert "mc_prior_001" in captured_profile_prompt["content"]
    assert "Python async patterns" in captured_profile_prompt["content"]
    assert "ruff over black" in captured_profile_prompt["content"]
    assert call_count == 2
