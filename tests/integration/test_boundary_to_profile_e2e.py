"""End-to-end pipeline test: messages → boundary → profile (with prior cluster).

LLM JSON follows ``PROFILE_INITIAL_EXTRACTION_PROMPT`` schema:
``{"explicit_info": [...], "implicit_traits": [...]}``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from everalgo.testing.assertions import assert_profile_shape
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import BoundaryDetector, to_memcells
from everalgo.user_memory.profile import ProfileExtractor

if TYPE_CHECKING:
    from everalgo.llm import LLMMessage


def _prior_cluster() -> list[MemCell]:
    """Two pre-existing MemCells for the same user as cluster context."""
    return [
        MemCell(
            items=[
                ChatMessage(
                    id="m10",
                    role="user",
                    content="I've been digging into Python async patterns lately.",
                    timestamp=1690000000000,
                    sender_id="u_alice",
                    sender_name="Alice",
                )
            ],
            timestamp=1690000000000,
        ),
        MemCell(
            items=[
                ChatMessage(
                    id="m11",
                    role="user",
                    content="I prefer ruff over black for formatting.",
                    timestamp=1695000000000,
                    sender_id="u_alice",
                    sender_name="Alice",
                )
            ],
            timestamp=1695000000000,
        ),
    ]


async def test_boundary_to_profile_pipeline_e2e() -> None:
    """Boundary detects 1 MemCell, profile synthesises a snapshot via single-call initial extraction.

    Two LLM calls total:
    - Call 1 (boundary detect): batch boundary detection.
    - Call 2 (profile initial extraction): emits ``{explicit_info, implicit_traits}``.

    Verifies:
    1. Cluster MemCells land in the profile ``{conversation_text}`` placeholder.
    2. ``explicit_info`` comes back as typed items on Profile; the reply's ``implicit_traits`` are dropped, since a
       first profile keeps no traits.
    3. ``assert_profile_shape`` passes.
    """
    call_count = 0
    captured_profile_prompt: dict[str, str] = {}

    def handler(messages: list[LLMMessage], **kwargs: Any) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return '{"reasoning": "single coherent topic", "boundaries": []}'
        assert isinstance(messages[0].content, str)  # narrow for test
        captured_profile_prompt["text"] = messages[0].content
        return (
            '{"explicit_info": ['
            '{"category": "Technical Skills",'
            ' "description": "Alice is a Python developer focused on async patterns.",'
            ' "evidence": "Alice asked about async retry semantics."}'
            "],"
            '"implicit_traits": ['
            '{"trait": "[Pragmatic]",'
            ' "description": "Prefers tooling that minimises ceremony.",'
            ' "basis": "Repeated preference for ruff over black."}'
            "]}"
        )

    fake = FakeLLMClient(handler=handler)
    new_msgs = [
        ChatMessage(
            id="m1",
            role="user",
            content="Quick question on async retry semantics in Python.",
            timestamp=1700000010000,
            sender_id="u_alice",
            sender_name="Alice",
        ),
        ChatMessage(
            id="m2",
            role="assistant",
            content="Sure — what's the failure mode you're seeing?",
            timestamp=1700000011000,
            sender_id="assistant",
        ),
    ]

    cuts = await BoundaryDetector(llm=fake).adetect_batch(new_msgs, is_final=True)
    assert new_msgs[cuts.tail_start :] == []
    (mc,) = to_memcells(new_msgs, cuts)

    profile = await ProfileExtractor(llm=fake).aextract([*_prior_cluster(), mc], sender_id="u_alice")

    pf = assert_profile_shape(profile)
    assert pf.owner_id == "u_alice"
    assert "Python" in pf.summary
    assert pf.explicit_info[0].category == "Technical Skills"
    # A first profile keeps no traits (``ProfileExtractor(new_traits=False)``, the default): the reply's one trait is dropped.
    assert pf.implicit_traits == []

    # Cluster rendering reached the profile prompt
    prompt_text = captured_profile_prompt["text"]
    assert "Python async patterns" in prompt_text
    assert "ruff over black" in prompt_text
    assert call_count == 2  # 1 boundary + 1 profile
