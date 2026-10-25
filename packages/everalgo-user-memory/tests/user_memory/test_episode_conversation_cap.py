"""The episode conversation cap: what a cell larger than the extraction budget turns into.

Boundary detection is supposed to keep a cell under 65_536 tokens, but a backfill, an import or a raised
limit can hand the extractor a larger one. Before the cap, the whole cell was rendered and every backend
rejected the request as ``prompt_too_long`` (2026-10-25: a 756 KB cell, all four fallbacks failing).
"""

from __future__ import annotations

from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory.episode._render import MAX_CONVERSATION_TOKENS, _render_conversation
from everalgo.utils import count_tokens

_WEDNESDAY_14_00_59 = 1_712_757_659_999  # 2024-04-10T14:00:59.999Z


def _cell(*contents: str) -> MemCell:
    messages = [
        ChatMessage(
            id=f"m{index}",
            role="user",
            content=content,
            timestamp=_WEDNESDAY_14_00_59,
            sender_id="u_alice",
            sender_name="Alice",
        )
        for index, content in enumerate(contents)
    ]
    return MemCell(items=list(messages), timestamp=_WEDNESDAY_14_00_59)


def test_a_conversation_that_fits_is_rendered_whole() -> None:
    rendered = _render_conversation(_cell("first", "second"))

    assert "messages omitted" not in rendered
    assert '<conversation count="2">' in rendered
    assert "Alice: first" in rendered
    assert "Alice: second" in rendered


def test_a_conversation_over_the_budget_keeps_its_head_and_tail() -> None:
    # 300 messages of ~2 KB each is far past 48_000 tokens.
    rendered = _render_conversation(_cell(*[f"message {index}: " + "x" * 2000 for index in range(300)]))

    assert "messages omitted" in rendered
    assert "message 0: " in rendered
    assert "message 299: " in rendered
    # The gap sits in the middle: neither the opening nor the closing message is dropped.
    assert rendered.index("message 0: ") < rendered.index("messages omitted") < rendered.index("message 299: ")
    # The rendered conversation is bounded by the budget, plus the wrapper and the marker line.
    assert count_tokens(rendered) <= MAX_CONVERSATION_TOKENS + 200


def test_a_single_message_over_the_budget_is_truncated_in_place() -> None:
    rendered = _render_conversation(_cell("x" * 400_000))

    assert "messages omitted" not in rendered
    assert "[... trimmed " in rendered
    assert count_tokens(rendered) <= MAX_CONVERSATION_TOKENS + 200
