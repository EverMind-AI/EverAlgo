"""BoundaryDetector — chat messages → MemCell segmentation.

Demonstrates how a sequence of ``ChatMessage`` objects flows through
``BoundaryDetector.adetect(...)`` to produce one or more ``MemCell``
boundary-detection outputs.  Uses ``FakeLLMClient`` so no API key is needed.

Run:
    uv run python examples/01_boundary_chat.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, ConversationItem
from everalgo.user_memory import BoundaryDetector

# ---------------------------------------------------------------------------
# Scripted LLM response — mirrors the JSON shape BoundaryDetector parses.
# No boundaries → all messages land in a single MemCell.
# ---------------------------------------------------------------------------

_BOUNDARY_JSON = json.dumps(
    {
        "reasoning": "Three messages form a single coherent topic about Python async.",
        "boundaries": [],
        "should_wait": False,
    }
)


def _make_fake() -> FakeLLMClient:
    """Single scripted response: LLM says the 3 messages form one continuous topic."""

    def handler(messages: list[LLMChatMessage], **_kwargs: Any) -> ChatResponse:
        return ChatResponse(content=_BOUNDARY_JSON, model="fake")

    return FakeLLMClient(handler=handler)


def _make_messages() -> list[ChatMessage]:
    """Three-turn chat: user asks about Python async, assistant replies, user follows up."""
    return [
        ChatMessage(
            id="m1",
            role="user",
            content="I want to learn Python async/await — where do I start?",
            timestamp=1_700_000_000_000,
            sender_id="u_alice",
            sender_name="Alice",
        ),
        ChatMessage(
            id="m2",
            role="assistant",
            content="Start with `asyncio.run()` and `await`-able coroutines, then explore `asyncio.gather`.",
            timestamp=1_700_000_001_000,
            sender_id="assistant",
        ),
        ChatMessage(
            id="m3",
            role="user",
            content="What about error handling inside coroutines?",
            timestamp=1_700_000_002_000,
            sender_id="u_alice",
            sender_name="Alice",
        ),
    ]


async def main() -> None:
    """Run boundary detection on 3 chat messages and print resulting MemCell info."""
    fake = _make_fake()
    messages = _make_messages()

    result = await BoundaryDetector(llm=fake).adetect(messages, is_final=True)

    print(f"cells produced : {len(result.cells)}")
    print(f"tail (held back): {len(result.tail)} messages")
    # The LLM's verdict on the tail, not a restatement of its length: True means the trailing segment
    # is too thin to place in an episode yet. None means no path judged it.
    print(f"should_wait    : {result.should_wait}")

    if result.cells:
        first_cell = result.cells[0]
        print(f"first cell timestamp : {first_cell.timestamp}")
        print(f"first cell item count: {len(first_cell.items)}")
        item: ConversationItem
        for i, item in enumerate(first_cell.items):
            if isinstance(item, ChatMessage):
                print(f"  item[{i}] role={item.role!r}  content={item.content[:60]!r}")


if __name__ == "__main__":
    asyncio.run(main())
