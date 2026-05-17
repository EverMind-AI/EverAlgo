"""AgentCaseExtractor — agent trajectory → AgentCase (agent memory write path).

Shows an agent trajectory (user message → tool call → tool result → assistant
reply) wrapped in a ``MemCell`` and processed by ``AgentCaseExtractor.aextract``.

The trajectory has **two** tool rounds, so the LLM filter step is bypassed and
only one LLM call (the compress step) is made.  ``FakeLLMClient`` supplies the
scripted compress response.

Run:
    uv run python examples/04_agent_memory_case.py
"""

from __future__ import annotations

import asyncio
import json

from everalgo.agent_memory.case import AgentCaseExtractor
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import (
    AgentCase,
    ChatMessage,
    ConversationItem,
    MemCell,
    ToolCall,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)

# ---------------------------------------------------------------------------
# Scripted compress response — two tool rounds bypass the filter, so only
# the compress LLM call is made (1 response total).
# ---------------------------------------------------------------------------

_COMPRESS_JSON = json.dumps(
    {
        "task_intent": "Search for Python async retry libraries and summarise options",
        "approach": "1. Search for retry libraries.\n2. Filter by async support.\n3. Summarise top picks.",
        "quality_score": 0.82,
        "key_insight": "Use `tenacity` with `AsyncRetrying` for async-native exponential back-off.",
    }
)


def _make_memcell() -> MemCell:
    """Two-round tool trajectory: search → filter → final assistant answer."""
    items: list[ConversationItem] = [
        ChatMessage(
            id="u1",
            role="user",
            content="What are the best Python libraries for async retry logic?",
            timestamp=1_700_000_000_000,
            sender_id="user",
        ),
        ToolCallRequest(
            tool_calls=[
                ToolCall(
                    id="c1",
                    function=ToolCallFunction(name="web.search", arguments='{"query": "Python async retry library"}'),
                )
            ],
            content="Let me search for that.",
            timestamp=1_700_000_000_100,
            sender_id="assistant",
        ),
        ToolCallResult(tool_call_id="c1", content="Found: tenacity, stamina, backoff.", timestamp=1_700_000_000_200),
        ToolCallRequest(
            tool_calls=[
                ToolCall(
                    id="c2",
                    function=ToolCallFunction(
                        name="web.search", arguments='{"query": "tenacity AsyncRetrying example"}'
                    ),
                )
            ],
            content="Drilling deeper into tenacity.",
            timestamp=1_700_000_000_300,
            sender_id="assistant",
        ),
        ToolCallResult(
            tool_call_id="c2",
            content="tenacity.AsyncRetrying supports async context manager.",
            timestamp=1_700_000_000_400,
        ),
        ChatMessage(
            id="a1",
            role="assistant",
            content="The best option is `tenacity` with `AsyncRetrying` for native async support and exponential back-off.",
            timestamp=1_700_000_000_500,
            sender_id="assistant",
        ),
    ]
    return MemCell(items=items, timestamp=1_700_000_000_500)


async def main() -> None:
    """Extract an AgentCase from a two-round tool trajectory and print its fields."""
    # Two tool rounds → LLM filter is skipped; exactly one LLM call (compress).
    fake = FakeLLMClient(responses=[ChatResponse(content=_COMPRESS_JSON, model="fake")])
    mc = _make_memcell()

    cases: list[AgentCase] = await AgentCaseExtractor(llm=fake).aextract(mc)

    print(f"AgentCase count  : {len(cases)}")
    if cases:
        case = cases[0]
        print(f"task_intent      : {case.task_intent!r}")
        print(f"approach (first line): {case.approach.splitlines()[0]!r}")
        print(f"quality_score    : {case.quality_score}")
        print(f"key_insight      : {case.key_insight!r}")
        print(f"llm calls made   : {fake.call_count}")


if __name__ == "__main__":
    asyncio.run(main())
