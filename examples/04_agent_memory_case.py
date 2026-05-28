"""AgentCaseExtractor — agent trajectory → AgentCase (agent memory write path).

Shows an agent trajectory (user message → tool calls → tool results → assistant
reply) wrapped in a ``MemCell`` and processed by ``AgentCaseExtractor.aextract``.

The trajectory has **three** tool rounds: it clears the default
``min_tool_call_rounds`` gate (3) and, sitting at or below the complex-task
threshold (20), runs the LLM filter — which reports an exploration signal — then
the compress step.  ``FakeLLMClient`` supplies both scripted responses (filter
then compress), so exactly two LLM calls are made.

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
# Scripted LLM responses. Three tool rounds clear the min-rounds gate and stay
# under the complex-task threshold, so the filter runs first (exploration
# signal → keep) and then the compress step — two responses total.
# ---------------------------------------------------------------------------

_FILTER_JSON = json.dumps(
    {
        "has_exploration": True,
        "has_user_correction": False,
        "reason": "Multi-step search and verification before answering.",
    }
)

_COMPRESS_JSON = json.dumps(
    {
        "task_intent": "Search for Python async retry libraries and summarise options",
        "approach": "1. Search for retry libraries.\n2. Drill into tenacity.\n3. Verify the async API.",
        "quality_score": 0.82,
        "key_insight": "Use `tenacity` with `AsyncRetrying` for async-native exponential back-off.",
    }
)


def _make_memcell() -> MemCell:
    """Three-round tool trajectory: search → drill-in → verify → final assistant answer."""
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
        ToolCallResult(tool_call_id="c1", content="Found: tenacity, stamina, backoff.", timestamp=1_700_000_000_150),
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
            timestamp=1_700_000_000_200,
            sender_id="assistant",
        ),
        ToolCallResult(
            tool_call_id="c2",
            content="tenacity.AsyncRetrying supports async context manager.",
            timestamp=1_700_000_000_250,
        ),
        ToolCallRequest(
            tool_calls=[
                ToolCall(
                    id="c3",
                    function=ToolCallFunction(name="web.fetch", arguments='{"url": "https://tenacity.readthedocs.io"}'),
                )
            ],
            content="Verifying the AsyncRetrying API against the docs.",
            timestamp=1_700_000_000_300,
            sender_id="assistant",
        ),
        ToolCallResult(
            tool_call_id="c3",
            content="Docs confirm `async for attempt in AsyncRetrying(...)` with exponential back-off.",
            timestamp=1_700_000_000_350,
        ),
        ChatMessage(
            id="a1",
            role="assistant",
            content="The best option is `tenacity` with `AsyncRetrying` for native async support and exponential back-off.",
            timestamp=1_700_000_000_400,
            sender_id="assistant",
        ),
    ]
    return MemCell(items=items, timestamp=1_700_000_000_400)


async def main() -> None:
    """Extract an AgentCase from a three-round tool trajectory and print its fields."""
    # Three tool rounds → min-rounds gate cleared, filter runs (exploration), then compress: two LLM calls.
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content=_FILTER_JSON, model="fake"),
            ChatResponse(content=_COMPRESS_JSON, model="fake"),
        ]
    )
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
