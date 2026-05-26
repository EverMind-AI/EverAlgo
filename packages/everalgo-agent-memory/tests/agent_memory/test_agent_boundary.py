"""Tests for everalgo.agent_memory.boundary — AgentBoundaryDetector.

Locks the filter → detect → remap correctness invariants.  All LLM calls use
:class:`everalgo.testing.fake_llm.FakeLLMClient` — no real network calls.

LLM response format expected by ``detect_boundaries`` (boundary chat.py ``_detect_boundaries``):
``{"boundaries": [<1-based indices of first msg in each new cell>], "should_wait": <bool>}``.
"""

from __future__ import annotations

import json

import pytest

from everalgo.agent_memory.boundary import AgentBoundaryDetector
from everalgo.llm.types import ChatResponse as LLMChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, ConversationItem, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult

# ── Fixture helpers ──────────────────────────────────────────────────────────────────────────────────

_TS = 1700000000000  # base timestamp (ms)


def _chat(role: str, content: str, *, ts: int = _TS, idx: int = 0) -> ChatMessage:
    return ChatMessage(id=f"msg_{idx}", role=role, content=content, timestamp=ts, sender_id=role)  # type: ignore[arg-type]


def _tool_req(*, ts: int = _TS, call_id: str = "call_1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_calls=[ToolCall(id=call_id, function=ToolCallFunction(name="search", arguments="{}"))],
        content=None,
        timestamp=ts,
        sender_id="assistant",
    )


def _tool_res(content: str = "result", *, ts: int = _TS, call_id: str = "call_1") -> ToolCallResult:
    return ToolCallResult(tool_call_id=call_id, content=content, timestamp=ts)


def _boundary_response(
    boundaries: list[int],
    *,
    should_wait: bool = False,
    reasoning: str = "test boundary decision reasoning",
) -> str:
    """Build the JSON string ``detect_boundaries`` expects from the LLM.

    ``reasoning`` is required by ``_BoundaryDetectionLLMResponse`` schema; provides
    a default so existing call sites stay terse.
    """
    return json.dumps({"reasoning": reasoning, "boundaries": boundaries, "should_wait": should_wait})


# ── 1. Chat-only passthrough ─────────────────────────────────────────────────────────────────────────


async def test_adetect_chat_only_passthrough() -> None:
    """Pure ChatMessage trajectory → single cell containing all items; empty tail."""
    items: list[ConversationItem] = [
        _chat("user", "hello", idx=0),
        _chat("assistant", "hi there", idx=1),
    ]
    fake = FakeLLMClient(responses=[_boundary_response([])])
    result = await AgentBoundaryDetector(llm=fake).adetect(items, is_final=True)

    assert len(result.cells) == 1
    assert result.cells[0].items == items
    assert result.tail == []


# ── 2. Tool calls filtered from LLM prompt ──────────────────────────────────────────────────────────


async def test_adetect_filters_tool_calls_from_llm_prompt() -> None:
    """LLM prompt must see only chat content — tool call / result text must not appear."""
    tool_func_name = "my_unique_function_xyz"
    tool_result_content = "unique_tool_result_abc"
    items: list[ConversationItem] = [
        _chat("user", "do something", idx=0),
        ToolCallRequest(
            tool_calls=[ToolCall(id="c1", function=ToolCallFunction(name=tool_func_name, arguments="{}"))],
            content=None,
            timestamp=_TS + 1,
            sender_id="assistant",
        ),
        ToolCallResult(tool_call_id="c1", content=tool_result_content, timestamp=_TS + 2),
        _chat("assistant", "done", idx=3, ts=_TS + 3),
    ]

    captured: list[str] = []

    def _capture_handler(messages: list[ChatMessage], **_kwargs: object) -> LLMChatResponse:
        captured.extend(str(m.content) for m in messages if hasattr(m, "content") and isinstance(m.content, str))
        return LLMChatResponse(content=_boundary_response([]), model="fake")

    fake = FakeLLMClient(handler=_capture_handler)
    await AgentBoundaryDetector(llm=fake).adetect(items, is_final=True)

    assert captured, "no LLM call was made"
    prompt_text = " ".join(captured)
    assert tool_func_name not in prompt_text, "tool function name leaked into LLM prompt"
    assert tool_result_content not in prompt_text, "tool result content leaked into LLM prompt"
    assert "do something" in prompt_text, "chat content should appear in prompt"


# ── 3. Tool calls preserved in output cells ──────────────────────────────────────────────────────────


async def test_adetect_remap_preserves_tool_calls_in_output_cells() -> None:
    """Single-cell result: all 4 items (chat+tool+tool+chat) appear in the output cell."""
    chat_0 = _chat("user", "search for X", idx=0)
    tool_req = _tool_req(ts=_TS + 1)
    tool_res = _tool_res("results", ts=_TS + 2)
    chat_1 = _chat("assistant", "here is X", idx=3, ts=_TS + 3)
    items: list[ConversationItem] = [chat_0, tool_req, tool_res, chat_1]

    fake = FakeLLMClient(responses=[_boundary_response([])])
    result = await AgentBoundaryDetector(llm=fake).adetect(items, is_final=True)

    assert len(result.cells) == 1
    assert result.cells[0].items == items
    assert result.tail == []


# ── 4. Mid-trajectory split correctness (load-bearing remap test) ────────────────────────────────────


async def test_adetect_remap_with_split() -> None:
    """Split after 1st chat: cell[0]=[chat_0, tool_req, tool_res], cell[1]=[chat_1, chat_2]."""
    chat_0 = _chat("user", "task A", idx=0)
    tool_req = _tool_req(ts=_TS + 1)
    tool_res = _tool_res("result A", ts=_TS + 2)
    chat_1 = _chat("assistant", "done A", idx=3, ts=_TS + 3)
    chat_2 = _chat("user", "task B", idx=4, ts=_TS + 4)
    items: list[ConversationItem] = [chat_0, tool_req, tool_res, chat_1, chat_2]

    # boundaries=[1] means: split after chat index 0 (chat_0); second cell starts at chat index 1.
    fake = FakeLLMClient(responses=[_boundary_response([1])])
    result = await AgentBoundaryDetector(llm=fake).adetect(items, is_final=True)

    assert len(result.cells) == 2
    assert result.cells[0].items == [chat_0, tool_req, tool_res]
    assert result.cells[1].items == [chat_1, chat_2]
    assert result.tail == []


# ── 5. No chat items → empty cells, everything in tail ───────────────────────────────────────────────


async def test_adetect_no_chat_items_returns_empty_cells() -> None:
    """Tool-only trajectory has no LLM signal; cells must be empty and tail equals items."""
    items: list[ConversationItem] = [_tool_req(ts=_TS), _tool_res(ts=_TS + 1)]
    fake = FakeLLMClient(responses=[])  # no call expected
    result = await AgentBoundaryDetector(llm=fake).adetect(items)

    assert result.cells == []
    assert list(result.tail) == items


# ── 6. Total item count invariant ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "items,boundary_indices,is_final",
    [
        # Pure chat, no split
        (
            [_chat("user", "hi", idx=0), _chat("assistant", "hello", idx=1)],
            [],
            True,
        ),
        # Mixed, no split, is_final=True
        (
            [
                _chat("user", "go", idx=0),
                _tool_req(ts=_TS + 1),
                _tool_res(ts=_TS + 2),
                _chat("assistant", "done", idx=3, ts=_TS + 3),
            ],
            [],
            True,
        ),
        # Mixed, no split, is_final=False (trailing tool items go to tail)
        (
            [_chat("user", "go", idx=0), _chat("assistant", "ok", idx=1, ts=_TS + 1), _tool_req(ts=_TS + 2)],
            [],
            False,
        ),
        # Split in the middle
        (
            [
                _chat("user", "A", idx=0),
                _tool_req(ts=_TS + 1),
                _tool_res(ts=_TS + 2),
                _chat("assistant", "done", idx=3, ts=_TS + 3),
                _chat("user", "B", idx=4, ts=_TS + 4),
            ],
            [1],
            True,
        ),
        # Tool-only: cells=[], tail=all
        (
            [_tool_req(ts=_TS), _tool_res(ts=_TS + 1)],
            None,  # no LLM response needed
            False,
        ),
    ],
)
async def test_adetect_invariant_total_count_preserved(
    items: list[ConversationItem],
    boundary_indices: list[int] | None,
    is_final: bool,  # noqa: FBT001
) -> None:
    """sum(len(cell.items) for cell in cells) + len(tail) == len(items) for all scenarios."""
    if boundary_indices is None:
        fake = FakeLLMClient(responses=[])  # tool-only → no LLM call
    else:
        fake = FakeLLMClient(responses=[_boundary_response(boundary_indices)])

    result = await AgentBoundaryDetector(llm=fake).adetect(items, is_final=is_final)

    total = sum(len(c.items) for c in result.cells) + len(result.tail)
    assert total == len(items), (
        f"invariant broken: cells={[len(c.items) for c in result.cells]}, tail={len(result.tail)}, "
        f"expected {len(items)}"
    )
