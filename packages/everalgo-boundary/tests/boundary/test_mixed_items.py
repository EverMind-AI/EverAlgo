"""Tests for the batch path over mixed agent trajectories (``ChatMessage`` + ``ToolCallRequest`` + ``ToolCallResult``).

Only chat messages reach the model; the cuts it makes over them are carried back to the full item list, so every
tool item lands in exactly one cell or in the tail. All LLM calls use
:class:`everalgo.testing.fake_llm.FakeLLMClient` — no real network calls.

LLM reply format (``everalgo.boundary._replies.Reply``):
``{"reasoning": <str>, "boundaries": [<0-based indices of first msg in each new cell>]}``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from everalgo.boundary import BoundaryDetector, adetect_boundaries
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, ConversationItem, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult

if TYPE_CHECKING:
    from everalgo.llm import LLMMessage

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


def _boundary_response(boundaries: list[int]) -> str:
    return json.dumps({"reasoning": "test boundary decision", "boundaries": boundaries})


# ── Chat-only input maps to itself ───────────────────────────────────────────────────────────────────


async def test_chat_only_trajectory_closes_into_one_cell() -> None:
    items: list[ConversationItem] = [_chat("user", "hello", idx=0), _chat("assistant", "hi there", idx=1)]
    result = await BoundaryDetector(llm=FakeLLMClient(responses=[_boundary_response([])])).adetect(items, is_final=True)

    assert [cell.items for cell in result.cells] == [items]
    assert result.tail == []


# ── Tool items never reach the model ─────────────────────────────────────────────────────────────────


async def test_tool_items_are_filtered_from_the_llm_prompt() -> None:
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
    captured: list[LLMMessage] = []

    def _capture_handler(messages: list[LLMMessage], **_kwargs: object) -> str:
        captured.extend(messages)
        return _boundary_response([])

    await adetect_boundaries(items, llm=FakeLLMClient(handler=_capture_handler), is_final=True)

    prompt_text = " ".join(str(m.content) for m in captured)
    assert tool_func_name not in prompt_text
    assert tool_result_content not in prompt_text
    assert captured[-1].content == (
        '<messages count="2">\n'
        "[0] [2023-11-14 22:13 UTC (Tuesday)] user: do something\n"
        "[1] [2023-11-14 22:13 UTC (Tuesday)] assistant: done\n"
        "</messages>"
    ), "chat messages are numbered among themselves"


# ── Where tool items land ────────────────────────────────────────────────────────────────────────────


async def test_tool_items_between_chat_messages_stay_with_the_earlier_one() -> None:
    chat_0 = _chat("user", "task A", idx=0)
    tool_req = _tool_req(ts=_TS + 1)
    tool_res = _tool_res("result A", ts=_TS + 2)
    chat_1 = _chat("assistant", "done A", idx=3, ts=_TS + 3)
    chat_2 = _chat("user", "task B", idx=4, ts=_TS + 4)
    items: list[ConversationItem] = [chat_0, tool_req, tool_res, chat_1, chat_2]

    result = await adetect_boundaries(items, llm=FakeLLMClient(responses=[_boundary_response([1])]), is_final=True)

    assert [cell.items for cell in result.cells] == [[chat_0, tool_req, tool_res], [chat_1, chat_2]]
    assert result.tail == []


async def test_tool_items_ahead_of_the_first_chat_message_join_the_first_cell() -> None:
    lead_req, lead_res = _tool_req(ts=_TS), _tool_res(ts=_TS + 1)
    chat_0 = _chat("user", "A", idx=2, ts=_TS + 2)
    chat_1 = _chat("user", "B", idx=3, ts=_TS + 3)
    items: list[ConversationItem] = [lead_req, lead_res, chat_0, chat_1]

    result = await adetect_boundaries(items, llm=FakeLLMClient(responses=[_boundary_response([1])]))

    assert [cell.items for cell in result.cells] == [[lead_req, lead_res, chat_0]]
    assert result.tail == [chat_1]


async def test_trailing_tool_items_join_the_last_cell_when_final() -> None:
    chat_0 = _chat("user", "go", idx=0)
    chat_1 = _chat("assistant", "ok", idx=1, ts=_TS + 1)
    trailing = _tool_req(ts=_TS + 2)
    items: list[ConversationItem] = [chat_0, chat_1, trailing]

    result = await adetect_boundaries(items, llm=FakeLLMClient(responses=[_boundary_response([1])]), is_final=True)

    assert [cell.items for cell in result.cells] == [[chat_0], [chat_1, trailing]]
    assert result.tail == []


async def test_trailing_tool_items_stay_in_the_tail_when_not_final() -> None:
    chat_0 = _chat("user", "go", idx=0)
    chat_1 = _chat("assistant", "ok", idx=1, ts=_TS + 1)
    trailing = _tool_req(ts=_TS + 2)
    items: list[ConversationItem] = [chat_0, chat_1, trailing]

    result = await adetect_boundaries(items, llm=FakeLLMClient(responses=[_boundary_response([1])]))

    assert [cell.items for cell in result.cells] == [[chat_0]]
    assert result.tail == [chat_1, trailing]


async def test_cell_timestamp_is_its_last_chat_message() -> None:
    chat_0 = _chat("user", "go", idx=0, ts=_TS)
    trailing = _tool_res(ts=_TS + 99)
    items: list[ConversationItem] = [chat_0, trailing]

    result = await adetect_boundaries(items, llm=FakeLLMClient(responses=[_boundary_response([])]), is_final=True)

    assert [(cell.items, cell.timestamp) for cell in result.cells] == [(items, _TS)]


# ── Every item lands exactly once ────────────────────────────────────────────────────────────────────


def _layouts() -> dict[str, list[ConversationItem]]:
    return {
        "chat-only": [_chat("user", "hi", idx=0), _chat("assistant", "hello", idx=1), _chat("user", "bye", idx=2)],
        "leading-tools": [_tool_req(), _tool_res(), _chat("user", "A", idx=2), _chat("user", "B", idx=3)],
        "middle-tools": [_chat("user", "A", idx=0), _tool_req(), _tool_res(), _chat("user", "B", idx=3)],
        "trailing-tools": [_chat("user", "A", idx=0), _chat("user", "B", idx=1), _tool_req(), _tool_res()],
        "everywhere": [
            _tool_req(),
            _chat("user", "A", idx=1),
            _tool_res(),
            _chat("user", "B", idx=3),
            _tool_req(),
            _chat("user", "C", idx=5),
            _tool_res(),
        ],
    }


@pytest.mark.parametrize("layout", list(_layouts()))
@pytest.mark.parametrize("boundaries", [[], [1], [1, 2]], ids=str)
@pytest.mark.parametrize("is_final", [False, True], ids=["ongoing", "final"])
async def test_every_item_lands_exactly_once_and_in_order(layout: str, boundaries: list[int], is_final: bool) -> None:  # noqa: FBT001
    """Cells then tail, concatenated, give back ``items`` — nothing dropped, duplicated or reordered."""
    items = _layouts()[layout]

    result = await adetect_boundaries(
        items, llm=FakeLLMClient(responses=[_boundary_response(boundaries)]), is_final=is_final
    )

    assert sum(len(cell.items) for cell in result.cells) + len(result.tail) == len(items)
    assert [item for cell in result.cells for item in cell.items] + result.tail == items


# ── No chat message, no model ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("is_final", [False, True], ids=["ongoing", "final"])
async def test_tool_only_trajectory_reaches_no_llm_and_goes_to_the_tail(is_final: bool) -> None:  # noqa: FBT001
    items: list[ConversationItem] = [_tool_req(), _tool_res()]
    fake = FakeLLMClient(responses=[])

    result = await adetect_boundaries(items, llm=fake, is_final=is_final)

    assert result.cells == []
    assert result.tail == items
    assert fake.call_count == 0
