"""Tests for BoundaryDetector.adetect_step — 93 incremental mirror.

``adetect_step`` returns :class:`DetectionResult` (``cells, tail, should_wait``):

- ``cells`` carries the 0-or-1 closed MemCell.
- ``tail`` carries the new history the caller must feed back into the next call,
  encapsulating the cut-and-bridge (``[history[-1], new]`` under smart-mask,
  ``[new]`` for a clean cut) or the accumulation (``[*history, new]``).
- ``should_wait`` is always ``None`` — this path's prompt never judges it.
"""

from __future__ import annotations

import json
from typing import Any

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import BoundaryDetector


def _msg(idx: int, content: str = "hello world") -> ChatMessage:
    return ChatMessage(
        id=f"m{idx}",
        role="user",
        content=content,
        timestamp=1_700_000_000_000 + idx * 30_000,
        sender_id=f"u{idx}",
        sender_name=None,
    )


def _ok_response(*, should_end: bool = False, topic_summary: str = "") -> str:
    return json.dumps(
        {
            "should_end": should_end,
            "reasoning": "ok",
            "confidence": 1.0,
            "topic_summary": topic_summary,
        }
    )


async def test_force_split_on_msg_limit_no_llm_call() -> None:
    """93 :416-470 — total messages >= hard_message_limit → cut without LLM.

    With smart_mask active (default), force-split tail bridges ``[history[-1], new]``;
    pass ``smart_mask=False`` for a clean cut.
    """
    detector = BoundaryDetector(llm=FakeLLMClient(responses=[]))
    history = [_msg(i) for i in range(60)]
    new = _msg(99)
    result = await detector.adetect_step(history, new, hard_message_limit=50)
    assert len(result.cells) == 1
    assert isinstance(result.cells[0], MemCell)
    assert result.tail == [history[-1], new]

    # smart_mask=False → clean cut, no bridge.
    detector_no_mask = BoundaryDetector(llm=FakeLLMClient(responses=[]))
    result_no_mask = await detector_no_mask.adetect_step(history, new, smart_mask=False, hard_message_limit=50)
    assert result_no_mask.tail == [new]


async def test_force_split_when_history_too_short_falls_through_to_llm() -> None:
    """93 :472-478 — needs_force_split but history < 2 → still call LLM, not force-cut.

    With ``should_end=False`` from the LLM, tail accumulates ``[*history, new]``.
    """
    payload = _ok_response(should_end=False)
    fake = FakeLLMClient(responses=[payload])
    detector = BoundaryDetector(llm=fake)
    # history has 1 huge message, would trip token limit but len(history) < 2
    huge = ChatMessage(
        id="big",
        role="user",
        content="x" * 100_000,
        timestamp=1_700_000_000_000,
        sender_id="u0",
        sender_name=None,
    )
    new = _msg(1)
    result = await detector.adetect_step([huge], new, hard_token_limit=100)
    # Falls through to LLM because len(history) == 1 < 2; LLM says not should_end
    assert result.cells == []
    assert result.tail == [huge, new]
    assert fake.call_count == 1


async def test_smart_mask_excludes_last_history_message_from_prompt() -> None:
    """93 :481-485 — smart-mask sends history[:-1] to LLM, cut bridges history[-1] forward.

    With ``len(history) = 4 > smart_mask_threshold=3`` smart-mask is active.
    On ``should_end`` the tail must be ``[history[-1], new]`` (bridge), not ``[new]``.
    """
    captured: dict[str, Any] = {}

    def handler(messages: list[LLMChatMessage], **_: Any) -> ChatResponse:
        captured["content"] = messages[0].content
        return ChatResponse(
            content=json.dumps(
                {
                    "should_end": True,
                    "reasoning": "topic shift",
                    "confidence": 0.9,
                    "topic_summary": "Discussion",
                }
            ),
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)
    detector = BoundaryDetector(llm=fake)
    history = [_msg(0, "alpha"), _msg(1, "beta"), _msg(2, "gamma"), _msg(3, "delta")]
    new = _msg(4, "epsilon")
    result = await detector.adetect_step(history, new, smart_mask=True, smart_mask_threshold=3)
    # The closed cell covers the full history (smart-mask only masks the prompt, not the cell).
    assert len(result.cells) == 1
    assert len(result.cells[0].items) == 4
    # Smart-mask cut-and-bridge: tail starts with history[-1] ("delta") then the new msg ("epsilon").
    assert result.tail == [history[-1], new]
    # Prompt verification: history[:-1] = [m0,m1,m2]; "delta" is excluded.
    assert "alpha" in captured["content"]
    assert "beta" in captured["content"]
    assert "gamma" in captured["content"]
    assert "delta" not in captured["content"]


async def test_keep_waiting_returns_empty_cells_and_accumulated_tail() -> None:
    """``not should_end`` → no cells emitted; tail accumulates ``[*history, new]``.

    Threshold high enough to disable smart-mask (clean path).
    """
    payload = _ok_response(should_end=False)
    fake = FakeLLMClient(responses=[payload])
    detector = BoundaryDetector(llm=fake)
    history = [_msg(0), _msg(1)]
    new = _msg(2)
    result = await detector.adetect_step(history, new, smart_mask=True, smart_mask_threshold=100)
    assert result.cells == []
    assert result.tail == [*history, new]
    assert fake.call_count == 1


async def test_should_end_without_smart_mask_clean_cut_tail() -> None:
    """``should_end`` without smart-mask → tail is just ``[new]`` (no history bridge).

    ``smart_mask=False`` bypasses the threshold entirely.
    """
    payload = _ok_response(should_end=True, topic_summary="Discussion topic")
    fake = FakeLLMClient(responses=[payload])
    detector = BoundaryDetector(llm=fake)
    history = [_msg(0), _msg(1), _msg(2)]
    new = _msg(3)
    result = await detector.adetect_step(history, new, smart_mask=False)
    assert len(result.cells) == 1
    # Cell carries the full history; bridge happens in tail only.
    assert len(result.cells[0].items) == 3
    assert result.tail == [new]


def test_sync_bridge_exists() -> None:
    """detect_step = async_to_sync(adetect_step) — usable from non-event-loop callers."""
    detector = BoundaryDetector(llm=FakeLLMClient(responses=[]))
    assert hasattr(detector, "detect_step")
    assert callable(detector.detect_step)


# ===========================================================================
# should_wait — this path does not judge it, and says so
# ===========================================================================


async def test_should_wait_is_none_when_accumulating() -> None:
    payload = _ok_response(should_end=False)
    detector = BoundaryDetector(llm=FakeLLMClient(responses=[payload]))

    result = await detector.adetect_step([_msg(0), _msg(1)], _msg(2))

    assert result.should_wait is None


async def test_should_wait_is_none_when_the_episode_closes() -> None:
    payload = _ok_response(should_end=True, topic_summary="Topic")
    detector = BoundaryDetector(llm=FakeLLMClient(responses=[payload]))

    result = await detector.adetect_step([_msg(0), _msg(1)], _msg(2))

    assert len(result.cells) == 1
    assert result.should_wait is None


async def test_should_wait_is_not_the_inverse_of_should_end() -> None:
    """Inverting ``should_end`` would make the common case report "wait" and is not what the field means.

    ``should_end`` answers whether a new episode began; ``should_wait`` answers whether the trailing
    segment carries enough to be placed in one. Both default to false for unrelated reasons, so an
    inversion reports "wait" on nearly every step — the step prompt's own principle is to keep related
    content together — and claims "no need to wait" about a tail this path never evaluated.
    """
    accumulating = await BoundaryDetector(llm=FakeLLMClient(responses=[_ok_response(should_end=False)])).adetect_step(
        [_msg(0), _msg(1)], _msg(2)
    )
    closing = await BoundaryDetector(
        llm=FakeLLMClient(responses=[_ok_response(should_end=True, topic_summary="T")])
    ).adetect_step([_msg(0), _msg(1)], _msg(2))

    assert accumulating.should_wait is None
    assert closing.should_wait is None
