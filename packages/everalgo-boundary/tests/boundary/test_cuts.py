"""The ``Cuts`` contract: what both detection methods promise about their indices, and ``to_memcells``.

Invariants, for an input of ``len(items)`` items: ``ends`` strictly increasing, ``0 < ends[0]``,
``ends[-1] <= len(items)``, ``0 <= tail_start <= len(items)``; indices count tool items too.
"""

from __future__ import annotations

import json
import random
from itertools import product
from typing import TYPE_CHECKING, Any

import pytest

from everalgo.boundary import BoundaryDetector, Cuts, to_memcells
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, ConversationItem, ToolCall, ToolCallFunction, ToolCallRequest, ToolCallResult
from everalgo.utils import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Callable

_TS = 1_700_000_000_000


def _chat(index: int, text: str = "hello") -> ChatMessage:
    return ChatMessage(id=f"m{index}", role="user", content=f"{text} {index}", timestamp=_TS + index, sender_id="u")


def _tool_request(index: int) -> ToolCallRequest:
    return ToolCallRequest(
        tool_calls=[ToolCall(id=f"c{index}", function=ToolCallFunction(name="search", arguments="{}"))],
        timestamp=_TS + index,
        sender_id="assistant",
    )


def _tool_result(index: int) -> ToolCallResult:
    return ToolCallResult(tool_call_id=f"c{index}", content="result", timestamp=_TS + index)


def _layout(pattern: str) -> list[ConversationItem]:
    """``c`` chat, ``q`` tool request, ``r`` tool result; item ``i`` has timestamp ``_TS + i``."""
    makers: dict[str, Callable[[int], ConversationItem]] = {"c": _chat, "q": _tool_request, "r": _tool_result}
    return [makers[kind](index) for index, kind in enumerate(pattern)]


def _assert_invariants(cuts: Cuts, n_items: int) -> None:
    assert all(a < b for a, b in zip(cuts.ends, cuts.ends[1:], strict=False)), cuts
    assert not cuts.ends or (cuts.ends[0] > 0 and cuts.ends[-1] <= n_items), cuts
    assert 0 <= cuts.tail_start <= n_items, cuts


# ── to_memcells ──────────────────────────────────────────────────────────────────────────────────────


def test_no_ends_means_no_cells() -> None:
    assert to_memcells(_layout("ccc"), Cuts(ends=[], tail_start=0)) == []


def test_each_segment_is_the_half_open_slice_up_to_its_end() -> None:
    items = _layout("cccccc")

    cells = to_memcells(items, Cuts(ends=[2, 4], tail_start=4))

    assert [cell.items for cell in cells] == [items[0:2], items[2:4]]


def test_the_last_end_may_be_the_input_length() -> None:
    items = _layout("ccc")

    assert [cell.items for cell in to_memcells(items, Cuts(ends=[1, 3], tail_start=3))] == [items[:1], items[1:]]


def test_a_cell_is_stamped_with_its_last_chat_message_not_its_last_item() -> None:
    items = _layout("qcrcqr")

    (cell,) = to_memcells(items, Cuts(ends=[6], tail_start=6))

    assert cell.items == items
    assert cell.timestamp == _TS + 3


def test_a_bridged_tail_may_start_inside_the_last_segment() -> None:
    items = _layout("cccc")

    assert [cell.items for cell in to_memcells(items, Cuts(ends=[3], tail_start=2))] == [items[:3]]


def test_cells_hold_copies_of_the_segment() -> None:
    items = _layout("cc")

    (cell,) = to_memcells(items, Cuts(ends=[2], tail_start=2))
    cell.items.append(_chat(9))

    assert len(items) == 2


@pytest.mark.parametrize(
    ("cuts", "message"),
    [
        pytest.param(
            Cuts([0, 2], 2), r"^cuts\.ends must be strictly increasing within 1\.\.4, got \[0, 2\]: ", id="zero"
        ),
        pytest.param(Cuts([2, 2], 2), r"got \[2, 2\]: ", id="repeated"),
        pytest.param(Cuts([3, 2], 3), r"got \[3, 2\]: ", id="decreasing"),
        pytest.param(Cuts([5], 4), r"got \[5\]: ", id="past-the-end"),
        pytest.param(Cuts([2], -1), r"^cuts\.tail_start must be within 0\.\.4, got -1: ", id="negative-tail"),
        pytest.param(Cuts([2], 5), r"^cuts\.tail_start must be within 0\.\.4, got 5: ", id="tail-past-the-end"),
    ],
)
def test_cuts_that_break_an_invariant_are_refused(cuts: Cuts, message: str) -> None:
    """A caller's-input error: ``InvalidInputError``, which existing ``except ValueError`` handlers also catch."""
    with pytest.raises(InvalidInputError, match=message) as excinfo:
        to_memcells(_layout("cccc"), cuts)

    assert isinstance(excinfo.value, ValueError)


def test_a_segment_without_a_chat_message_is_refused() -> None:
    with pytest.raises(InvalidInputError, match=r"^segment starting at item 1 holds no ChatMessage, so it has no time"):
        to_memcells(_layout("cqrc"), Cuts(ends=[1, 3], tail_start=3))


# ── Both methods keep the invariants ─────────────────────────────────────────────────────────────────


def _batch_cases() -> list[tuple[str, list[int], bool, int]]:
    """Random layouts and replies; about a third with a message limit small enough to force-split."""
    rng = random.Random(1109)
    cases = []
    for _ in range(200):
        pattern = "".join(rng.choice("cccqr") for _ in range(rng.randint(0, 25)))
        chat_count = pattern.count("c")
        boundaries = [rng.randint(-2, chat_count + 2) for _ in range(rng.randint(0, 5))]
        message_limit = rng.randint(2, 8) if rng.random() < 0.3 else 500
        cases.append((pattern, boundaries, rng.random() < 0.5, message_limit))
    return cases


@pytest.mark.parametrize(("pattern", "boundaries", "is_final", "message_limit"), _batch_cases())
async def test_batch_cuts_keep_the_invariants_and_never_overlap(
    *, pattern: str, boundaries: list[int], is_final: bool, message_limit: int
) -> None:
    items = _layout(pattern)
    fake = FakeLLMClient(handler=lambda *_, **__: json.dumps({"reasoning": "r", "boundaries": boundaries}))

    cuts = await BoundaryDetector(llm=fake).adetect_batch(items, is_final=is_final, hard_message_limit=message_limit)

    _assert_invariants(cuts, len(items))
    assert cuts.tail_start == (cuts.ends[-1] if cuts.ends else 0)
    assert not (is_final and "c" in pattern) or cuts.ends[-1] == len(items)
    cells, tail = to_memcells(items, cuts), items[cuts.tail_start :]
    assert [item for cell in cells for item in cell.items] + tail == items


_STEP_OPTIONS: tuple[dict[str, Any], ...] = (
    {},
    {"smart_mask": False},
    {"warmup_messages": 0, "smart_mask_threshold": 0},
    {"hard_message_limit": 4},
)


def _step_cases() -> list[dict[str, Any]]:
    return [
        {"count": count, "should_end": should_end, **options}
        for count, should_end, options in product((1, 2, 3, 5, 6, 7, 12), (False, True), _STEP_OPTIONS)
    ]


@pytest.mark.parametrize("case", _step_cases(), ids=str)
async def test_step_cuts_keep_the_invariants_and_take_one_of_three_shapes(case: dict[str, Any]) -> None:
    options = dict(case)
    count, should_end = options.pop("count"), options.pop("should_end")
    items = [_chat(index) for index in range(count)]
    n = count - 1
    reply = json.dumps({"should_end": should_end})

    cuts = await BoundaryDetector(llm=FakeLLMClient(handler=lambda *_, **__: reply)).adetect_step(items, **options)

    _assert_invariants(cuts, len(items))
    assert cuts in (Cuts([], 0), Cuts([n], n), Cuts([n], n - 1))
    tail = items[cuts.tail_start :]
    assert tail[-1] == items[-1], "the new message is always carried forward"
    assert [item for cell in to_memcells(items, cuts) for item in cell.items] == items[: (cuts.ends or [0])[-1]]
