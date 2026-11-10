"""What both detection paths return: where the caller's input is cut, as indices, and the cells those cuts make.

A cut is the position of the first item of a new segment, which is also where the segment before it ends: segment
``i`` is the half-open slice ``items[ends[i-1]:ends[i]]`` (the first one starts at 0), so every index slices the
caller's list as it is, with no off-by-one adjustment.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

from everalgo.types import ChatMessage, MemCell
from everalgo.utils import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.types import ConversationItem


_PASS_THE_RETURNED_CUTS = "pass the Cuts a detection method returned for these same items"


class Cuts(NamedTuple):
    """Where to cut the input a detection method was given, and where its tail starts.

    Indices count every item of that input, tool items included. ``ends`` is strictly increasing with
    ``0 < ends[0]`` and ``ends[-1] <= len(items)``, so every segment holds at least one item, and
    ``0 <= tail_start <= len(items)``::

        cuts = await detector.adetect_batch(items)
        cells = to_memcells(items, cuts)  # persist
        tail = items[cuts.tail_start :]  # carry into the next call

    Attributes:
        ends: The end (exclusive) of each closed segment: segment ``i`` is ``items[ends[i-1]:ends[i]]``, the first
            starting at 0. Empty when nothing was closed.
        tail_start: Where the part left for the next call starts: ``items[tail_start:]``. It may lie before
            ``ends[-1]``, in which case the items between them are both closed and carried forward.
    """

    ends: list[int]
    tail_start: int


def to_memcells(items: Sequence[ConversationItem], cuts: Cuts) -> list[MemCell]:
    """Cut ``items`` into one :class:`~everalgo.types.MemCell` per segment of ``cuts.ends``.

    Each cell holds its segment's items in order and the timestamp of its last ``ChatMessage``; tool items after
    that message do not move it. The tail is not a cell: slice it with ``items[cuts.tail_start:]``.

    Args:
        items: The input the detection method was given.
        cuts: What the method returned for ``items``.

    Returns:
        The closed cells, in order; empty when ``cuts.ends`` is.

    Raises:
        InvalidInputError: If ``cuts`` breaks an invariant for ``len(items)``, or a segment holds no
            ``ChatMessage``: ``cuts`` was not returned for these ``items``. A ``ValueError`` subclass.
    """
    _check_invariants(cuts, len(items))
    return [_memcell(items[start:end], start) for start, end in pairwise([0, *cuts.ends])]


def _check_invariants(cuts: Cuts, n_items: int) -> None:
    """Raise if ``cuts`` cannot describe a list of ``n_items`` items.

    Raises:
        InvalidInputError: If ``ends`` is not strictly increasing within ``1..n_items`` or ``tail_start`` is
            outside ``0..n_items``.
    """
    in_range = all(1 <= end <= n_items for end in cuts.ends)
    increasing = all(a < b for a, b in pairwise(cuts.ends))
    if not (in_range and increasing):
        raise InvalidInputError(
            f"cuts.ends must be strictly increasing within 1..{n_items}, got {cuts.ends}: {_PASS_THE_RETURNED_CUTS}"
        )
    if not 0 <= cuts.tail_start <= n_items:
        raise InvalidInputError(
            f"cuts.tail_start must be within 0..{n_items}, got {cuts.tail_start}: {_PASS_THE_RETURNED_CUTS}"
        )


def _memcell(segment: Sequence[ConversationItem], start: int) -> MemCell:
    """A cell over a copy of ``segment``, stamped with its last chat message's time.

    Raises:
        InvalidInputError: If ``segment`` holds no ``ChatMessage``.
    """
    last_chat = next((item for item in reversed(segment) if isinstance(item, ChatMessage)), None)
    if last_chat is None:
        raise InvalidInputError(
            f"segment starting at item {start} holds no ChatMessage, so it has no time to stamp: "
            f"{_PASS_THE_RETURNED_CUTS}"
        )
    return MemCell(items=list(segment), timestamp=last_chat.timestamp)
