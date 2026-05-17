"""Agent trajectory boundary detection facade.

Wraps the low-level ``everalgo.boundary.detect_boundaries`` primitive with a filter → detect → remap
pipeline to handle mixed trajectories (``ChatMessage`` + ``ToolCallRequest`` + ``ToolCallResult``).

Tool items are filtered out before the LLM detection step so the LLM sees a clean conversation
view; boundary indices from the chat-only result are then remapped back to the original
``ConversationItem`` index space so that the output ``MemCell.items`` preserves the full
trajectory (including tool calls).

Uses explicit composition rather than inheritance: filter → detect → remap layers are separate
functions rather than an extended class hierarchy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.boundary import DetectionResult, detect_boundaries
from everalgo.types import ChatMessage, MemCell

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import ConversationItem

logger = logging.getLogger(__name__)

__all__ = ["AgentBoundaryDetector"]


class AgentBoundaryDetector:
    """Boundary detection on an agent trajectory (mixed ``ConversationItem`` list).

    The filter → detect → remap pipeline:

    1. **Filter** — extract only ``ChatMessage`` items and remember their original indices.
    2. **Detect** — run ``everalgo.boundary.detect_boundaries`` on the chat-only view.
    3. **Remap** — translate chat-space cell boundaries back to the original ``ConversationItem``
       index space; ``ToolCallRequest`` / ``ToolCallResult`` items between adjacent chat
       messages are folded into the cell that contains the preceding chat message.

    Invariant: ``sum(len(c.items) for c in result.cells) + len(result.tail) == len(items)``.

    The LLM client is bound to the instance at construction time.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def adetect(
        self,
        items: list[ConversationItem],
        *,
        is_final: bool = False,
        prompt: str | None = None,
    ) -> DetectionResult:
        """Detect boundaries on a mixed agent trajectory.

        Args:
            items: Ordered ``ConversationItem`` list for one trajectory slice — may contain
                ``ChatMessage``, ``ToolCallRequest``, and ``ToolCallResult`` in any mix.
            is_final: When ``True``, the caller guarantees no more items will arrive; trailing tool items
                after the last cell's final chat message are folded into that cell rather than the tail.
                When ``False`` (default), trailing tool items go to the tail.
            prompt: Per-call boundary detection prompt override; ``None`` uses the built-in default.

        Returns:
            ``cells`` — remapped ``MemCell`` list preserving tool calls inside each cell.
            ``tail`` — remainder items (not yet committed to a cell) for the non-final case.
        """
        chat_items: list[ChatMessage] = []
        chat_to_orig: list[int] = []
        for orig_idx, item in enumerate(items):
            if isinstance(item, ChatMessage):
                chat_items.append(item)
                chat_to_orig.append(orig_idx)

        if not chat_items:
            # No chat messages → no LLM signal; everything goes to tail per should_wait semantics.
            logger.debug("adetect: no ChatMessage items in trajectory (n_items=%d)", len(items))
            return DetectionResult(cells=[], tail=list(items))  # type: ignore[arg-type]

        chat_result = await detect_boundaries(chat_items, llm=self._llm, is_final=is_final, prompt=prompt)

        remapped_cells: list[MemCell] = []
        chat_cursor = 0
        for cell in chat_result.cells:
            cell_chat_count = len(cell.items)
            start_chat_idx = chat_cursor
            end_chat_idx = chat_cursor + cell_chat_count - 1
            chat_cursor += cell_chat_count

            start_orig = chat_to_orig[start_chat_idx]

            if chat_cursor < len(chat_to_orig):
                # More cells follow: include everything up to (exclusive) the next cell's first chat.
                end_orig_exclusive = chat_to_orig[chat_cursor]
            elif is_final:
                # Last cell, caller says stream is done: absorb all remaining items.
                end_orig_exclusive = len(items)
            else:
                # Last cell, stream is ongoing: include only up to (inclusive) the last chat in
                # this cell; trailing tool items go to the tail so they can be re-processed.
                end_orig_exclusive = chat_to_orig[end_chat_idx] + 1

            remapped_cells.append(MemCell(items=list(items[start_orig:end_orig_exclusive]), timestamp=cell.timestamp))

        # Tail: items after the last consumed original index.
        if remapped_cells:
            if is_final:
                last_consumed_exclusive = len(items)
            elif chat_cursor < len(chat_to_orig):
                last_consumed_exclusive = chat_to_orig[chat_cursor]
            else:
                # chat_cursor == len(chat_to_orig): all chats consumed; last cell boundary already
                # set to chat_to_orig[end_chat_idx] + 1, so tail starts right after.
                last_consumed_exclusive = chat_to_orig[-1] + 1
            tail = list(items[last_consumed_exclusive:])
        else:
            # detect_boundaries returned no cells — everything is tail.
            tail = list(items)

        return DetectionResult(cells=remapped_cells, tail=tail)  # type: ignore[arg-type]

    detect = async_to_sync(adetect)
    """Sync bridge — only callable from non-event-loop contexts."""
