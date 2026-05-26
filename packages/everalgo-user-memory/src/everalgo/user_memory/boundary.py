"""User-scenario boundary detection facade.

Thin wrapper around the low-level ``everalgo.boundary.detect_boundaries`` primitive. Accepts
``list[ChatMessage]`` (user-scenario path; agent trajectories with tool calls go through
``everalgo.agent_memory.AgentBoundaryDetector`` instead).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

from asgiref.sync import async_to_sync

from everalgo._tokenize import count_tokens
from everalgo.boundary import DetectionResult, adetect_boundary_step, detect_boundaries
from everalgo.types import ConversationItem, MemCell

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import ChatMessage


logger = logging.getLogger(__name__)


class BoundaryDetector:
    """Boundary detection on a list of ``ChatMessage``. Native-async with sync bridge.

    Two entry points share the same ``DetectionResult`` return shape:

    - :meth:`adetect` — batch path; forwards to ``everalgo.boundary.detect_boundaries``.
    - :meth:`adetect_step` — single-step incremental path; one LLM call plus the
      cut-and-bridge state transition for the caller.

    The LLM client is bound to the instance at construction time.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def adetect(
        self,
        messages: list[ChatMessage],
        *,
        is_final: bool = False,
        prompt: str | None = None,
    ) -> DetectionResult:
        """Detect conversation boundaries in a list of ``ChatMessage``.

        Args:
            messages: Ordered list of chat messages to split into :class:`~everalgo.types.MemCell` slices.
            is_final: When ``True``, treat the message stream as complete — the tail is flushed into the
                last cell rather than held back as a pending partial window.
            prompt: Optional prompt template override passed through to ``detect_boundaries``.

        Returns:
            Named tuple ``(cells, tail)`` — ``cells`` contains completed :class:`~everalgo.types.MemCell`
            slices; ``tail`` carries any unconfirmed trailing messages.
        """
        return await detect_boundaries(messages, llm=self._llm, is_final=is_final, prompt=prompt)

    detect = async_to_sync(adetect)
    """Sync bridge — only callable from non-event-loop contexts."""

    async def adetect_step(
        self,
        history: list[ChatMessage],
        new: ChatMessage,
        *,
        smart_mask: bool = True,
        smart_mask_threshold: int = 5,
        hard_token_limit: int = 8_192,
        hard_message_limit: int = 50,
        prompt: str | None = None,
    ) -> DetectionResult:
        """Single-step incremental boundary detection — one LLM call + state transition.

        The caller owns only
        the outer loop (front-buffer + final-tail flush); this method encapsulates the
        smart-mask threshold check, masking, force-split short-circuit, LLM call,
        topic_summary fallback, and the cut-and-bridge transition.

        Args:
            history: Prior accumulated messages.
            new: The single newly-arrived message whose boundary status is judged against
                ``history``.
            smart_mask: When ``True`` (default), enables the smart-mask path. Mask is
                only actually applied when ``len(history) > smart_mask_threshold``;
                otherwise the LLM sees the full history regardless.
            smart_mask_threshold: Smart-mask only fires when ``len(history)`` exceeds
                this value. Default is ``5``.
            hard_token_limit: Force-split when ``count_tokens(history+new) >= limit`` and
                ``len(history) >= 2``. Default 8192 matches 93 ``DEFAULT_HARD_TOKEN_LIMIT``.
            hard_message_limit: Force-split when ``len(history)+1 >= limit`` and
                ``len(history) >= 2``. Default 50 matches 93 ``DEFAULT_HARD_MESSAGE_LIMIT``.
            prompt: Prompt override; ``None`` uses ``STEP_BOUNDARY_DETECT_PROMPT_EN``.

        Returns:
            :class:`DetectionResult`:

            - ``cells`` — exactly one closed :class:`MemCell` when ``should_end`` fires
              (built from ``history`` only); empty when ``not should_end``.
            - ``tail`` — the new history the caller must feed into the next call:

              * ``should_end`` with smart-mask active: ``[history[-1], new]`` (bridge).
              * ``should_end`` without smart-mask: ``[new]`` (clean cut).
              * ``not should_end``: ``[*history, new]`` (accumulate).
        """
        # Smart-mask threshold gating — computed up-front so force-split also bridges.
        smart_mask_flag = smart_mask and len(history) > smart_mask_threshold

        # Phase 1: force-split
        history_tokens = _count_history_tokens(history)
        new_tokens = _count_history_tokens([new])
        total_tokens = history_tokens + new_tokens
        total_messages = len(history) + 1
        needs_force_split = total_tokens >= hard_token_limit or total_messages >= hard_message_limit
        if needs_force_split and len(history) >= 2:
            new_history = [history[-1], new] if smart_mask_flag else [new]
            return DetectionResult(cells=[_make_step_cell(history)], tail=new_history)
        # else: needs_force_split but history too short → fall through to LLM

        # Phase 2: LLM decision
        masked_history = history[:-1] if smart_mask_flag else history
        time_gap = _calculate_time_gap(masked_history, [new])
        decision = await adetect_boundary_step(
            masked_history,
            [new],
            llm=self._llm,
            prompt=prompt,
            time_gap_info=time_gap,
        )

        if decision.should_end:
            if not decision.topic_summary:
                logger.warning("BoundaryDetector: should_end=True but topic_summary is empty; using fallback")
            cell = _make_step_cell(history)
            new_history = [history[-1], new] if smart_mask_flag else [new]
            return DetectionResult(cells=[cell], tail=new_history)

        # not should_end: caller keeps accumulating.
        return DetectionResult(cells=[], tail=[*history, new])

    detect_step = async_to_sync(adetect_step)
    """Sync bridge — only callable from non-event-loop contexts."""


# ---------------------------------------------------------------------------
# Module-level helpers (private, used by BoundaryDetector.adetect_step).
# ---------------------------------------------------------------------------


def _count_history_tokens(messages: list[ChatMessage]) -> int:
    """Per-message encode + sum (counts tokens line by line, not joined)."""
    total = 0
    for m in messages:
        speaker = m.sender_name or m.sender_id
        content = (
            m.content
            if isinstance(m.content, str)
            else " ".join(block.text for block in m.content if hasattr(block, "text"))
        )
        text = f"{speaker}: {content}" if speaker else content
        total += count_tokens(text)
    return total


def _make_step_cell(slice_msgs: list[ChatMessage]) -> MemCell:
    """Build a MemCell from a non-empty message slice (incremental-step path)."""
    return MemCell(items=cast("list[ConversationItem]", slice_msgs), timestamp=slice_msgs[-1].timestamp)


def _calculate_time_gap(
    history: list[ChatMessage],
    new_messages: list[ChatMessage],
) -> str:
    """Bucket the time difference between the last history msg and first new msg.

    Operates on ``ChatMessage.timestamp: int`` (ms-since-epoch).

    Args:
        history: Prior accumulated messages.
        new_messages: Newly arrived messages.

    Returns:
        Bucket label ready to inject into the boundary detection prompt ``{time_gap_info}`` placeholder.
    """
    if not history or not new_messages:
        return "No time gap information available"
    last_ts_ms = history[-1].timestamp
    first_ts_ms = new_messages[0].timestamp
    if last_ts_ms == 0 or first_ts_ms == 0:
        return "No timestamp information available"
    total_seconds = (first_ts_ms - last_ts_ms) // 1000
    if total_seconds < 0:
        return "Time gap: Messages appear to be out of order"
    if total_seconds < 60:
        return f"Time gap: {int(total_seconds)} seconds (immediate response)"
    if total_seconds < 3600:
        return f"Time gap: {int(total_seconds // 60)} minutes (recent conversation)"
    if total_seconds < 86_400:
        return f"Time gap: {int(total_seconds // 3600)} hours (same day, but significant pause)"
    return f"Time gap: {int(total_seconds // 86_400)} days (long gap, likely new conversation)"
