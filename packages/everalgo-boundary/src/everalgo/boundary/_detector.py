"""Boundary detection: where to cut a conversation, in one batch or one message at a time.

:class:`BoundaryDetector` is the only entry, one method per path, and both return :class:`~everalgo.boundary.Cuts`:
indices into the caller's input, which the caller slices itself or hands to :func:`~everalgo.boundary.to_memcells`.

Batch (:meth:`BoundaryDetector.adetect_batch`): keep the chat messages → hard-limit pre-cut → render → one model call
→ carry the cuts back to the full item list. The hard limits count chat messages only, so the filter comes first.
Step (:meth:`BoundaryDetector.adetect_step`): judge the last message against the history before it — warmup,
hard-limit cut, or one model call — and decide where the next history starts.

Every model call sends two messages: a system message with the rules, closed by a caller's
``custom_instructions=`` when given, and a user message with the rendered conversation, which the operator always
builds.
"""

from __future__ import annotations

import logging
from datetime import UTC
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.boundary._cuts import Cuts
from everalgo.boundary._render import _calculate_time_gap, _format_messages_step, _format_messages_with_indices
from everalgo.boundary._replies import _call_llm_for_batch, _call_llm_for_step
from everalgo.boundary._segments import _chat_view, _force_split_cuts, _item_cuts, _needs_force_split
from everalgo.boundary.prompts.en.chat import (
    BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT,
    BATCH_BOUNDARY_DETECT_USER_PROMPT,
    STEP_BOUNDARY_DETECT_SYSTEM_PROMPT,
    STEP_BOUNDARY_DETECT_USER_PROMPT,
    assemble_system_prompt,
)
from everalgo.llm import LLMMessage
from everalgo.types import ChatMessage
from everalgo.utils import InvalidInputError, render_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import tzinfo

    from everalgo.llm.protocols import LLMClient
    from everalgo.types import ConversationItem


logger = logging.getLogger("everalgo.boundary")


class BoundaryDetector:
    """Boundary detection with the LLM client bound once. Native-async with sync bridges.

    - :meth:`adetect_batch` — one window of a conversation, ``ConversationItem`` with tool items mixed in; one
      model call places every cut in it.
    - :meth:`adetect_step` — the judged history plus one new ``ChatMessage``; one decision on whether the new
      message starts a new segment.

    Both return :class:`~everalgo.boundary.Cuts` over the input they were given. Every strategy parameter is a
    keyword with the recommended value as its default.

    Args:
        llm: LLM client every call goes through.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def adetect_batch(
        self,
        items: Sequence[ConversationItem],
        *,
        is_final: bool = False,
        custom_instructions: str | None = None,
        tz: tzinfo = UTC,
        hard_token_limit: int = 65_536,
        hard_message_limit: int = 500,
    ) -> Cuts:
        """Cut one window of a conversation — typically the prior tail plus the new items — in one model call.

        Only the ``ChatMessage`` items reach the model; ``ToolCallRequest`` / ``ToolCallResult`` items keep their
        place. Tool items between two chat messages go with the earlier one, tool items ahead of the first chat
        message with the first segment, and trailing tool items into the tail (or into the last segment when
        ``is_final``). Leading segments are first cut off without the model until the chat messages left are under
        both hard limits; the model then places the cuts in what is left. Nothing overlaps: the tail starts at the
        last cut.

        Args:
            items: Ordered conversation items; tool items may be mixed in.
            is_final: When ``True``, the trailing segment is closed too: ``len(items)`` is the last cut and the tail
                is empty.
            custom_instructions: The caller's instructions, appended verbatim after the bundled rules as the system
                message's last block, ``<custom_instructions>``. They add to the rules, never replace them, and are
                never rendered as a template, so ``{{`` stays literal. ``None`` or a blank string leaves the system
                message unchanged. The user message always carries the data.
            tz: The zone message times are shown in, as :func:`everalgo.utils.format_time` renders them: ``UTC``
                (the default) or local time with its offset, ``UTC+08:00``; the date and weekday follow it.
            hard_token_limit: Leading segments are cut off, without the model, until the chat messages left hold
                fewer tokens than this.
            hard_message_limit: Likewise, until fewer chat messages than this are left.

        Returns:
            :class:`~everalgo.boundary.Cuts` over ``items``, with ``tail_start`` at the last cut (0 without one).
            Without any ``ChatMessage`` no model is called and the result is ``Cuts([], 0)``. The caller owns idle
            or end-of-session flushing.

        Raises:
            LLMOutputError: If the LLM reply is not a valid batch output. The client's own exceptions propagate
                unchanged; retry policy is the caller's concern.
            InvalidInputError: If a chat message the prompt shows has text but no timestamp (0).
        """
        chat_positions, messages = _chat_view(items)
        if not messages:
            return Cuts(ends=[], tail_start=0)

        chat_cuts = _force_split_cuts(messages, hard_token_limit, hard_message_limit)
        window_start = chat_cuts[-1] if chat_cuts else 0
        boundaries = await _ask_for_boundaries(messages[window_start:], self._llm, custom_instructions, tz=tz)
        chat_cuts += [window_start + boundary for boundary in boundaries]
        if is_final:
            chat_cuts.append(len(messages))

        ends = _item_cuts(chat_cuts, chat_positions, len(items))
        return Cuts(ends=ends, tail_start=ends[-1] if ends else 0)

    detect_batch = async_to_sync(adetect_batch)
    """Sync bridge — only callable from non-event-loop contexts."""

    async def adetect_step(
        self,
        items: Sequence[ChatMessage],
        *,
        custom_instructions: str | None = None,
        tz: tzinfo = UTC,
        warmup_messages: int = 2,
        smart_mask: bool = True,
        smart_mask_threshold: int = 5,
        hard_token_limit: int = 8_192,
        hard_message_limit: int = 50,
    ) -> Cuts:
        """Judge whether the last of ``items`` starts a new segment.

        ``items`` is the history already judged — what the previous call left at ``items[tail_start:]`` — followed
        by one new message. With ``n`` history messages the decision is made in this order:

        1. ``n < warmup_messages``: no cut, and no model call.
        2. ``history + [new]`` reaches a hard limit and ``n >= 2``: cut, without the model.
        3. Otherwise one model call. Once ``n`` exceeds ``smart_mask_threshold`` (and ``smart_mask`` is on), the
           model is not shown the history's last message, and a cut carries that message into the next segment
           as well (the bridge).

        Args:
            items: The judged history, then the new message; ``ChatMessage`` only.
            custom_instructions: The caller's instructions, appended verbatim after the bundled rules as the system
                message's last block, ``<custom_instructions>``. They add to the rules, never replace them, and are
                never rendered as a template, so ``{{`` stays literal. ``None`` or a blank string leaves the system
                message unchanged. The user message always carries the data.
            tz: The zone message times are shown in, as :func:`everalgo.utils.format_time` renders them: ``UTC``
                (the default) or local time with its offset, ``UTC+08:00``; the date and weekday follow it.
            warmup_messages: A history shorter than this is extended without asking the model.
            smart_mask: Whether a long history is masked and bridged; see ``smart_mask_threshold``.
            smart_mask_threshold: The history length the mask and the bridge start above.
            hard_token_limit: Cut without the model once ``history + [new]`` holds at least this many tokens.
            hard_message_limit: Cut without the model once ``history + [new]`` holds at least this many messages.

        Returns:
            One of three :class:`~everalgo.boundary.Cuts` over ``items``: ``Cuts([], 0)`` when the new message
            continues the history; ``Cuts([n], n)`` when the history closes and the new message starts the next
            segment; ``Cuts([n], n - 1)`` when the history closes and its last message is bridged into the next.

        Raises:
            InvalidInputError: If ``items`` is empty or holds anything but ``ChatMessage``, or a message the prompt
                shows has text but no timestamp (0).
            LLMOutputError: If the LLM reply is not a valid step output. The client's own exceptions propagate
                unchanged; retry policy is the caller's concern.
        """
        _check_step_items(items)
        n = len(items) - 1
        history, new = items[:n], items[n]
        if n < warmup_messages:
            return Cuts(ends=[], tail_start=0)

        bridge = smart_mask and n > smart_mask_threshold
        closed = Cuts(ends=[n], tail_start=n - 1 if bridge else n)
        if _needs_force_split(history, new, hard_token_limit, hard_message_limit):
            return closed
        shown = history[:-1] if bridge else history
        # An empty history cannot end an episode, so it is never put to the model.
        if shown and await _ask_should_end(shown, new, self._llm, custom_instructions, tz=tz):
            return closed
        return Cuts(ends=[], tail_start=0)

    detect_step = async_to_sync(adetect_step)
    """Sync bridge — only callable from non-event-loop contexts."""


def _check_step_items(items: Sequence[object]) -> None:
    """Raise unless ``items`` is a non-empty run of chat messages.

    Raises:
        InvalidInputError: If ``items`` is empty or holds anything but ``ChatMessage``.
    """
    if not items:
        raise InvalidInputError("adetect_step got no items: pass the judged history followed by the new message")
    for index, item in enumerate(items):
        if not isinstance(item, ChatMessage):
            raise InvalidInputError(
                f"adetect_step item {index} is a {type(item).__name__}, not a ChatMessage: the step path judges "
                "chat messages only; send trajectories with tool items through adetect_batch"
            )


async def _ask_for_boundaries(
    messages: Sequence[ChatMessage],
    llm: LLMClient,
    custom_instructions: str | None,
    *,
    tz: tzinfo,
) -> list[int]:
    """Call the LLM once and return the usable new-episode start indices.

    The schema guarantees integers only; which of them can cut ``messages`` is checked here: indices outside
    ``1 <= b < len(messages)`` are dropped, the rest deduplicated and sorted.

    Raises:
        LLMOutputError: If the LLM reply is not a valid batch output.
        InvalidInputError: If a message the prompt shows has no timestamp (0).
    """
    reply = await _call_llm_for_batch(llm, _batch_request(messages, custom_instructions, tz=tz))
    in_range = [b for b in reply.boundaries if 1 <= b < len(messages)]
    boundaries = sorted(set(in_range))
    logger.info(
        "boundary batch reply: messages=%d boundaries=%d dropped_out_of_range=%d",
        len(messages),
        len(boundaries),
        len(reply.boundaries) - len(in_range),
    )
    return boundaries


async def _ask_should_end(
    history: Sequence[ChatMessage],
    new: ChatMessage,
    llm: LLMClient,
    custom_instructions: str | None,
    *,
    tz: tzinfo,
) -> bool:
    """Call the LLM once: does ``new`` end the episode ``history`` holds and start the next?

    Raises:
        LLMOutputError: If the LLM reply is not a valid step output.
        InvalidInputError: If a message the prompt shows has no timestamp (0).
    """
    reply = await _call_llm_for_step(llm, _step_request(history, new, custom_instructions, tz=tz))
    logger.info("boundary step reply: should_end=%s", reply.should_end)
    return reply.should_end


def _batch_request(messages: Sequence[ChatMessage], custom_instructions: str | None, *, tz: tzinfo) -> list[LLMMessage]:
    """The batch call's ``[system, user]`` pair; ``count`` is the window size, so it bounds the indices.

    Raises:
        InvalidInputError: If a message the prompt shows has no timestamp (0).
    """
    user = render_prompt(
        BATCH_BOUNDARY_DETECT_USER_PROMPT,
        None,
        count=len(messages),
        messages=_format_messages_with_indices(messages, tz=tz),
    )
    return [
        LLMMessage(
            role="system",
            content=assemble_system_prompt(
                BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT, custom_instructions=custom_instructions
            ),
        ),
        LLMMessage(role="user", content=user),
    ]


def _step_request(
    history: Sequence[ChatMessage],
    new: ChatMessage,
    custom_instructions: str | None,
    *,
    tz: tzinfo,
) -> list[LLMMessage]:
    """The step call's ``[system, user]`` pair over a non-empty ``history``; the time gap runs from its last message.

    Raises:
        InvalidInputError: If a message the prompt shows has no timestamp (0).
    """
    user = render_prompt(
        STEP_BOUNDARY_DETECT_USER_PROMPT,
        None,
        conversation_history=_format_messages_step(history, tz=tz),
        time_gap_info=_calculate_time_gap(history[-1], new),
        new_messages=_format_messages_step([new], tz=tz),
    )
    return [
        LLMMessage(
            role="system",
            content=assemble_system_prompt(STEP_BOUNDARY_DETECT_SYSTEM_PROMPT, custom_instructions=custom_instructions),
        ),
        LLMMessage(role="user", content=user),
    ]
