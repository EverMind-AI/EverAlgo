"""Low-level chat boundary detection primitive — splits a flat list of ChatMessages into MemCell slices.

Facade classes for the two upstream scenarios live in ``everalgo-user-memory`` (``BoundaryDetector``)
and ``everalgo-agent-memory`` (``AgentBoundaryDetector``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from everalgo._tokenize import count_tokens
from everalgo.boundary.prompts.en.chat import BATCH_BOUNDARY_DETECT_PROMPT_EN, STEP_BOUNDARY_DETECT_PROMPT_EN
from everalgo.llm.format import format_message_timestamp
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import ChatMessage, ConversationItem, MemCell

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


DEFAULT_HARD_TOKEN_LIMIT = 8192
DEFAULT_HARD_MSG_LIMIT = 50


@dataclass(frozen=True)
class _BatchBoundaryResult:
    """Parsed LLM batch response."""

    boundaries: list[int]
    should_wait: bool


class DetectionResult(NamedTuple):
    """Return type of :func:`detect_boundaries`.

    NamedTuple subclass of ``tuple`` — supports positional unpacking, named access, and index access::

        cells, tail = await detect_boundaries(messages, llm=client)
        result = await detect_boundaries(messages, llm=client)
        result.cells  # list[MemCell]
        result.tail  # list[ChatMessage]
        result[0], result[1]

    Attributes:
        cells: Zero or more :class:`~everalgo.types.MemCell` instances the algorithm has confidently
            closed.  Caller persists immediately.
        tail: Trailing messages the algorithm did NOT close — the LLM cannot judge whether the
            conversation continues beyond the last seen message, so the trailing segment is left as a
            tail for the caller's state machine.  When ``is_final=True``, tail is forced into the final
            MemCell and this field is guaranteed empty.
    """

    cells: list[MemCell]
    tail: list[ChatMessage]


async def detect_boundaries(
    messages: list[ChatMessage],
    *,
    llm: LLMClient,
    is_final: bool = False,
    prompt: str | None = None,
    hard_token_limit: int = DEFAULT_HARD_TOKEN_LIMIT,
    hard_msg_limit: int = DEFAULT_HARD_MSG_LIMIT,
) -> DetectionResult:
    """Detect MemCell boundaries on a list of ChatMessages (chat-only path).

    Args:
        messages: Ordered chat messages, typically ``prior_tail + new_messages``.
        llm: LLM client. Algorithm always requires an LLM; the caller must supply one.
        is_final: When ``True``, forces the trailing segment into a cell so ``tail == []``.
        prompt: Prompt override; ``None`` uses the bundled default.
        hard_token_limit: Max tokens per cell before forced split.
        hard_msg_limit: Max messages per cell before forced split.

    Raises:
        ValueError: If the LLM response cannot be parsed as a valid JSON object.
        TypeError: If the ``boundaries`` field in the LLM response is not a list.
        LLMError: Propagated from the underlying LLM client call.
    """
    # Phase 1 — Input validation
    if not messages:
        return DetectionResult(cells=[], tail=[])

    # Phase 2 — Default resolution
    prompt_template = prompt or BATCH_BOUNDARY_DETECT_PROMPT_EN

    # Phase 3 — Force-split loop
    cells: list[MemCell] = []
    remaining: list[ChatMessage] = list(messages)
    while len(remaining) > 1:
        total_tokens = _count_history_tokens(remaining)
        total_msgs = len(remaining)
        if total_tokens < hard_token_limit and total_msgs < hard_msg_limit:
            break
        split_at = _find_force_split_point(remaining, hard_token_limit, hard_msg_limit)
        cells.append(_make_cell(remaining[:split_at]))
        remaining = remaining[split_at:]

    # Phase 4 — LLM batch detection (single call, multi-boundary).
    boundaries: list[int] = []
    if remaining:
        batch = await _detect_boundaries(remaining, llm, prompt_template)
        boundaries = batch.boundaries

    # Phase 5 — Slice by boundaries + is_final closure.
    prev = 0
    for b in boundaries:
        segment = remaining[prev:b]
        if segment:
            cells.append(_make_cell(segment))
        prev = b
    tail_msgs = remaining[prev:]

    if is_final and tail_msgs:
        cells.append(_make_cell(tail_msgs))
        tail: list[ChatMessage] = []
    else:
        tail = tail_msgs

    return DetectionResult(cells=cells, tail=tail)


# ---------------------------------------------------------------------------
# Single-step incremental boundary detection.
# ---------------------------------------------------------------------------


class BoundaryDecision(NamedTuple):
    """Single-step boundary decision returned by :func:`adetect_boundary_step`.

    Attributes:
        should_end: Whether the current episode should end (i.e. the LLM is ready to commit a cut between
            ``conversation_history`` and ``new_messages``). ``True`` means the segment is ready to commit;
            ``False`` means keep buffering (the current episode is not yet closed).
        reasoning: One-sentence LLM rationale.
        confidence: LLM-reported confidence in [0.0, 1.0].
        topic_summary: Topic label when ``should_end=True``; empty string otherwise.
    """

    should_end: bool
    reasoning: str
    confidence: float
    topic_summary: str


async def adetect_boundary_step(
    conversation_history: list[ChatMessage],
    new_messages: list[ChatMessage],
    *,
    llm: LLMClient,
    prompt: str | None = None,
    time_gap_info: str = "",
) -> BoundaryDecision:
    """Single-step LLM boundary decision over ``history + new_messages``.

    Caller is responsible for the outer accumulation loop (front-2-buffer / smart_mask / cut-and-bridge)
    and the force-split short-circuit; this primitive only owns the LLM call.

    Args:
        conversation_history: Prior accumulated messages (may be empty).
        new_messages: Newly arrived messages whose boundary status is being judged.
        llm: LLM client. Algorithm always requires an LLM; the caller must supply one.
        prompt: Prompt template override; ``None`` uses ``STEP_BOUNDARY_DETECT_PROMPT_EN``.
        time_gap_info: Pre-computed time-gap label injected into the prompt; empty string means
            "no time gap available" (still rendered as-is into the template).

    Returns:
        :class:`BoundaryDecision` with the LLM's verdict.

    Raises:
        ValueError: When the LLM fails to return a parseable JSON object. Algorithms fail-loud;
            retry/fallback policy is the caller's concern.
        LLMError: Propagated from the underlying LLM client.
    """
    # Phase 1: empty history short-circuit
    if not conversation_history:
        return BoundaryDecision(
            should_end=False,
            reasoning="First messages in conversation",
            confidence=1.0,
            topic_summary="",
        )

    prompt_template = prompt or STEP_BOUNDARY_DETECT_PROMPT_EN
    history_text = _format_messages_step(conversation_history)
    new_text = _format_messages_step(new_messages)
    rendered = prompt_template.format(
        conversation_history=history_text,
        new_messages=new_text,
        time_gap_info=time_gap_info or "No time gap information available",
    )

    return await _call_llm_for_boundary_step(llm, rendered)


async def _call_llm_for_boundary_step(llm: LLMClient, rendered: str) -> BoundaryDecision:
    """Inner async function decorated with retry; separated so the decorator applies at definition time."""
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in boundary LLM response: {text[:200]!r}")
    data = json.loads(match.group())
    if "should_end" not in data:
        raise ValueError(f"should_end missing from boundary step LLM response: {data!r}")
    return BoundaryDecision(
        should_end=bool(data["should_end"]),
        reasoning=str(data.get("reasoning", "")),
        confidence=float(data.get("confidence", 1.0)),
        topic_summary=str(data.get("topic_summary", "")),
    )


# ---------- Module-level helpers (stateless, prefix `_`) ----------


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


def _format_messages_step(messages: list[ChatMessage]) -> str:
    """Render messages for the incremental step path.

    Line format: ``[<ISO 8601 UTC>] <sender_name>: <content>`` — **no 1-based index prefix**.
    The step-mode LLM (``adetect_boundary_step``) returns a single ``should_end`` decision rather
    than a list of indexed boundaries, so the ``[N]`` prefix used by the batch path would be noise here.

    Empty-content messages are dropped (same as ``_format_messages_with_indices``).
    """
    lines: list[str] = []
    for msg in messages:
        content = (
            msg.content
            if isinstance(msg.content, str)
            else " ".join(block.text for block in msg.content if hasattr(block, "text"))
        )
        if not content:
            continue
        sender_name = msg.sender_name or msg.sender_id
        time_str = format_message_timestamp(msg.timestamp)
        lines.append(f"[{time_str}] {sender_name}: {content}")
    return "\n".join(lines)


def _format_messages_with_indices(messages: list[ChatMessage]) -> str:
    """Render messages as ``[N] [ISO+TZ] sender_name: content`` lines (1-based N). Empty-content messages are dropped."""
    lines: list[str] = []
    for i, msg in enumerate(messages, start=1):
        content = (
            msg.content
            if isinstance(msg.content, str)
            else " ".join(block.text for block in msg.content if hasattr(block, "text"))
        )
        if not content:
            continue
        sender_name = msg.sender_name or msg.sender_id
        time_str = format_message_timestamp(msg.timestamp)
        lines.append(f"[{i}] [{time_str}] {sender_name}: {content}")
    return "\n".join(lines)


def _find_force_split_point(messages: list[ChatMessage], hard_token_limit: int, hard_msg_limit: int) -> int:
    """Binary-halving search for a force-split index; floor 1 so even a single very-long message is accepted."""
    if len(messages) <= 1:
        return len(messages)
    candidate = min(hard_msg_limit - 1, len(messages) - 1)
    while candidate > 1 and _count_history_tokens(messages[:candidate]) >= hard_token_limit:
        candidate = max(1, candidate // 2)
    return candidate


async def _call_llm_for_batch_boundary(llm: LLMClient, rendered: str, num_messages: int) -> _BatchBoundaryResult:
    """Inner async function decorated with retry; separated so the decorator applies at definition time.

    Matches flat JSON only (no nested objects). ``boundaries`` is list[int] and ``should_wait`` is bool.
    """
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in batch boundary LLM response: {text[:200]!r}")
    data = json.loads(match.group())
    raw_boundaries = data.get("boundaries", [])
    if not isinstance(raw_boundaries, list):
        raise ValueError(f"boundaries must be a list, got {type(raw_boundaries).__name__}")  # noqa: TRY004
    coerced: list[int] = []
    for i, item in enumerate(cast("list[Any]", raw_boundaries)):  # type: ignore[redundant-cast]
        if item is None:
            continue
        try:
            coerced.append(int(item))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"boundaries[{i}]={item!r} cannot be coerced to int") from exc
    valid = sorted({b for b in coerced if 1 <= b < num_messages})
    if "should_wait" not in data:
        raise ValueError(f"should_wait missing from batch boundary LLM response: {data!r}")
    return _BatchBoundaryResult(boundaries=valid, should_wait=bool(data["should_wait"]))


async def _detect_boundaries(
    messages: list[ChatMessage],
    llm: LLMClient,
    prompt_template: str,
) -> _BatchBoundaryResult:
    """Call the LLM once; return parsed boundaries.

    Boundary indices validated to ``1 <= b < len(messages)`` and deduplicated.

    Raises:
        ValueError: If the LLM returns no parseable JSON object after 5 retries.
    """
    messages_text = _format_messages_with_indices(messages)
    rendered = render_prompt(prompt_template, None, messages=messages_text)
    return await _call_llm_for_batch_boundary(llm, rendered, len(messages))


def _make_cell(slice_msgs: list[ChatMessage]) -> MemCell:
    """Build a MemCell from a non-empty message slice."""
    return MemCell(items=cast("list[ConversationItem]", slice_msgs), timestamp=slice_msgs[-1].timestamp)
