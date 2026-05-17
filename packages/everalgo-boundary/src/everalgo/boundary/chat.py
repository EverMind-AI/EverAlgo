"""Low-level chat boundary detection primitive — splits a flat list of ChatMessages into MemCell slices.

Facade classes for the two upstream scenarios live in ``everalgo-user-memory`` (``BoundaryDetector``)
and ``everalgo-agent-memory`` (``AgentBoundaryDetector``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, cast

from everalgo._tokenize import count_tokens
from everalgo.boundary.prompts.en.chat import CHAT_BOUNDARY_DETECT_PROMPT_EN
from everalgo.llm.format import format_message_timestamp
from everalgo.llm.parse import parse_llm_json_object
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import ChatMessage, ConversationItem, MemCell

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


DEFAULT_HARD_TOKEN_LIMIT = 65536
DEFAULT_HARD_MSG_LIMIT = 500


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
    prompt_template = prompt or CHAT_BOUNDARY_DETECT_PROMPT_EN

    # Phase 3 — Force-split loop
    cells: list[MemCell] = []
    remaining: list[ChatMessage] = list(messages)
    while len(remaining) > 1:
        total_tokens = count_tokens(_render_for_token_count(remaining))
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


# ---------- Module-level helpers (stateless, prefix `_`) ----------


def _render_for_token_count(messages: list[ChatMessage]) -> str:
    """Render messages as ``"{sender}: {content}"`` lines for token counting.

    Includes the sender prefix so the count reflects the actual prompt budget.
    """
    lines: list[str] = []
    for m in messages:
        speaker = m.sender_name or m.sender_id
        content = (
            m.content
            if isinstance(m.content, str)
            else " ".join(block.text for block in m.content if hasattr(block, "text"))
        )
        lines.append(f"{speaker}: {content}")
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
    while candidate > 1 and count_tokens(_render_for_token_count(messages[:candidate])) >= hard_token_limit:
        candidate = max(1, candidate // 2)
    return candidate


async def _detect_boundaries(
    messages: list[ChatMessage],
    llm: LLMClient,
    prompt_template: str,
) -> _BatchBoundaryResult:
    """Call the LLM once; return parsed boundaries.

    Boundary indices validated to ``1 <= b < len(messages)`` and deduplicated.

    Raises:
        ValueError: If the LLM response cannot be parsed as a valid JSON object.
        TypeError: If the ``boundaries`` field is not a list.
    """
    messages_text = _format_messages_with_indices(messages)
    rendered = render_prompt(prompt_template, None, messages=messages_text)

    response = await llm.chat(
        messages=[LLMChatMessage(role="user", content=rendered)],
        response_format={"type": "json_object"},
    )
    result = _parse_batch_boundary_response(response.content)
    valid = sorted({b for b in result.boundaries if 1 <= b < len(messages)})
    return _BatchBoundaryResult(boundaries=valid, should_wait=result.should_wait)


def _parse_batch_boundary_response(raw: str) -> _BatchBoundaryResult:
    """Parse LLM batch-boundary response.

    Schema: ``{"boundaries": list[int], "should_wait": bool, ...}``.
    Non-int boundary entries are silently skipped.

    Raises:
        ValueError: If the response cannot be parsed as a valid JSON object.
        TypeError: If ``boundaries`` is not a list.
    """
    data = parse_llm_json_object(raw)

    raw_boundaries_val = data.get("boundaries", [])
    if not isinstance(raw_boundaries_val, list):
        raise TypeError("LLM boundary response 'boundaries' field is not a list")
    boundaries: list[int] = [int(item) for item in cast("list[int | str]", raw_boundaries_val)]

    return _BatchBoundaryResult(boundaries=boundaries, should_wait=bool(data.get("should_wait", False)))


def _make_cell(slice_msgs: list[ChatMessage]) -> MemCell:
    """Build a MemCell from a non-empty message slice."""
    return MemCell(items=cast("list[ConversationItem]", slice_msgs), timestamp=slice_msgs[-1].timestamp)
