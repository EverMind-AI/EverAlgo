"""Chat-style MemCell extractor — batch multi-boundary detection aligned with new-release opensource.

Interface follows ``2026-05-13-chat-boundary-detector-design.md`` (5-phase ``adetect`` returning
``DetectionOutput(cells, tail)`` with ``is_final``, ``hard_token_limit``, ``hard_msg_limit``).

Algorithm and prompt content are ported verbatim from new-release opensource
``opensource/evermemos-opensource/src/memory_layer/memcell_extractor/conv_memcell_extractor.py``:

- Force-split loop (token + message limit) before LLM call (line 503-530).
- Single batch LLM call returns all boundaries + ``should_wait`` (line 373-451).
- 3-tier JSON parse: ```` ```json ... ``` ```` fence -> direct ``json.loads`` -> outermost ``{...}``
  extraction (line 314-371).
- 5-retry on JSON parse failure; on exhaustion ``raise RuntimeError`` (line 446-451).
- ``_extract_participant_ids`` only collects ``sender_id`` from ``role == "user"`` messages, no
  ``refer_list`` walk (line 129-151).
- Message rendering ``[N] [ISO+TZ] sender_name: content`` (line 272-312).
- Token count includes sender prefix so it matches what the LLM sees (line 104-127).

Deliberate divergences from opensource (per design doc Section 3):
- Public interface is a **stateless function** (no ``__init__`` DI); LLM / prompt / limits per-call.
- Return type is ``DetectionOutput(cells, tail)`` instead of ``(cells, StatusResult)``; tail is explicit
  rather than via an implicit history buffer the caller maintains.
- ``is_final: bool`` parameter replaces ``request.flush: bool``; algorithm has no request object.
- ``should_wait`` is parsed but not exposed; caller derives "should I wait" from ``len(tail) > 0``
  combined with the ``is_final`` they chose to pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, NamedTuple

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo._tokenize import count_tokens
from everalgo.boundary.prompts.en.chat import CHAT_BOUNDARY_DETECT_PROMPT_EN
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.types import MemCell, Message, MessageRole, RawDataType

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


DEFAULT_HARD_TOKEN_LIMIT = 65536
"""Force-split threshold by token count (per design doc §1)."""

DEFAULT_HARD_MSG_LIMIT = 500
"""Force-split threshold by message count (per design doc §1)."""

_MAX_LLM_RETRIES = 5
"""Maximum retries for LLM JSON / schema failures (opensource line 408)."""


@dataclass(frozen=True)
class _BatchBoundaryResult:
    """Parsed LLM batch response (opensource ``BatchBoundaryResult`` line 38-44)."""

    boundaries: list[int]
    should_wait: bool


class DetectionOutput(NamedTuple):
    """Return type of :meth:`ChatMemCellExtractor.adetect`.

    NamedTuple subclass of ``tuple`` — supports positional unpacking, named access, and index access::

        cells, tail = await ChatMemCellExtractor().adetect(messages)
        output = await ChatMemCellExtractor().adetect(messages)
        output.cells  # list[MemCell]
        output.tail  # list[Message]
        output[0], output[1]

    Semantics
    ---------
    cells
        Zero or more :class:`MemCell` instances the algorithm has confidently closed. Caller persists
        immediately. Each cell carries ``participants`` / ``sender_ids`` (distinct ``sender_id`` values
        from ``role == "user"`` messages) and ``type == RawDataType.CONVERSATION``.
    tail
        Trailing messages the algorithm did NOT close — the LLM cannot judge whether the conversation
        continues beyond the last seen message, so the trailing segment is left as a tail for the caller's
        state machine. When ``is_final=True``, tail is forced into the final MemCell and this field is
        guaranteed empty.
    """

    cells: list[MemCell]
    tail: list[Message]


class ChatMemCellExtractor:
    """Stateless conversation boundary detector — batch multi-boundary detection.

    No ``__init__`` / no instance state — instances are interchangeable, async-safe, thread-safe. Customize
    per call via ``llm=`` / ``prompt=`` / ``is_final=`` / ``hard_token_limit=`` / ``hard_msg_limit=``.

    Algorithm: 5 phases per design doc §2, with new-release batch LLM call inside Phase 4.

    1. **Input validation** — empty ``messages`` short-circuits to ``([], [])``.
    2. **Default resolution** — resolve ``llm`` via the 3-layer fallback; resolve ``prompt`` via the
       optional caller override + :data:`CHAT_BOUNDARY_DETECT_PROMPT_EN`.
    3. **Force-split loop** — while ``count_tokens >= hard_token_limit`` or ``len >= hard_msg_limit``, pick
       a binary-halving split point and emit a force MemCell. Runs **before** the LLM loop to avoid
       context overflow.
    4. **LLM batch detection** — single LLM call returns all ``boundaries`` + advisory ``should_wait``.
       Retries 5 times on JSON parse / schema failure; on exhaustion raises ``RuntimeError``.
       Infrastructure errors (LLMError / network / auth) propagate.
    5. **is_final closure** — if ``is_final=True``, the final trailing segment is closed as the last
       MemCell; otherwise it becomes ``tail``.

    Cell construction extracts ``participants`` / ``sender_ids`` from messages and stamps
    ``type=RawDataType.CONVERSATION``.
    """

    async def adetect(
        self,
        messages: list[Message],
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
        is_final: bool = False,
        hard_token_limit: int = DEFAULT_HARD_TOKEN_LIMIT,
        hard_msg_limit: int = DEFAULT_HARD_MSG_LIMIT,
    ) -> DetectionOutput:
        """Split messages into MemCells; return ``(cells, tail)``.

        Parameters
        ----------
        messages : list[Message]
            Ordered chat messages. Typically the caller's prior ``tail`` prepended to fresh new messages.
        llm : LLMClient or None, optional
            Per-call LLM override; 3-layer fallback. Raises :class:`LLMNotConfiguredError` if all None.
        prompt : str or None, optional
            Per-call boundary-detection prompt template. Defaults to
            :data:`CHAT_BOUNDARY_DETECT_PROMPT_EN`. Must contain a ``{messages}`` placeholder.
        is_final : bool, optional
            When ``True``, the final trailing segment is forced into a closing MemCell so ``tail == []``.
            When ``False`` (default), allow tail to be non-empty.
        hard_token_limit : int, optional
            Force-split threshold by token count. Defaults to 65536.
        hard_msg_limit : int, optional
            Force-split threshold by message count. Defaults to 500.

        Returns
        -------
        DetectionOutput
            ``(cells, tail)``. ``is_final=False`` → tail may be non-empty;
            ``is_final=True`` → ``tail == []``.

        Raises
        ------
        LLMNotConfiguredError
            No LLM resolvable through the 3-layer fallback chain.
        RuntimeError
            All 5 LLM retries exhausted on JSON parse / schema failure.
        """
        # Phase 1 — Input validation
        if not messages:
            return DetectionOutput(cells=[], tail=[])

        # Phase 2 — Default resolution (eager so missing-LLM fails fast on entry).
        client = everalgo.llm.resolve(llm)
        prompt_template = prompt or CHAT_BOUNDARY_DETECT_PROMPT_EN

        # Phase 3 — Force-split loop
        cells: list[MemCell] = []
        remaining: list[Message] = list(messages)
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
            batch = await _detect_boundaries(remaining, client, prompt_template)
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
            tail: list[Message] = []
        else:
            tail = tail_msgs

        return DetectionOutput(cells=cells, tail=tail)

    detect = async_to_sync(adetect)
    """Sync bridge — only callable from non-event-loop contexts."""


# ---------- Module-level helpers (stateless, prefix `_`) ----------


def _render_for_token_count(messages: list[Message]) -> str:
    """Render messages as ``f"{sender}: {content}"`` lines for token counting.

    Matches opensource ``_count_tokens`` line 104-127: include the sender prefix that's actually sent to
    the LLM so the count reflects the prompt budget. Falls back to ``role.value`` when ``sender_name`` is
    missing, but real production input is expected to carry ``sender_name`` (set by upstream enrichment).
    """
    lines: list[str] = []
    for m in messages:
        speaker = m.sender_name or m.role.value
        lines.append(f"{speaker}: {m.content}")
    return "\n".join(lines)


def _format_messages_with_indices(messages: list[Message]) -> str:
    """Render messages as ``[N] [YYYY-MM-DD HH:MM:SS+TZ] sender_name: content`` lines.

    Ported from opensource ``_format_messages_with_indices`` line 272-312. ``N`` is 1-based. Timestamp is
    rendered as ISO-format with UTC timezone offset; the LLM consumes this to reason about cross-day /
    long-gap boundaries directly (no separate natural-language ``time_gap_info`` is injected).

    Messages with empty content are silently dropped (matches opensource line 307-310).
    """
    lines: list[str] = []
    for i, msg in enumerate(messages, start=1):
        if not msg.content:
            continue
        sender_name = msg.sender_name or msg.role.value
        time_str = datetime.fromtimestamp(msg.timestamp / 1000, tz=UTC).isoformat(sep=" ", timespec="seconds")
        lines.append(f"[{i}] [{time_str}] {sender_name}: {msg.content}")
    return "\n".join(lines)


def _find_force_split_point(messages: list[Message], hard_token_limit: int, hard_msg_limit: int) -> int:
    """Binary-halving search for a force-split index.

    Ported from opensource ``_find_force_split_point`` line 153-179. Starts at
    ``min(hard_msg_limit - 1, len - 1)``; halves while the head exceeds ``hard_token_limit``. Floor 1 —
    single-message head is acceptable when one message is internally very long.
    """
    if len(messages) <= 1:
        return len(messages)
    candidate = min(hard_msg_limit - 1, len(messages) - 1)
    while candidate > 1 and count_tokens(_render_for_token_count(messages[:candidate])) >= hard_token_limit:
        candidate = max(1, candidate // 2)
    return candidate


def _extract_participants(messages: list[Message]) -> list[str]:
    """Collect distinct ``sender_id`` values from ``role == "user"`` messages.

    Ported from opensource ``_extract_participant_ids`` line 129-151. Preserves first-occurrence order;
    deduplicated. Unlike the older release, ``refer_list`` is NOT walked.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for msg in messages:
        if msg.role != MessageRole.USER:
            continue
        if not msg.sender_id or msg.sender_id in seen:
            continue
        seen.add(msg.sender_id)
        ordered.append(msg.sender_id)
    return ordered


async def _detect_boundaries(
    messages: list[Message],
    client: LLMClient,
    prompt_template: str,
) -> _BatchBoundaryResult:
    """Call the LLM once with the batch prompt; return parsed boundaries + advisory should_wait.

    Ported from opensource ``_detect_boundaries`` line 373-451. Retries up to 5 times when JSON parsing /
    schema validation fails; infrastructure errors propagate. On exhaustion raises ``RuntimeError`` (the
    opensource ``logger.error + raise`` pattern).

    Boundary indices are validated to ``1 <= b < len(messages)`` (1-based; strict-less-than-end so the
    tail is preserved) and deduplicated via ``sorted(set(...))``.
    """
    messages_text = _format_messages_with_indices(messages)
    rendered = prompt_template.format(messages=messages_text)

    for _attempt in range(_MAX_LLM_RETRIES):
        try:
            response = await client.chat(
                messages=[LLMChatMessage(role="user", content=rendered)],
                response_format={"type": "json_object"},
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
        result = _parse_batch_boundary_response(response.content)
        if result is None:
            continue
        valid = sorted({b for b in result.boundaries if 1 <= b < len(messages)})
        return _BatchBoundaryResult(boundaries=valid, should_wait=result.should_wait)

    raise RuntimeError("All 5 retries exhausted for boundary detection")


def _parse_batch_boundary_response(raw: str) -> _BatchBoundaryResult | None:
    """Parse new-release LLM batch response. Return result or ``None`` on failure.

    Schema: ``{"boundaries": list[int], "should_wait": bool, ...}``. 3-tier parse matching opensource
    ``_parse_batch_boundary_response`` line 314-371:

    1. Markdown ```` ```json ... ``` ```` fence first.
    2. Direct ``json.loads`` on the stripped string.
    3. Outermost ``{...}`` extraction via ``find('{') / rfind('}')`` — handles nested braces.

    Individual boundary entries that fail ``int(...)`` coercion are skipped (not fatal).
    """
    data = _parse_json_three_tier(raw)
    if not isinstance(data, dict):
        return None

    raw_boundaries = data.get("boundaries", [])
    if not isinstance(raw_boundaries, list):
        return None
    boundaries: list[int] = []
    for item in raw_boundaries:
        try:
            boundaries.append(int(item))
        except (TypeError, ValueError):
            continue

    return _BatchBoundaryResult(boundaries=boundaries, should_wait=bool(data.get("should_wait", False)))


def _parse_json_three_tier(raw: str) -> Any:
    """Try parsing ``raw`` as JSON via fence -> direct -> outermost-braces fallback.

    Returns the parsed object on success, ``None`` if all three strategies fail.
    """
    fence_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _make_cell(slice_msgs: list[Message]) -> MemCell:
    """Build a MemCell from a non-empty message slice.

    ``original_data`` is the opensource ``[{"message": msg_dict}]`` wrapping (line 181-202).
    ``participants`` / ``sender_ids`` are the same set per opensource note (memory_types.py line 152-155).

    ``event_id`` is filled with a deterministic ``mc_<first_ts>_<last_ts>`` placeholder so downstream
    extractors (Episode / Foresight / AtomicFact, all of which require non-null ``parent_id``) can run
    standalone without a persistence round-trip. Callers persisting through their own pipeline may
    overwrite this field with a DB-assigned identifier before recording the MemCell. This is a deliberate
    divergence from opensource (which leaves ``event_id=None`` until the DB save step) — documented in the
    design doc Section 3.
    """
    participants = _extract_participants(slice_msgs)
    first_ts = slice_msgs[0].timestamp
    last_ts = slice_msgs[-1].timestamp
    return MemCell(
        user_id_list=[],
        original_data=[{"message": m.model_dump(exclude_none=True)} for m in slice_msgs],
        timestamp=last_ts,
        event_id=f"mc_{first_ts}_{last_ts}",
        group_id=None,
        participants=participants,
        sender_ids=participants,
        type=RawDataType.CONVERSATION,
    )
