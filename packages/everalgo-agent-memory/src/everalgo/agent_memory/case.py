"""AgentCaseExtractor — distil one agent execution trajectory into at most one :class:`AgentCase`.

Algorithm ports the opensource ``memory_layer/memory_extractor/agent_case_extractor.py`` 11-step pipeline
verbatim; the only changes are EverAlgo's mandatory compliance edits — see DESIGN.md §4.5 for the full
diff. Public API: :class:`AgentCaseExtractor` with ``aextract`` (async) + ``extract`` (sync bridge).

Internals operate directly on typed :class:`Message` objects throughout — no upfront dict conversion.
OpenAI-format dicts are produced only at the LLM-prompt boundary (and parsed back into :class:`Message`
when the LLM returns messages), so EverAlgo-private fields (``timestamp`` / ``sender_id`` /
``sender_name`` / ``refer_list``) never leak into LLM prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

import everalgo.llm
from everalgo.agent_memory._text import count_tokens, json_default, truncate_text
from everalgo.agent_memory.prompts.case_compress import AGENT_CASE_COMPRESS_PROMPT
from everalgo.agent_memory.prompts.case_filter import AGENT_CASE_FILTER_PROMPT
from everalgo.agent_memory.prompts.tool_pre_compress import AGENT_TOOL_PRE_COMPRESS_PROMPT
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import AgentCase, MemCell, Message, MessageRole

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


__all__ = [
    # Re-exported prompt constants — monkey-patch at startup to override the LLM prompts
    "AGENT_CASE_COMPRESS_PROMPT",
    "AGENT_CASE_FILTER_PROMPT",
    "AGENT_TOOL_PRE_COMPRESS_PROMPT",
    "AgentCaseExtractor",
]


# ── Heuristic constants (opensource :50-65) ─────────────────────────────────────────────────────────
# Tunable algorithm-IP thresholds; override at startup via monkey-patch (DESIGN.md §4.6).

PRE_COMPRESS_CHUNK_SIZE = 100_000
"""Tool-content token threshold above which selective LLM pre-compression kicks in (opensource :52)."""

HIGH_MESSAGE_COUNT_THRESHOLD = 100
"""Halve the scale_trigger when message count exceeds this (opensource :57)."""

MAX_TOOL_OUTPUT_TOKENS = 1000
MAX_TOOL_ARGS_TOKENS = 800
MAX_ASSISTANT_RESPONSE_TOKENS = 3000

MAX_TASK_INTENT_TOKENS = 300
"""Hard cap on ``task_intent`` token length after LLM extraction (opensource :65, head-only truncation)."""

FILTER_NO_TOOL_MAX_MESSAGES = 4
FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS = 200


class AgentCaseExtractor:
    """Distil one agent-trajectory MemCell into at most one :class:`AgentCase`.

    Stateless callable class — no ``__init__``, no instance state. Returns ``list[AgentCase]`` (per
    DESIGN.md §4.2 / O-1) of length 0 or 1: empty list when the pre-filter / LLM filter rejects the
    trajectory, a single-element list on success.

    Customize per call via ``llm=`` and per-prompt overrides (``prompt_filter`` / ``prompt_compress`` /
    ``prompt_tool_pre_compress``); the module-level constants in :mod:`everalgo.agent_memory.prompts.*` are
    used as defaults when the overrides are ``None``. Startup-time global override is also possible via
    monkey-patching the prompt constants on this module.
    """

    async def aextract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt_filter: str | None = None,
        prompt_compress: str | None = None,
        prompt_tool_pre_compress: str | None = None,
    ) -> list[AgentCase]:
        """Async main implementation — runs the 11-step pipeline (DESIGN.md §1.1).

        Parameters
        ----------
        memcell : MemCell
            Boundary output. ``memcell.messages`` MUST carry the OpenAI agent trajectory: USER / ASSISTANT
            (with optional ``tool_calls``) / TOOL (with ``tool_call_id``). System prompts are upstream
            framing and excluded from the schema. Empty / missing messages yields ``[]`` immediately.
        llm : LLMClient or None, optional
            Per-call LLM override; falls back through the 3-layer chain.
        prompt_filter, prompt_compress, prompt_tool_pre_compress : str or None, optional
            Per-call prompt template overrides for steps 7, 8, and 6 respectively. ``None`` falls back to
            the corresponding built-in constant on this module.

        Returns
        -------
        list[AgentCase]
            Length 0 when filtered out; length 1 on successful extraction. Caller embeds the resulting
            ``task_intent`` before persisting (no ``vector`` field on :class:`AgentCase`).

        Raises
        ------
        LLMNotConfiguredError
            No LLM resolvable through the 3-layer chain.
        LLMError
            Any provider-side failure (no internal retry — see DESIGN.md §2 / ADR 012).
        """
        # MemCell.event_id is assigned by the persistence layer and may be None when the algorithm runs
        # before persistence — use empty-string fallback so log messages and parent_id stay non-None.
        memcell_id = memcell.event_id or ""

        if not memcell.messages:
            logger.info("no messages on memcell %s, skipping", memcell_id)
            return []

        client = everalgo.llm.resolve(llm)

        # Step 1+2: typed view from MemCell, strip system head (drop anything before first user)
        msgs = _strip_before_first_user(memcell.messages)

        # Step 3: structural + heuristic pre-filter
        if reason := _should_skip(msgs):
            logger.info("skipping memcell %s: %s", memcell_id, reason)
            return []

        # Step 4: heuristic trim (with scale_trigger adaptation)
        msgs, total_tokens = _heuristic_trim(msgs)
        logger.info(
            "memcell %s pre-trim total_tokens=%d, message_count=%d",
            memcell_id,
            total_tokens,
            len(msgs),
        )

        # Step 5: over-size bail — re-count only when trim scaled down
        if total_tokens > PRE_COMPRESS_CHUNK_SIZE:
            trimmed_tokens = count_tokens(_dump_messages(msgs))
            if trimmed_tokens > PRE_COMPRESS_CHUNK_SIZE * 2:
                logger.info(
                    "memcell %s still %d tokens after trim (> %d), skipping",
                    memcell_id,
                    trimmed_tokens,
                    PRE_COMPRESS_CHUNK_SIZE * 2,
                )
                return []

        # Step 6: selective LLM pre-compression of largest tool-call groups
        msgs = await _pre_compress_to_list(msgs, client, prompt=prompt_tool_pre_compress)
        messages_json = _dump_messages(msgs)

        # Step 7: LLM filter — only for trajectories with ≤ 1 tool-call round
        if _count_tool_call_rounds(msgs) <= 1 and not await _is_worth_extracting(
            messages_json, client, prompt=prompt_filter
        ):
            return []

        # Step 8: single LLM compress (no retry)
        exp = await _compress_experience(messages_json, client, prompt=prompt_compress)
        if not exp:
            return []

        # Step 9: hard truncate task_intent (head-only)
        original_intent = exp.get("task_intent", "")
        intent = truncate_text(original_intent, MAX_TASK_INTENT_TOKENS, head_ratio=1.0)
        if intent != original_intent:
            logger.info(
                "memcell %s truncated task_intent to %d tokens",
                memcell_id,
                MAX_TASK_INTENT_TOKENS,
            )

        # Step 10: ❌ embedding moved to caller — AgentCase emitted without vector.

        # Step 11: construct
        case = AgentCase(
            id=uuid.uuid4().hex,
            timestamp=memcell.timestamp,
            parent_type="memcell",
            parent_id=memcell_id,
            task_intent=intent,
            approach=exp.get("approach", "") or "",
            quality_score=_clamp_quality_score(exp.get("quality_score", 0.5)),
            key_insight=exp.get("key_insight", "") or "",
        )
        return [case]

    extract = async_to_sync(aextract)
    """Sync bridge — only callable from non-event-loop contexts."""


# ── Serialization helpers (Message → OpenAI Chat Completions wire format) ───────────────────────────


def _to_openai_dict(msg: Message) -> dict[str, Any]:
    """Convert a :class:`Message` to an OpenAI Chat Completions wire-format dict.

    Keeps only OpenAI-recognised fields (``role`` / ``content`` / ``tool_calls`` / ``tool_call_id``);
    drops EverAlgo-private fields (``timestamp`` / ``sender_id`` / ``sender_name`` / ``refer_list``) that
    the LLM does not need. ``content`` is omitted when ``None`` (assistant messages carrying only
    ``tool_calls`` follow OpenAI's wire format); ``tool_calls`` / ``tool_call_id`` are emitted only when
    set.
    """
    d: dict[str, Any] = {"role": msg.role.value}
    if msg.content is not None:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [tc.model_dump(mode="json", exclude_none=True) for tc in msg.tool_calls]
    if msg.tool_call_id is not None:
        d["tool_call_id"] = msg.tool_call_id
    return d


def _to_openai_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    """Vector form of :func:`_to_openai_dict`."""
    return [_to_openai_dict(m) for m in messages]


def _dump_messages(messages: list[Message]) -> str:
    """JSON-dump messages in OpenAI wire format — used for LLM prompts and token counting.

    A single canonical (compact) dump shape keeps the trim/over-size-bail heuristics counting tokens
    against the exact byte sequence that downstream LLM prompts consume.
    """
    return json.dumps(_to_openai_dicts(messages), ensure_ascii=False, default=json_default)


# ── Module-level helpers (port of opensource private methods) ───────────────────────────────────────


def _strip_before_first_user(messages: list[Message]) -> list[Message]:
    """Drop everything before the first user message (e.g. system prompts). Opensource :501-509."""
    for i, msg in enumerate(messages):
        if msg.role == MessageRole.USER:
            return list(messages[i:])
    return []


def _has_tool_calls(messages: list[Message]) -> bool:
    """Return ``True`` iff any message has tool_calls or is a tool response. Opensource :486-490."""
    return any(msg.tool_calls or msg.role == MessageRole.TOOL for msg in messages)


def _count_tool_call_rounds(messages: list[Message]) -> int:
    """Count assistant messages that contain tool_calls. Opensource :493-499."""
    return sum(1 for msg in messages if msg.role == MessageRole.ASSISTANT and msg.tool_calls)


def _should_skip(messages: list[Message]) -> str | None:
    """Pre-filter combining structural + heuristic checks. Opensource :511-561.

    Returns the skip reason (a short string) or ``None`` if the trajectory is worth extracting.
    """
    if not messages:
        return "No messages after stripping system prompts"
    if not any(msg.role == MessageRole.USER for msg in messages):
        return "No user messages found"
    if not any(msg.role == MessageRole.ASSISTANT for msg in messages):
        return "No assistant messages found"

    last_msg = messages[-1]
    if last_msg.role != MessageRole.ASSISTANT or last_msg.tool_calls:
        return "Incomplete agent trajectory (last message is not a final assistant response)"

    has_tools = _has_tool_calls(messages)
    if not has_tools:
        user_count = sum(1 for msg in messages if msg.role == MessageRole.USER)
        if user_count < 2:
            return "Single-turn conversation without tool calls"

        if len(messages) <= FILTER_NO_TOOL_MAX_MESSAGES:
            return (
                f"No-tool conversation with only {len(messages)} messages (max {FILTER_NO_TOOL_MAX_MESSAGES}), skipping"
            )

        assistant_content = " ".join(
            (msg.content or "") for msg in messages if msg.role == MessageRole.ASSISTANT and not msg.tool_calls
        )
        assistant_tokens = count_tokens(assistant_content)
        if assistant_tokens < FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS:
            return (
                f"No-tool conversation with brief assistant response "
                f"({assistant_tokens} tokens < {FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS}), skipping"
            )

    return None


def _calc_tool_content_size(msg: Message) -> int:
    """Tool-related token count of a single message. Opensource :140-150."""
    if msg.role == MessageRole.TOOL:
        return count_tokens(msg.content or "")
    if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
        return sum(count_tokens(tc.function.get("arguments", "") or "") for tc in msg.tool_calls)
    return 0


def _heuristic_trim(messages: list[Message]) -> tuple[list[Message], int]:
    """Truncate oversized tool outputs / args / assistant responses; opensource :173-223 + :625-655.

    Auto-scales the per-message caps inversely to overage when total tokens > scale_trigger. Returns
    ``(trimmed_messages, original_total_tokens)``.
    """
    total_tokens = count_tokens(_dump_messages(messages))

    # High message count signals lower per-message value: halve the trigger so trim kicks in earlier.
    scale_trigger = (
        PRE_COMPRESS_CHUNK_SIZE // 2 if len(messages) > HIGH_MESSAGE_COUNT_THRESHOLD else PRE_COMPRESS_CHUNK_SIZE
    )
    needs_scale = total_tokens > scale_trigger
    if needs_scale:
        scale = scale_trigger / total_tokens
        trim_tool_output = max(200, int(MAX_TOOL_OUTPUT_TOKENS * scale))
        trim_tool_args = max(200, int(MAX_TOOL_ARGS_TOKENS * scale))
        trim_assistant = max(500, int(MAX_ASSISTANT_RESPONSE_TOKENS * scale))
        logger.info(
            "scale trim active: total=%d > trigger=%d, scale=%.2f -> output=%d args=%d assistant=%d",
            total_tokens,
            scale_trigger,
            scale,
            trim_tool_output,
            trim_tool_args,
            trim_assistant,
        )
    else:
        trim_tool_output = MAX_TOOL_OUTPUT_TOKENS
        trim_tool_args = MAX_TOOL_ARGS_TOKENS
        trim_assistant = MAX_ASSISTANT_RESPONSE_TOKENS

    trimmed = _apply_truncation(messages, trim_tool_output, trim_tool_args, trim_assistant)
    return trimmed, total_tokens


def _apply_truncation(  # noqa: C901  — algorithm-intrinsic branches mirror opensource :173-223
    messages: list[Message],
    max_tool_output: int,
    max_tool_args: int,
    max_assistant: int,
    head_ratio: float = 0.7,
) -> list[Message]:
    """Apply per-message head+tail truncation on a deep copy. Opensource :173-223."""
    result = [m.model_copy(deep=True) for m in messages]
    trimmed_count = 0
    for msg in result:
        if msg.role == MessageRole.TOOL and msg.content:
            original = msg.content
            msg.content = truncate_text(original, max_tool_output, head_ratio=head_ratio)
            if msg.content != original:
                trimmed_count += 1
        elif msg.role == MessageRole.ASSISTANT:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = tc.function.get("arguments", "")
                    if args:
                        new_args = truncate_text(args, max_tool_args, head_ratio=head_ratio)
                        if new_args != args:
                            tc.function["arguments"] = new_args
                            trimmed_count += 1
            if msg.content:
                new_content = truncate_text(msg.content, max_assistant, head_ratio=head_ratio)
                if new_content != msg.content:
                    msg.content = new_content
                    trimmed_count += 1
    if trimmed_count > 0:
        logger.info("heuristic trim: truncated %d content fields", trimmed_count)
    return result


def _collect_tool_call_groups(items: list[Message]) -> list[list[int]]:
    """Collect atomic ``assistant-with-tool_calls + following tool responses`` groups. Opensource :225-245.

    Groups must stay together across chunk boundaries — otherwise the LLM compression loses the
    request/response pairing.
    """
    groups: list[list[int]] = []
    i = 0
    while i < len(items):
        msg = items[i]
        if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
            group = [i]
            j = i + 1
            while j < len(items) and items[j].role == MessageRole.TOOL:
                group.append(j)
                j += 1
            groups.append(group)
            i = j
        else:
            i += 1
    return groups


def _calc_group_size(items: list[Message], group: list[int]) -> int:
    """Total tool-content tokens of a tool-call group."""
    return sum(_calc_tool_content_size(items[idx]) for idx in group)


async def _pre_compress_to_list(  # noqa: C901  — algorithm-intrinsic branches mirror opensource :251-363
    original_data: list[Message],
    client: LLMClient,
    *,
    prompt: str | None = None,
) -> list[Message]:
    """Run selective LLM compression on the largest tool-call groups. Opensource :251-363.

    Only the largest groups (by token size descending) are compressed until estimated total drops below
    :data:`PRE_COMPRESS_CHUNK_SIZE`. Compressed chunks run in parallel via :func:`asyncio.gather`; per-chunk
    LLM failure falls back to the originals.
    """
    items = [m.model_copy(deep=True) for m in original_data]

    groups = _collect_tool_call_groups(items)
    if not groups:
        return items

    groups_with_size = [(i, g, _calc_group_size(items, g)) for i, g in enumerate(groups)]
    total_size = sum(s for _, _, s in groups_with_size)
    if total_size <= PRE_COMPRESS_CHUNK_SIZE:
        logger.debug(
            "tool content %d tokens <= %d, no compression",
            total_size,
            PRE_COMPRESS_CHUNK_SIZE,
        )
        return items

    # Pick the largest groups, assume ~90% reduction each, until estimated total drops below threshold.
    groups_by_size = sorted(groups_with_size, key=lambda x: x[2], reverse=True)
    compress_indices: set[int] = set()
    estimated_total: float = float(total_size)
    for idx, _g, size in groups_by_size:
        if estimated_total <= PRE_COMPRESS_CHUNK_SIZE:
            break
        compress_indices.add(idx)
        estimated_total -= size * 0.9

    groups_to_compress = [g for i, g in enumerate(groups) if i in compress_indices]
    logger.debug(
        "selective compression: %d/%d groups, %d total tokens",
        len(groups_to_compress),
        len(groups),
        total_size,
    )

    # Pack selected groups into chunks of PRE_COMPRESS_CHUNK_SIZE.
    chunks: list[list[list[int]]] = []
    current_chunk: list[list[int]] = []
    current_size = 0
    for group in groups_to_compress:
        group_size = _calc_group_size(items, group)
        if current_chunk and current_size + group_size > PRE_COMPRESS_CHUNK_SIZE:
            chunks.append(current_chunk)
            current_chunk = [group]
            current_size = group_size
        else:
            current_chunk.append(group)
            current_size += group_size
    if current_chunk:
        chunks.append(current_chunk)

    chunk_msg_lists: list[list[Message]] = []
    for chunk_groups in chunks:
        chunk_indices = [idx for group in chunk_groups for idx in group]
        chunk_msg_lists.append([items[idx] for idx in chunk_indices])

    # Compress all chunks in parallel.
    results = await asyncio.gather(
        *(_compress_tool_chunk(chunk_msgs, client, prompt=prompt) for chunk_msgs in chunk_msg_lists),
        return_exceptions=True,
    )
    all_compressed: list[Message] = []
    for round_idx, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.warning("chunk %d compression error: %s, keeping originals", round_idx + 1, result)
            all_compressed.extend(chunk_msg_lists[round_idx])
        elif result is not None:
            all_compressed.extend(result)
        else:
            logger.warning("chunk %d compression failed, keeping originals", round_idx + 1)
            all_compressed.extend(chunk_msg_lists[round_idx])

    selected_indices = sorted(idx for group in groups_to_compress for idx in group)
    if len(all_compressed) == len(selected_indices):
        for i, idx in enumerate(selected_indices):
            items[idx] = all_compressed[i]
    else:
        logger.warning(
            "compressed count %d != selected count %d, keeping originals",
            len(all_compressed),
            len(selected_indices),
        )
    return items


async def _compress_tool_chunk(
    messages: list[Message],
    client: LLMClient,
    *,
    prompt: str | None = None,
) -> list[Message] | None:
    """LLM-compress a single chunk of tool-related messages. Opensource :365-396.

    Dumps the chunk to OpenAI wire format for the prompt (EverAlgo-private fields excluded); parses the
    LLM response back into typed :class:`Message` objects, preserving each original message's
    ``timestamp`` (the LLM does not re-emit it). No internal retry (ADR 012). Returns ``None`` on parse
    failure / shape mismatch / per-message validation failure — caller falls back to originals.
    """
    rendered = render_prompt(
        AGENT_TOOL_PRE_COMPRESS_PROMPT,
        prompt,
        messages_json=_dump_messages(messages),
        new_count=len(messages),
    )
    try:
        response = await client.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
        )
        data: Any = json.loads(response.content)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("tool pre-compress JSON parse failed: %s", exc)
        return None

    if not isinstance(data, dict):
        return None
    data_dict = cast("dict[str, Any]", data)
    compressed = data_dict.get("compressed_messages")
    if not isinstance(compressed, list):
        logger.warning("tool pre-compress invalid shape (got non-list, want %d)", len(messages))
        return None
    compressed_list = cast("list[Any]", compressed)  # type: ignore[redundant-cast]
    if len(compressed_list) != len(messages):
        logger.warning("tool pre-compress invalid shape (got %d, want %d)", len(compressed_list), len(messages))
        return None

    rebuilt: list[Message] = []
    for orig, comp in zip(messages, compressed_list, strict=True):
        if not isinstance(comp, dict):
            logger.warning("tool pre-compress non-dict message in compressed list")
            return None
        comp_dict = cast("dict[str, Any]", comp)
        # LLM-compressed messages carry no timestamp; preserve the original message's timestamp so the
        # rebuilt Message validates and downstream ordering / logging stays meaningful.
        merged = {**comp_dict, "timestamp": orig.timestamp}
        try:
            rebuilt.append(Message.model_validate(merged))
        except ValueError as exc:
            logger.warning("tool pre-compress message validation failed: %s", exc)
            return None
    return rebuilt


async def _is_worth_extracting(
    messages_json: str,
    client: LLMClient,
    *,
    prompt: str | None = None,
) -> bool:
    """LLM filter: decide whether the trajectory is worth extracting. Opensource :398-413.

    Returns ``True`` when the trajectory should proceed to the compress step, ``False`` when the LLM
    judges it not worth extracting. Used only when ``_count_tool_call_rounds <= 1`` (multi-round
    trajectories skip this LLM call). ``prompt`` overrides :data:`AGENT_CASE_FILTER_PROMPT` per call;
    ``None`` keeps the built-in default. Defaults to ``True`` when the LLM response is malformed
    (opensource opinionated fallback :413 — fail-open so a parse error never silently drops a memory).
    """
    rendered = render_prompt(AGENT_CASE_FILTER_PROMPT, prompt, messages=messages_json)
    try:
        response = await client.chat(
            messages=[LLMChatMessage(role="user", content=rendered)],
            response_format={"type": "json_object"},
        )
        data: Any = json.loads(response.content)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("filter LLM JSON parse failed: %s; defaulting to extract", exc)
        return True

    if isinstance(data, dict) and "worth_extracting" in data:
        data_dict = cast("dict[str, Any]", data)
        worth = bool(data_dict["worth_extracting"])
        if not worth:
            logger.info("filtered out by LLM: %s", data_dict.get("reason", ""))
        return worth
    return True


async def _compress_experience(
    messages_json: str,
    client: LLMClient,
    *,
    prompt: str | None = None,
) -> dict[str, Any] | None:
    """Single LLM call to extract task_intent / approach / quality_score / key_insight. Opensource :415-449.

    ``prompt`` overrides :data:`AGENT_CASE_COMPRESS_PROMPT` per call; ``None`` keeps the built-in default.
    No internal retry (ADR 012). Returns ``None`` when the LLM emits empty ``task_intent`` or ``approach`` —
    the upstream skip semantics from opensource.
    """
    rendered = render_prompt(AGENT_CASE_COMPRESS_PROMPT, prompt, messages=messages_json)
    response = await client.chat(
        messages=[LLMChatMessage(role="user", content=rendered)],
        response_format={"type": "json_object"},
    )
    try:
        data: Any = json.loads(response.content)
    except json.JSONDecodeError as exc:
        logger.warning("experience compress JSON parse failed: %s", exc)
        return None

    if not isinstance(data, dict) or "task_intent" not in data:
        logger.warning("experience compress missing 'task_intent' field")
        return None
    data_dict = cast("dict[str, Any]", data)
    if not data_dict.get("task_intent"):
        logger.info("LLM returned empty 'task_intent', skipping")
        return None
    if not data_dict.get("approach"):
        logger.warning("LLM returned empty 'approach', skipping")
        return None
    return data_dict


def _clamp_quality_score(value: Any) -> float:
    """Clamp to [0.0, 1.0]; non-numeric falls back to ``0.5``. Opensource :451-459."""
    if value is None:
        return 0.5
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5
