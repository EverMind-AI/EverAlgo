"""AgentCaseExtractor — distil one agent trajectory MemCell into at most one :class:`AgentCase`.

Typed :class:`ConversationItem` objects are used throughout; OpenAI-format dicts are produced only at
the LLM prompt boundary so EverAlgo-private fields never leak into prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, cast

from asgiref.sync import async_to_sync

from everalgo.agent_memory._text import count_tokens, json_default, truncate_text
from everalgo.agent_memory.prompts.case_compress import AGENT_CASE_COMPRESS_PROMPT
from everalgo.agent_memory.prompts.case_filter import AGENT_CASE_FILTER_PROMPT
from everalgo.agent_memory.prompts.tool_pre_compress import AGENT_TOOL_PRE_COMPRESS_PROMPT
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt
from everalgo.types import AgentCase, ChatMessage, ConversationItem, MemCell, ToolCall, ToolCallRequest, ToolCallResult
from everalgo.types._render import render_content

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


__all__ = [
    # Re-exported prompt constants — monkey-patch at startup to override the LLM prompts
    "AGENT_CASE_COMPRESS_PROMPT",
    "AGENT_CASE_FILTER_PROMPT",
    "AGENT_TOOL_PRE_COMPRESS_PROMPT",
    "AgentCaseExtractor",
]


# ── Heuristic constants ─────────────────────────────────────────────────────────────────────────────
# Tunable algorithm-IP thresholds; override at startup via monkey-patch (DESIGN.md §4.6).

PRE_COMPRESS_CHUNK_SIZE = 100_000
HIGH_MESSAGE_COUNT_THRESHOLD = 100  # halve scale_trigger when message count exceeds this
MAX_TOOL_OUTPUT_TOKENS = 1000
MAX_TOOL_ARGS_TOKENS = 800
MAX_ASSISTANT_RESPONSE_TOKENS = 3000
MAX_TASK_INTENT_TOKENS = 300  # head-only truncation after LLM extraction
FILTER_NO_TOOL_MAX_MESSAGES = 4
FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS = 200

# Filter-only view caps. The worth-extracting signals (exploration / correction) are driven by user
# messages and the assistant's action trajectory — which tools, with what args — not by tool-result
# bodies. So the filter sees an aggressively slimmed view (tool outputs shrink to a head just large
# enough to reveal error-vs-success); the fuller `messages_json` still feeds the compress step.
FILTER_TOOL_OUTPUT_TOKENS = 80
FILTER_TOOL_ARGS_TOKENS = 120
FILTER_ASSISTANT_TOKENS = 600


class AgentCaseExtractor:
    """Distil one agent-trajectory MemCell into at most one AgentCase.

    Returns a list of length 0 (filtered out) or 1 (successful extraction).
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        min_tool_call_rounds: int = 3,
        complex_task_tool_call_round_threshold: int = 20,
    ) -> None:
        """Construct the extractor.

        Args:
            llm: LLM client used for the pre-compress / filter / compress steps.
            min_tool_call_rounds: When ``> 0``, trajectories whose tool-call round count is
                strictly below this value are rejected as too simple before any LLM call. Default
                ``3``; set to ``0`` to disable this gate.
            complex_task_tool_call_round_threshold: Tool-call rounds strictly greater than this
                value mark the trajectory as a complex task and fast-pass the LLM filter step.
                Default ``20``.
        """
        self._llm = llm
        self._min_tool_call_rounds = min_tool_call_rounds
        self._complex_task_round_threshold = complex_task_tool_call_round_threshold

    async def aextract(
        self,
        memcell: MemCell,
        *,
        prompt_filter: str | None = None,
        prompt_compress: str | None = None,
        prompt_tool_pre_compress: str | None = None,
    ) -> list[AgentCase]:
        """Run the 11-step pipeline on one MemCell; return ``[]`` (filtered) or ``[AgentCase]``.

        Raises:
            LLMError: From any provider-side LLM call (no internal retry).
        """
        if not memcell.items:
            logger.info("no items on memcell (n_items=0), skipping")
            return []

        client = self._llm

        # Step 1+2: typed view from MemCell, strip system head (drop anything before first user)
        msgs = _strip_before_first_user(memcell.items)

        # Step 3: structural + heuristic pre-filter
        if reason := _should_skip(msgs):
            logger.info("skipping memcell (n_items=%d): %s", len(memcell.items), reason)
            return []

        # Step 3b: minimum-rounds gate — drop trajectories that are too simple to be worth learning
        if self._min_tool_call_rounds > 0:
            rounds = _count_tool_call_rounds(msgs)
            if rounds < self._min_tool_call_rounds:
                logger.info(
                    "skipping memcell — only %d tool-call rounds < min %d",
                    rounds,
                    self._min_tool_call_rounds,
                )
                return []

        # Step 4: heuristic trim (with scale_trigger adaptation)
        msgs, total_tokens = _heuristic_trim(msgs)
        logger.info(
            "memcell pre-trim total_tokens=%d, message_count=%d",
            total_tokens,
            len(msgs),
        )

        # Step 5: over-size bail — re-count only when trim scaled down
        if total_tokens > PRE_COMPRESS_CHUNK_SIZE:
            trimmed_tokens = count_tokens(_dump_messages(msgs))
            if trimmed_tokens > PRE_COMPRESS_CHUNK_SIZE * 2:
                logger.info(
                    "memcell still %d tokens after trim (> %d), skipping",
                    trimmed_tokens,
                    PRE_COMPRESS_CHUNK_SIZE * 2,
                )
                return []

        # Step 6: selective LLM pre-compression of largest tool-call groups
        msgs = await _pre_compress_to_list(msgs, client, prompt=prompt_tool_pre_compress)
        messages_json = _dump_messages(msgs)

        # Step 7: LLM filter — many tool-call rounds auto-passes as complex task; else judge signals
        # on a slimmed view (tool-result bodies shrunk) to cut prompt tokens. `and` short-circuits, so
        # the slim view is built only when the trajectory actually reaches the LLM filter.
        if _count_tool_call_rounds(msgs) <= self._complex_task_round_threshold and not await _is_worth_extracting(
            _dump_messages_for_filter(msgs), _count_user_messages(msgs), client, prompt=prompt_filter
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
                "memcell truncated task_intent to %d tokens",
                MAX_TASK_INTENT_TOKENS,
            )

        case = AgentCase(
            id=uuid.uuid4().hex,
            timestamp=memcell.timestamp,
            task_intent=intent,
            approach=exp.get("approach", "") or "",
            quality_score=_clamp_quality_score(exp.get("quality_score", 0.5)),
            key_insight=exp.get("key_insight", "") or "",
        )
        return [case]

    extract = async_to_sync(aextract)


# ── Serialization helpers (ConversationItem → OpenAI Chat Completions wire format) ─────────────────────────


def _to_openai_dict(msg: ConversationItem) -> dict[str, Any]:
    """Convert a ConversationItem to OpenAI Chat Completions wire-format, stripping EverAlgo-private fields."""
    if isinstance(msg, ChatMessage):
        d: dict[str, Any] = {"role": msg.role, "content": render_content(msg.content)}
    elif isinstance(msg, ToolCallRequest):
        d = {"role": "assistant"}
        if msg.content is not None:
            d["content"] = msg.content
        d["tool_calls"] = [tc.model_dump(mode="json") for tc in msg.tool_calls]
    else:  # ToolCallResult
        d = {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
    return d


def _to_openai_dicts(messages: Sequence[ConversationItem]) -> list[dict[str, Any]]:
    """Vector form of :func:`_to_openai_dict`."""
    return [_to_openai_dict(m) for m in messages]


def _dump_messages(messages: Sequence[ConversationItem]) -> str:
    """JSON-dump messages in OpenAI wire format for LLM prompts and token counting."""
    return json.dumps(_to_openai_dicts(messages), ensure_ascii=False, default=json_default)


def _dump_messages_for_filter(messages: Sequence[ConversationItem]) -> str:
    """Slim JSON view for the worth-extracting filter (step 7).

    Keeps user messages whole and preserves the assistant's action trajectory (tool names + short
    args), but shrinks tool-result bodies to a short head — they barely inform the exploration /
    correction signals, so dropping them cuts filter-prompt tokens sharply on tool-heavy
    trajectories. The compress step still uses the fuller ``messages_json``.
    """
    slim = _apply_truncation(
        messages,
        FILTER_TOOL_OUTPUT_TOKENS,
        FILTER_TOOL_ARGS_TOKENS,
        FILTER_ASSISTANT_TOKENS,
        head_ratio=1.0,
    )
    return _dump_messages(slim)


# ── Module-level helpers ────────────────────────────────────────────────────────────────────────────


def _strip_before_first_user(messages: Sequence[ConversationItem]) -> list[ConversationItem]:
    """Drop everything before the first user message (e.g. system prompts)."""
    for i, msg in enumerate(messages):
        if isinstance(msg, ChatMessage) and msg.role == "user":
            return list(messages[i:])
    return []


def _has_tool_calls(messages: Sequence[ConversationItem]) -> bool:
    """Return ``True`` iff any message is a ToolCallRequest or ToolCallResult."""
    return any(isinstance(msg, (ToolCallRequest, ToolCallResult)) for msg in messages)


def _count_tool_call_rounds(messages: Sequence[ConversationItem]) -> int:
    """Count ToolCallRequest messages (= tool-call rounds; parallel calls within one request count once)."""
    return sum(1 for msg in messages if isinstance(msg, ToolCallRequest))


def _count_user_messages(messages: Sequence[ConversationItem]) -> int:
    """Count user-role ChatMessage entries in the trajectory."""
    return sum(1 for msg in messages if isinstance(msg, ChatMessage) and msg.role == "user")


def _should_skip(messages: Sequence[ConversationItem]) -> str | None:
    """Pre-filter combining structural + heuristic checks. Returns skip reason or ``None``."""
    if not messages:
        return "No messages after stripping system prompts"
    if not any(isinstance(msg, ChatMessage) and msg.role == "user" for msg in messages):
        return "No user messages found"
    has_assistant = any(
        (isinstance(msg, ChatMessage) and msg.role == "assistant") or isinstance(msg, ToolCallRequest)
        for msg in messages
    )
    if not has_assistant:
        return "No assistant messages found"

    last_msg = messages[-1]
    if isinstance(last_msg, ToolCallRequest) or not (
        isinstance(last_msg, ChatMessage) and last_msg.role == "assistant"
    ):
        return "Incomplete agent trajectory (last message is not a final assistant response)"

    has_tools = _has_tool_calls(messages)
    if not has_tools:
        user_count = sum(1 for msg in messages if isinstance(msg, ChatMessage) and msg.role == "user")
        if user_count < 2:
            return "Single-turn conversation without tool calls"

        if len(messages) <= FILTER_NO_TOOL_MAX_MESSAGES:
            return (
                f"No-tool conversation with only {len(messages)} messages (max {FILTER_NO_TOOL_MAX_MESSAGES}), skipping"
            )

        assistant_content = " ".join(
            render_content(msg.content) for msg in messages if isinstance(msg, ChatMessage) and msg.role == "assistant"
        )
        assistant_tokens = count_tokens(assistant_content)
        if assistant_tokens < FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS:
            return (
                f"No-tool conversation with brief assistant response "
                f"({assistant_tokens} tokens < {FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS}), skipping"
            )

    return None


def _calc_tool_content_size(msg: ConversationItem) -> int:
    """Tool-related token count of a single message."""
    if isinstance(msg, ToolCallResult):
        return count_tokens(msg.content or "")
    if isinstance(msg, ToolCallRequest):
        return sum(count_tokens(tc.function.arguments or "") for tc in msg.tool_calls)
    return 0


def _heuristic_trim(messages: Sequence[ConversationItem]) -> tuple[list[ConversationItem], int]:
    """Truncate oversized tool outputs / args / assistant responses; auto-scales caps on overage.

    Returns ``(trimmed_messages, original_total_tokens)``.
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


def _apply_truncation(  # noqa: C901  — algorithm-intrinsic branches
    messages: Sequence[ConversationItem],
    max_tool_output: int,
    max_tool_args: int,
    max_assistant: int,
    head_ratio: float = 0.7,
) -> list[ConversationItem]:
    """Apply per-message head+tail truncation on a deep copy."""
    result = [m.model_copy(deep=True) for m in messages]
    trimmed_count = 0
    for msg in result:
        if isinstance(msg, ToolCallResult) and msg.content:
            original = msg.content
            msg.content = truncate_text(original, max_tool_output, head_ratio=head_ratio)
            if msg.content != original:
                trimmed_count += 1
        elif isinstance(msg, ToolCallRequest):
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if args:
                    new_args = truncate_text(args, max_tool_args, head_ratio=head_ratio)
                    if new_args != args:
                        tc.function.arguments = new_args
                        trimmed_count += 1
            if msg.content:
                new_content = truncate_text(msg.content, max_assistant, head_ratio=head_ratio)
                if new_content != msg.content:
                    msg.content = new_content
                    trimmed_count += 1
        elif isinstance(msg, ChatMessage) and msg.role == "assistant" and isinstance(msg.content, str) and msg.content:
            new_content = truncate_text(msg.content, max_assistant, head_ratio=head_ratio)
            if new_content != msg.content:
                msg.content = new_content
                trimmed_count += 1
    if trimmed_count > 0:
        logger.info("heuristic trim: truncated %d content fields", trimmed_count)
    return result


def _collect_tool_call_groups(items: Sequence[ConversationItem]) -> list[list[int]]:
    """Collect atomic ``ToolCallRequest + following ToolCallResult`` groups (must not be split across chunks)."""
    groups: list[list[int]] = []
    i = 0
    while i < len(items):
        msg = items[i]
        if isinstance(msg, ToolCallRequest):
            group = [i]
            j = i + 1
            while j < len(items) and isinstance(items[j], ToolCallResult):
                group.append(j)
                j += 1
            groups.append(group)
            i = j
        else:
            i += 1
    return groups


def _calc_group_size(items: Sequence[ConversationItem], group: list[int]) -> int:
    """Total tool-content tokens of a tool-call group."""
    return sum(_calc_tool_content_size(items[idx]) for idx in group)


async def _pre_compress_to_list(  # noqa: C901  — algorithm-intrinsic branches
    original_data: Sequence[ConversationItem],
    client: LLMClient,
    *,
    prompt: str | None = None,
) -> list[ConversationItem]:
    """Selectively compress the largest tool-call groups via parallel LLM calls.

    Targets the largest groups until estimated total drops below ``PRE_COMPRESS_CHUNK_SIZE``.

    Raises:
        LLMError: Propagated from a chunk compression LLM call.
        json.JSONDecodeError: If a chunk LLM response is not valid JSON.
        ValueError: If a chunk returns None (shape mismatch) or the total compressed count mismatches.
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

    chunk_msg_lists: list[list[ConversationItem]] = []
    for chunk_groups in chunks:
        chunk_indices = [idx for group in chunk_groups for idx in group]
        chunk_msg_lists.append([items[idx] for idx in chunk_indices])

    # Compress all chunks in parallel.
    results = await asyncio.gather(
        *(_compress_tool_chunk(chunk_msgs, client, prompt=prompt) for chunk_msgs in chunk_msg_lists),
    )
    all_compressed: list[ConversationItem] = []
    for round_idx, result in enumerate(results):
        if result is None:
            raise ValueError(f"chunk {round_idx + 1} compression returned None (shape mismatch or invalid message)")
        all_compressed.extend(result)

    selected_indices = sorted(idx for group in groups_to_compress for idx in group)
    if len(all_compressed) == len(selected_indices):
        for i, idx in enumerate(selected_indices):
            items[idx] = all_compressed[i]
    else:
        raise ValueError(f"compressed count {len(all_compressed)} != selected count {len(selected_indices)}")
    return items


async def _compress_tool_chunk(
    messages: Sequence[ConversationItem],
    client: LLMClient,
    *,
    prompt: str | None = None,
) -> list[ConversationItem] | None:
    """LLM-compress one chunk of tool messages; preserves original timestamps.

    Returns ``None`` for structural validation failures (wrong shape, non-dict messages) — those are
    business validation signals, not parse errors. Parse and LLM errors propagate as exceptions.

    Raises:
        LLMError: Propagated from the underlying LLM client call.
        json.JSONDecodeError: If the LLM response is not valid JSON.
    """
    rendered = render_prompt(
        AGENT_TOOL_PRE_COMPRESS_PROMPT,
        prompt,
        messages_json=_dump_messages(messages),
        new_count=len(messages),
    )
    compressed_list = await _call_llm_for_tool_pre_compress(client, rendered)

    if len(compressed_list) != len(messages):
        logger.warning("tool pre-compress invalid shape (got %d, want %d)", len(compressed_list), len(messages))
        return None

    rebuilt: list[ConversationItem] = []
    for orig, comp in zip(messages, compressed_list, strict=True):
        if not isinstance(comp, dict):
            logger.warning("tool pre-compress non-dict message in compressed list")
            return None
        # isinstance(comp, dict) is asserted above; cast to restore key/value types for dict spread.
        # LLM-compressed messages carry no timestamp; preserve the original message's timestamp so the
        # rebuilt ConversationItem validates and downstream ordering / logging stays meaningful.
        merged: dict[str, Any] = {**cast("dict[str, Any]", comp), "timestamp": orig.timestamp}
        try:
            rebuilt.append(_openai_dict_to_agent_item(merged, orig.timestamp))
        except (ValueError, KeyError) as exc:
            logger.warning("tool pre-compress message validation failed: %s", exc, exc_info=True)
            return None
    return rebuilt


def _openai_dict_to_agent_item(d: dict[str, Any], timestamp: int) -> ConversationItem:
    """Convert OpenAI-shaped dict back to ConversationItem, inferring discriminator from role / tool_calls."""
    role = d.get("role")
    if role == "tool":
        return ToolCallResult(
            kind="tool_result",
            tool_call_id=d["tool_call_id"],
            content=d.get("content", "") or "",
            timestamp=timestamp,
        )
    if role == "assistant" and d.get("tool_calls"):
        return ToolCallRequest(
            kind="tool_call",
            tool_calls=[ToolCall.model_validate(tc) for tc in d["tool_calls"]],
            content=d.get("content"),
            timestamp=timestamp,
            sender_id=d.get("sender_id") or "assistant",
        )
    # else: ChatMessage (user / assistant final response)
    return ChatMessage(
        kind="text",
        id=d.get("id") or "",
        role=role if role in ("user", "assistant") else "assistant",  # type: ignore[arg-type]
        content=d.get("content") or "",
        timestamp=timestamp,
        sender_id=d.get("sender_id") or role or "",
    )


async def _is_worth_extracting(
    messages_json: str,
    user_message_count: int,
    client: LLMClient,
    *,
    prompt: str | None = None,
) -> bool:
    """LLM filter: ``True`` only if at least one signal (exploration / user correction) is True.

    Used only when the tool-call round count is at or below the extractor's configured threshold;
    higher-volume trajectories pass automatically as complex tasks (handled upstream, not here).
    Missing or None signal fields are treated as False — precision over recall, skip when uncertain.

    ``has_user_correction`` is additionally validated against ``user_message_count``: a real
    correction requires at least two user messages (initial request + corrective follow-up). If the
    LLM claims a correction on a single-user-message trajectory, the signal is rejected as
    hallucination.

    Raises:
        LLMError: Propagated from the underlying LLM client call.
        json.JSONDecodeError: If the LLM response is not valid JSON.
    """
    rendered = render_prompt(AGENT_CASE_FILTER_PROMPT, prompt, messages=messages_json)
    filter_data = await _call_llm_for_case_filter(client, rendered)
    has_exploration = filter_data.get("has_exploration") is True
    has_user_correction = filter_data.get("has_user_correction") is True and user_message_count >= 2
    if filter_data.get("has_user_correction") is True and user_message_count < 2:
        logger.info(
            "rejecting has_user_correction=True on single-user-message trajectory (likely hallucination): %s",
            filter_data.get("reason", ""),
        )
    worth = has_exploration or has_user_correction
    if not worth:
        logger.info("filtered out by LLM: %s", filter_data.get("reason", ""))
    return worth


async def _compress_experience(
    messages_json: str,
    client: LLMClient,
    *,
    prompt: str | None = None,
) -> dict[str, Any] | None:
    """Extract task_intent / approach / quality_score / key_insight via one LLM call (no retry).

    Returns ``None`` when the LLM emits empty ``task_intent`` or ``approach`` — those are business
    validation failures (the trajectory isn't worth compressing), not parse errors.

    Raises:
        LLMError: Propagated from the underlying LLM client call.
        json.JSONDecodeError: If the LLM response is not valid JSON.
    """
    rendered = render_prompt(AGENT_CASE_COMPRESS_PROMPT, prompt, messages=messages_json)
    compress_data = await _call_llm_for_case_compress(client, rendered)
    if not compress_data.get("task_intent"):
        logger.info("LLM returned empty 'task_intent', skipping")
        return None
    if not compress_data.get("approach"):
        logger.warning("LLM returned empty 'approach', skipping")
        return None
    return compress_data


# ---------------------------------------------------------------------------
# LLM callsites — brace-balanced JSON extraction + 5-retry (mirror b150b32).
# ---------------------------------------------------------------------------


async def _call_llm_for_tool_pre_compress(llm: LLMClient, rendered: str) -> list[Any]:
    """Call LLM for tool pre-compression and return validated compressed_messages list.

    Uses brace-balanced extraction because ``compressed_messages`` is nested (list of dicts).

    Raises:
        ValueError: If no JSON found or ``compressed_messages`` key missing / not a list.
    """
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    json_str = _extract_json_object(text)
    data = json.loads(json_str)
    if "compressed_messages" not in data:
        raise ValueError(f"Tool pre-compress response missing 'compressed_messages': {list(data.keys())!r}")
    compressed = data["compressed_messages"]
    if not isinstance(compressed, list):
        raise ValueError(f"compressed_messages must be a list, got {type(compressed).__name__}: {compressed!r}")  # noqa: TRY004
    return cast("list[Any]", compressed)  # type: ignore[redundant-cast]


async def _call_llm_for_case_filter(llm: LLMClient, rendered: str) -> dict[str, Any]:
    """Call LLM for case filtering and return the parsed dict (has_exploration / has_user_correction / reason).

    The response is nominally flat but uses brace-balanced extraction for safety.
    """
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    json_str = _extract_json_object(text)
    result: dict[str, Any] = json.loads(json_str)
    return result


async def _call_llm_for_case_compress(llm: LLMClient, rendered: str) -> dict[str, Any]:
    """Call LLM for case compression and return validated dict with task_intent + approach + quality_score.

    Raises:
        ValueError: If no JSON found.
    """
    response = await llm.chat(messages=[LLMChatMessage(role="user", content=rendered)])
    text = response.content
    json_str = _extract_json_object(text)
    result: dict[str, Any] = json.loads(json_str)
    return result


def _extract_json_object(text: str) -> str:
    """First balanced {{...}} block in text (brace-balanced parser for nested JSON)."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in case LLM response: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Unbalanced JSON in case LLM response: {text[:200]!r}")


def _clamp_quality_score(value: Any) -> float:
    """Clamp to [0.0, 1.0]; non-numeric defaults to ``0.5``."""
    if value is None:
        return 0.5
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5
