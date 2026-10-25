"""What the model is shown: a system message of rules, then a user message holding the conversation or episode text.

The episode request is the EverAlgo LoCoMo reference run's (everalgo commit d9fe22e) in the operators' layout:
its conversation start time and its pseudo-JSON message blocks, laid out as that run laid them out. Every time is the
shared prompt shape with its weekday (``2024-03-14 15:00 UTC (Thursday)``, :func:`everalgo.utils.format_time`), in
the caller's zone.
"""

from __future__ import annotations

import logging
from datetime import UTC
from typing import TYPE_CHECKING

from everalgo.llm import LLMMessage
from everalgo.user_memory._language import (
    PARTICIPANT_LANGUAGE_RULE,
    SOURCE_TEXT_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
from everalgo.user_memory._render import chat_messages
from everalgo.user_memory.prompts.en.episode import EPISODE_USER_PROMPT, assemble_system_prompt
from everalgo.user_memory.prompts.en.episode_summary import EPISODE_SUMMARY_USER_PROMPT
from everalgo.user_memory.prompts.en.episode_summary import assemble_system_prompt as assemble_summary_system_prompt
from everalgo.utils import (
    InvalidInputError,
    count_tokens,
    format_time,
    render_prompt,
    speaker,
    truncate_text,
)

if TYPE_CHECKING:
    from datetime import tzinfo

    from everalgo.types import ChatMessage, MemCell

logger = logging.getLogger("everalgo.user_memory.episode")

#: The conversation budget of one episode extraction, in ``o200k_base`` tokens.
#:
#: Boundary detection caps a memcell at 65_536 tokens (``DEFAULT_HARD_TOKEN_LIMIT``), so a cell that
#: follows the contract already fits. This is the extraction side's own guard: a memcell can arrive
#: from a backfill, an import or a raised boundary limit, and a request past the model's context is
#: rejected by every backend. 48_000 leaves room for the system message and the reply inside that
#: 65_536. Without it a 756 KB cell reached ~3.8M tokens and all four fallback backends answered
#: ``prompt_too_long``, each attempt taking 15-120 s (2026-10-25).
MAX_CONVERSATION_TOKENS = 48_000

_HEAD_RATIO = 0.6


def _render_messages(
    *,
    conversation: str,
    custom_instructions: str | None,
    output_language: OutputLanguage | None,
) -> list[LLMMessage]:
    """The ``[system, user]`` pair that asks for the episode: the rules and call-level blocks, then the conversation.

    Args:
        conversation: The user message, as :func:`_render_conversation` renders it.
        custom_instructions: The caller's instructions. Placed verbatim in ``<custom_instructions>``, never rendered as
            a template; None or a blank string sends the reference run's default principles there instead.
        output_language: The language the caller named, sent as ``<output_language>``; None sends the participant rule
            there, so the episode is written in the language the participants use.
    """
    language_rule = build_language_rule(output_language, fallback=PARTICIPANT_LANGUAGE_RULE)
    system = assemble_system_prompt(custom_instructions=custom_instructions, language_rule=language_rule)
    return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=conversation)]


def _render_conversation(memcell: MemCell, *, tz: tzinfo = UTC) -> str:
    """The user message: the conversation start time and the ChatMessage items, as the reference run laid them out.

    The start time is the first item's. Each message with text is the reference run's pseudo-JSON block, its values
    unquoted and indented as that run's template indented them; a message with no text is dropped. Times are in ``tz``.

    The blocks are bounded by :data:`MAX_CONVERSATION_TOKENS`; a conversation that exceeds it keeps its head and tail
    and states how many messages were dropped, so a cell that arrived past the boundary limit still reaches the model
    instead of being rejected by every backend.

    Raises:
        InvalidInputError: If a message with text carries no timestamp (0). The error names the message's id only,
            never its text.
    """
    blocks = [_message_block(message, tz=tz) for message in chat_messages(memcell) if message.content]
    kept, omitted = _cap_blocks(blocks)
    if omitted:
        logger.warning(
            "episode conversation capped: %d of %d messages omitted, %d blocks sent, limit=%d tokens",
            omitted,
            len(blocks),
            len(kept),
            MAX_CONVERSATION_TOKENS,
        )
    conversation = "\n".join(kept).removeprefix("\n")
    if count_tokens(conversation) > MAX_CONVERSATION_TOKENS:
        # One block can outweigh the whole budget (a pasted transcript, a 40 KB tool result). The cap above
        # counts whole blocks, so it cannot cut that one; this second pass keeps its head and tail.
        logger.warning("episode conversation block over budget: truncating %d characters", len(conversation))
        conversation = truncate_text(conversation, MAX_CONVERSATION_TOKENS, head_ratio=_HEAD_RATIO)
    return render_prompt(
        EPISODE_USER_PROMPT,
        None,
        conversation_start_time=format_time(memcell.items[0].timestamp, tz=tz, weekday=True),
        conversation=conversation,
    )


def _message_block(message: ChatMessage, *, tz: tzinfo = UTC) -> str:
    """One message as the reference run laid it out: ``{"timestamp": ..., "speaker": ..., "content": ...}`` unquoted.

    Raises:
        InvalidInputError: If the message has no timestamp (0), which would render as 1970-01-01.
    """
    if not message.timestamp:
        raise InvalidInputError(
            f"chat message {message.id!r} has no timestamp: every ChatMessage must carry a positive "
            "epoch-millisecond timestamp; validate or fill it at ingress before calling the operator"
        )
    return f"""
                {{
                    "timestamp": {format_time(message.timestamp, tz=tz, weekday=True)},
                    "speaker": {speaker(message)},
                    "content": {message.content}
                }}"""


def _cap_blocks(blocks: list[str]) -> tuple[list[str], int]:
    """Keep the head and the tail of ``blocks`` inside :data:`MAX_CONVERSATION_TOKENS`.

    The middle goes rather than one end: an episode's opening establishes who is talking and about what, and
    its last messages carry the outcome the summary is written from. The gap is a rendered line rather than a
    comment outside the conversation, so the model can record it in the episode.

    Returns:
        The blocks to render, and how many were dropped (0 when the conversation already fits).
    """
    if len(blocks) < 3:
        return blocks, 0
    counts = [count_tokens(block) for block in blocks]
    if sum(counts) + len(blocks) - 1 <= MAX_CONVERSATION_TOKENS:
        return blocks, 0
    head_budget = int(MAX_CONVERSATION_TOKENS * _HEAD_RATIO)
    tail_budget = MAX_CONVERSATION_TOKENS - head_budget
    head_end = 0
    used = 0
    while head_end < len(blocks) and used + counts[head_end] <= head_budget:
        used += counts[head_end]
        head_end += 1
    tail_start = len(blocks)
    used_tail = 0
    while tail_start > head_end and used_tail + counts[tail_start - 1] <= tail_budget:
        used_tail += counts[tail_start - 1]
        tail_start -= 1
    omitted = tail_start - head_end
    if omitted <= 0:
        return blocks, 0
    marker = f"\n                [... {omitted} messages omitted: the conversation exceeded the extraction budget ...]"
    return [*blocks[:head_end], marker, *blocks[tail_start:]], omitted


def _render_summary_messages(
    *,
    episode_text: str,
    custom_instructions: str | None,
    output_language: OutputLanguage | None,
) -> list[LLMMessage]:
    """The ``[system, user]`` pair that asks for a summary: the rules and call-level blocks, then the episode text.

    Args:
        episode_text: The caller's text, placed verbatim in the user message's ``<episode_text>`` element and never
            rendered as a template.
        custom_instructions: The caller's instructions. Placed verbatim in ``<custom_instructions>``, never rendered as
            a template; None or a blank string leaves the block out.
        output_language: The language the caller named, or None to have the model follow the text's own.
    """
    system = assemble_summary_system_prompt(
        custom_instructions=custom_instructions,
        language_rule=build_language_rule(output_language, fallback=SOURCE_TEXT_LANGUAGE_RULE),
    )
    user = render_prompt(EPISODE_SUMMARY_USER_PROMPT, None, episode_text=episode_text)
    return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]
