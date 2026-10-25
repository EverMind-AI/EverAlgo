"""What the model is shown: a system message of rules, then a user message holding the conversation or episode text."""

from __future__ import annotations

import logging
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
from everalgo.utils import count_tokens, render_chat_line, render_prompt, truncate_text

if TYPE_CHECKING:
    from everalgo.types import MemCell

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
            a template; None or a blank string leaves the block out.
        output_language: The language the caller named, or None to have the model follow the participants.
    """
    system = assemble_system_prompt(
        custom_instructions=custom_instructions,
        language_rule=build_language_rule(output_language, fallback=PARTICIPANT_LANGUAGE_RULE),
    )
    return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=conversation)]


def _render_conversation(memcell: MemCell) -> str:
    """The user message: the ChatMessage items as lines in a ``<conversation>`` element.

    ``count`` is the number of lines rendered. Each line is :func:`~everalgo.utils.render_chat_line`'s,
    ``[<time> (<weekday>)] <speaker>: <text>``: the model resolves relative dates against each message's own
    timestamp. A message with no text is dropped.

    The lines are bounded by :data:`MAX_CONVERSATION_TOKENS`; a conversation that exceeds it keeps its head and
    tail and states how many messages were dropped, so a cell that arrived past the boundary limit still
    reaches the model instead of being rejected by every backend.

    Raises:
        InvalidInputError: If a message with text carries no timestamp (0). The error names the message's id only, never its
            text.
    """
    lines = [render_chat_line(message) for message in chat_messages(memcell) if message.content]
    kept, omitted = _cap_lines(lines)
    if omitted:
        logger.warning(
            "episode conversation capped: %d of %d messages omitted, %d lines sent, limit=%d tokens",
            omitted,
            len(lines),
            len(kept),
            MAX_CONVERSATION_TOKENS,
        )
    conversation = "\n".join(kept)
    if count_tokens(conversation) > MAX_CONVERSATION_TOKENS:
        # One line can outweigh the whole budget (a pasted transcript, a 40 KB tool result). The cap above
        # counts whole lines, so it cannot cut that one; this second pass keeps its head and tail.
        logger.warning("episode conversation line over budget: truncating %d characters", len(conversation))
        conversation = truncate_text(conversation, MAX_CONVERSATION_TOKENS, head_ratio=_HEAD_RATIO)
    return render_prompt(EPISODE_USER_PROMPT, None, count=len(kept), conversation=conversation)


def _cap_lines(lines: list[str]) -> tuple[list[str], int]:
    """Keep the head and the tail of ``lines`` inside :data:`MAX_CONVERSATION_TOKENS`.

    The middle goes rather than one end: an episode's opening establishes who is talking and about what, and
    its last messages carry the outcome the summary is written from. Replacing the middle with a rendered
    line, rather than a comment outside the conversation, keeps it material the model records the gap in,
    and keeps ``count`` the number of lines the model actually receives.

    Returns:
        The lines to render, and how many were dropped (0 when the conversation already fits).
    """
    if len(lines) < 3:
        return lines, 0
    counts = [count_tokens(line) for line in lines]
    if sum(counts) + len(lines) - 1 <= MAX_CONVERSATION_TOKENS:
        return lines, 0
    head_budget = int(MAX_CONVERSATION_TOKENS * _HEAD_RATIO)
    tail_budget = MAX_CONVERSATION_TOKENS - head_budget
    head_end = 0
    used = 0
    while head_end < len(lines) and used + counts[head_end] <= head_budget:
        used += counts[head_end]
        head_end += 1
    tail_start = len(lines)
    used_tail = 0
    while tail_start > head_end and used_tail + counts[tail_start - 1] <= tail_budget:
        used_tail += counts[tail_start - 1]
        tail_start -= 1
    omitted = tail_start - head_end
    if omitted <= 0:
        return lines, 0
    marker = f"[... {omitted} messages omitted: the conversation exceeded the extraction budget ...]"
    return [*lines[:head_end], marker, *lines[tail_start:]], omitted


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
