"""What the model is shown: a system message of rules, then a user message holding the conversation or episode text.

The episode request is the EverAlgo LoCoMo reference run's (everalgo commit d9fe22e) in the operators' layout:
its conversation start time and its pseudo-JSON message blocks, laid out as that run laid them out. Every time is the
shared prompt shape with its weekday (``2024-03-14 15:00 UTC (Thursday)``, :func:`everalgo.utils.format_time`), in
the caller's zone.
"""

from __future__ import annotations

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
from everalgo.utils import InvalidInputError, format_time, render_prompt, speaker

if TYPE_CHECKING:
    from datetime import tzinfo

    from everalgo.types import ChatMessage, MemCell


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
        output_language: The language the caller named, sent as ``<output_language>``; None sends no language rule,
            as the reference run sent none.
    """
    language_rule = (
        None if output_language is None else build_language_rule(output_language, fallback=PARTICIPANT_LANGUAGE_RULE)
    )
    system = assemble_system_prompt(custom_instructions=custom_instructions, language_rule=language_rule)
    return [LLMMessage(role="system", content=system), LLMMessage(role="user", content=conversation)]


def _render_conversation(memcell: MemCell, *, tz: tzinfo = UTC) -> str:
    """The user message: the conversation start time and the ChatMessage items, as the reference run laid them out.

    The start time is the first item's. Each message with text is the reference run's pseudo-JSON block, its values
    unquoted and indented as that run's template indented them; a message with no text is dropped. Times are in ``tz``.

    Raises:
        InvalidInputError: If a message with text carries no timestamp (0). The error names the message's id only,
            never its text.
    """
    blocks = [_message_block(message, tz=tz) for message in chat_messages(memcell) if message.content]
    return render_prompt(
        EPISODE_USER_PROMPT,
        None,
        conversation_start_time=format_time(memcell.items[0].timestamp, tz=tz, weekday=True),
        conversation="\n".join(blocks).removeprefix("\n"),
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
