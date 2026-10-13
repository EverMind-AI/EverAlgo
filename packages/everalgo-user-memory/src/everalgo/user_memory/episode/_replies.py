"""The episode reply model, the shape the prompt's output skeleton shows, and the LLM callsite held to it.

The callsite makes one call and validates the reply; there is no retry here, the host decides
(``docs/concepts/llm-contract.md`` §4, §5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.llm import LLMMessage, LLMOutputError, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


class _EpisodeReply(StrictOutputModel):
    """Episode reply: the ``{"content", "summary", "title"}`` skeleton of the episode prompt, in order."""

    content: str
    summary: str
    title: str


# The fields stored as text, which a blank value leaves without an honest stand-in.
_STORED_TEXT_FIELDS = ("content", "summary", "title")


async def _call_llm_for_episode(llm: LLMClient, messages: list[LLMMessage]) -> _EpisodeReply:
    """Request an episode reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid episode output, or leaves ``content``, ``summary`` or
            ``title`` blank.
    """
    text = await llm.achat(messages=messages, response_format=_EpisodeReply)
    reply = parse_reply(_EpisodeReply, text, operator="episode")
    _check_fields_written(reply)
    return reply


def _check_fields_written(reply: _EpisodeReply) -> None:
    """Reject a reply that leaves a stored text field blank (llm-contract §4.1).

    Raises:
        LLMOutputError: Naming each blank field, never the reply's content.
    """
    blank = [name for name in _STORED_TEXT_FIELDS if not getattr(reply, name).strip()]
    if blank:
        reasons = "; ".join(f"{name}: empty after stripping whitespace" for name in blank)
        raise LLMOutputError(f"episode reply leaves fields blank: {reasons}")
