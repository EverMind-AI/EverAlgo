"""The reply models, the shapes the prompts' output skeletons show, and the LLM callsites held to them.

``_EpisodeReply`` answers the episode prompt and ``_SummaryReply`` the episode-summary prompt. Each callsite makes one
call and validates the reply; there is no retry here, the host decides (``docs/concepts/llm-contract.md`` §4, §5).
"""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from everalgo.llm import LLMMessage, LLMOutputError, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

_EPISODE_LLM_TIMEOUT_SECONDS = 25.0
_SPEAKER_CONTENT_RE = re.compile(r'"speaker":\s*([^\n,]+),\s*\n\s*"content":\s*(.+?)\n\s*\}', re.DOTALL)


class _EpisodeReply(StrictOutputModel):
    """Episode reply: the ``{"title", "content"}`` skeleton of the episode prompt, in order (the reference run's)."""

    title: str
    content: str


class _SummaryReply(StrictOutputModel):
    """Summary reply: the ``{"summary"}`` skeleton of the episode-summary prompt."""

    summary: str


# The fields stored as text, which a blank value leaves without an honest stand-in.
_STORED_TEXT_FIELDS = ("title", "content")


async def _call_llm_for_episode(llm: LLMClient, messages: list[LLMMessage]) -> _EpisodeReply:
    """Request an episode reply and return it validated.

    Falls back deterministically to the rendered conversation messages when the LLM call exceeds
    ``_EPISODE_LLM_TIMEOUT_SECONDS``.

    Raises:
        LLMOutputError: If the reply is not a valid episode output, or leaves ``title`` or ``content`` blank.
    """
    try:
        async with asyncio.timeout(_EPISODE_LLM_TIMEOUT_SECONDS):
            text = await llm.achat(messages=messages, response_format=_EpisodeReply)
    except TimeoutError:
        return _deterministic_episode_from_messages(messages)
    reply = parse_reply(_EpisodeReply, text, operator="episode")
    _check_fields_written(reply, _STORED_TEXT_FIELDS, operator="episode")
    return reply


def _deterministic_episode_from_messages(messages: list[LLMMessage]) -> _EpisodeReply:
    """Build a deterministic episode reply from the rendered user message when the LLM call times out."""
    rendered = str(messages[-1].content) if messages else ""
    matches = _SPEAKER_CONTENT_RE.findall(rendered)
    if matches:
        lines = [f"{spk.strip()}: {cnt.strip()}" for spk, cnt in matches if cnt.strip()]
        body = "\n".join(lines).strip()
        first_line = matches[0][1].strip().splitlines()[0][:80] if matches[0][1].strip() else "Session Note"
        title = first_line or "Session Note"
    else:
        body = rendered[-1500:].strip() or "Session conversation recorded."
        title = "Session Note"
    return _EpisodeReply(title=title, content=body[:4000])


async def _call_llm_for_summary(llm: LLMClient, messages: list[LLMMessage]) -> _SummaryReply:
    """Request a summary reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid summary output, or leaves ``summary`` blank.
    """
    text = await llm.achat(messages=messages, response_format=_SummaryReply)
    reply = parse_reply(_SummaryReply, text, operator="episode summary")
    _check_fields_written(reply, ("summary",), operator="episode summary")
    return reply


def _check_fields_written(reply: StrictOutputModel, fields: tuple[str, ...], *, operator: str) -> None:
    """Reject a reply that leaves one of ``fields``, each stored as text, blank (llm-contract §4.1).

    Raises:
        LLMOutputError: Naming ``operator`` and each blank field, never the reply's content.
    """
    blank = [name for name in fields if not getattr(reply, name).strip()]
    if blank:
        reasons = "; ".join(f"{name}: empty after stripping whitespace" for name in blank)
        raise LLMOutputError(f"{operator} reply leaves fields blank: {reasons}")
