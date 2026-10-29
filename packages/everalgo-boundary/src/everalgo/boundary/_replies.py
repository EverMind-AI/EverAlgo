"""Each boundary call's reply model, the shape its prompt's output skeleton shows, and the LLM callsites held to it.

Each callsite makes one call and validates the reply against its model; there is no retry here, the host decides
(``docs/concepts/llm-contract.md`` §4, §5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.llm import LLMMessage, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


class Reply(StrictOutputModel):
    """Strict internal-cut response shared by both variants."""

    reasoning: str
    boundaries: list[int]


class _StepReply(StrictOutputModel):
    """Step reply: the ``{"reasoning", "should_end", "confidence", "topic_summary"}`` skeleton of the step prompt."""

    reasoning: str
    should_end: bool
    confidence: float
    topic_summary: str


async def _call_llm_for_batch(llm: LLMClient, messages: list[LLMMessage]) -> Reply:
    """Request a batch reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid batch output.
    """
    text = await llm.achat(messages=messages, response_format=Reply)
    return parse_reply(Reply, text, operator="boundary batch")


async def _call_llm_for_step(llm: LLMClient, messages: list[LLMMessage]) -> _StepReply:
    """Request a step reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid step output.
    """
    text = await llm.achat(messages=messages, response_format=_StepReply)
    return parse_reply(_StepReply, text, operator="boundary step")
