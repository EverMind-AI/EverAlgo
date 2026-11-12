"""Each boundary call's reply model, the shape its prompt's output skeleton shows, and the LLM callsites held to it.

Each callsite makes one call and validates the reply against its model; there is no retry here, the host decides
(``docs/concepts/llm-contract.md`` §4, §5).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from everalgo.llm import LLMMessage, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

_UNESCAPED_BACKSLASH_RE = re.compile(r'\\(?![/\"\\bfnrt]|u[0-9a-fA-F]{4})')
_PLAIN_NO_BOUNDARY_RE = re.compile(r"\b(no\s+boundaries|no\s+boundary|single\s+episode|same\s+episode)\b", re.IGNORECASE)


class Reply(StrictOutputModel):
    """Strict internal-cut response shared by both variants."""

    reasoning: str
    boundaries: list[int]


class _StepReply(StrictOutputModel):
    """Step reply: the ``{"reasoning", "should_end"}`` skeleton of the step prompt; any other key is rejected.

    Only ``should_end`` decides a cut, so only it is required. ``reasoning`` defaults to ``""``, as the parser the
    two-field reply was measured with filled it: a deliberate exception to the contract's no-default rule
    (``docs/concepts/llm-contract.md`` §2.2), covered by the prompt-only path.
    """

    reasoning: str = ""
    should_end: bool


async def _call_llm_for_batch(llm: LLMClient, messages: list[LLMMessage]) -> Reply:
    """Request a batch reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid batch output.
    """
    text = await llm.achat(messages=messages, response_format=Reply)
    if "{" not in text and _PLAIN_NO_BOUNDARY_RE.search(text):
        return Reply(reasoning="", boundaries=[])
    return parse_reply(Reply, _UNESCAPED_BACKSLASH_RE.sub(r"\\\\", text), operator="boundary batch")


async def _call_llm_for_step(llm: LLMClient, messages: list[LLMMessage]) -> _StepReply:
    """Request a step reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid step output.
    """
    text = await llm.achat(messages=messages, response_format=_StepReply)
    return parse_reply(_StepReply, text, operator="boundary step")
