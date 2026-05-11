"""LLM wire types — chat-style messages, response, token usage.

These are the on-the-wire data contracts a caller sees when invoking
``LLMClient.chat``. They mirror the OpenAI Chat Completions API closely so
the openai_compat provider can pass through values with minimal translation.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """Single chat-style message turn.

    The ``role`` set is intentionally narrow (3 values). ``tool`` and
    multimodal ``content`` blocks are out of EPISODE scope; adding them later
    is a SemVer minor bump (extending a Literal is a backward-compatible
    structural widening).
    """

    role: Literal["system", "user", "assistant"]
    content: str

    model_config = ConfigDict(extra="ignore")


class Usage(BaseModel):
    """Token usage from a single LLM call.

    Both fields are ``int | None`` because some self-hosted / OpenAI-compatible
    backends do not return ``usage`` in the response. ``None`` semantically
    distinguishes "missing data" from "zero tokens used".
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatResponse(BaseModel):
    """Structured response from a single LLM chat call."""

    content: str
    model: str
    usage: Usage | None = None
    finish_reason: Literal["stop", "length", "content_filter"] | None = None
    raw: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Provider-specific original response payload. Populated only when "
            "the provider implementation explicitly opts in (e.g. debug mode). "
            "Production callers should rely on the structured "
            "``content`` / ``usage`` / ``finish_reason`` fields and not "
            "depend on ``raw`` being non-None."
        ),
    )
