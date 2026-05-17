"""LLM wire types — ChatMessage, ChatResponse, Usage."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    """Single chat-style message turn (user / assistant / system)."""

    role: Literal["system", "user", "assistant"]
    content: str

    model_config = ConfigDict(extra="ignore")


class Usage(BaseModel):
    """Token usage from a single LLM call. Fields are ``int | None`` — some backends omit usage entirely."""

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
        description="Provider-specific original payload; populated only when the provider opts in (debug mode).",
    )
