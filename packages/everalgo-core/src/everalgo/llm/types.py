"""LLM wire types — chat-style messages, response, token usage.

These are the on-the-wire data contracts a caller sees when invoking ``LLMClient.chat``. They mirror the
OpenAI Chat Completions API closely so the openai_compat provider can pass through values with minimal
translation.
"""

from __future__ import annotations

import base64
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TextPart(BaseModel):
    """Text segment inside a multimodal ``content`` array.

    Serialises to ``{"type": "text", "text": "..."}`` — the OpenAI / OpenRouter wire form.
    """

    type: Literal["text"] = "text"
    text: str

    model_config = ConfigDict(extra="ignore")


class ImageUrlInner(BaseModel):
    """Inner object of an ``image_url`` content part."""

    url: str
    detail: Literal["low", "high", "auto"] | None = None

    model_config = ConfigDict(extra="ignore")


class ImageUrlPart(BaseModel):
    """File-by-data-URI content part.

    Serialises to ``{"type": "image_url", "image_url": {"url": "...", "detail": ...}}``. Named "image_url" for
    OpenAI / OpenRouter wire compatibility, but the data URI may carry any MIME the upstream model accepts
    (``application/pdf``, ``audio/wav``, ``image/png`` ...); OpenRouter + Gemini routes all of them through
    this single part type.
    """

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlInner

    model_config = ConfigDict(extra="ignore")


ContentPart = Annotated[TextPart | ImageUrlPart, Field(discriminator="type")]
"""Discriminated union of content parts. Add new arms as providers gain native types."""


def image_url_part_from_bytes(data: bytes, mime: str) -> ImageUrlPart:
    """Build an ``ImageUrlPart`` from raw bytes by base64-encoding into a data URI.

    Parameters
    ----------
    data : bytes
        Raw file payload (image / pdf / audio bytes).
    mime : str
        MIME type, e.g. ``image/png`` / ``application/pdf`` / ``audio/wav``.
    """
    b64 = base64.b64encode(data).decode("ascii")
    return ImageUrlPart(image_url=ImageUrlInner(url=f"data:{mime};base64,{b64}"))


class ChatMessage(BaseModel):
    """Single chat-style message turn.

    ``content`` accepts either a plain ``str`` (text-only turn, the common case) or a ``list[ContentPart]``
    for multimodal turns (text + image / pdf / audio data URIs). When serialised the OpenAI / OpenRouter wire
    form falls out automatically:

    - ``{"role": "user", "content": "hi"}`` — text
    - ``{"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:..."}}]}`` — multimodal

    The ``role`` set is intentionally narrow (3 values). ``tool`` is out of EPISODE scope; adding it later is
    a SemVer minor bump (extending a Literal is a backward-compatible structural widening).
    """

    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]

    model_config = ConfigDict(extra="ignore")


class Usage(BaseModel):
    """Token usage from a single LLM call.

    Both fields are ``int | None`` because some self-hosted / OpenAI-compatible backends do not return
    ``usage`` in the response. ``None`` semantically distinguishes "missing data" from "zero tokens used".
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ChatResponse(BaseModel):
    """Structured response from a single LLM chat call."""

    content: str
    model: str
    usage: Usage | None = None
    finish_reason: Literal["stop", "length", "content_filter"] | None = None
    parsed: BaseModel | None = Field(
        default=None,
        description=(
            "Structured output instance returned by providers that support the OpenAI "
            "Structured Outputs API (``client.beta.chat.completions.parse``).  "
            "Callers should cast to the concrete schema type they passed as "
            "``response_format``.  ``None`` when the provider used ``json_object`` "
            "mode or when no ``response_format`` was requested."
        ),
    )
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
