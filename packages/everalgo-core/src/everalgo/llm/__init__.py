"""LLM facade: the ``LLMClient`` contract, its request types, request builders for implementations, reply parsing."""

from __future__ import annotations

__all__ = [
    "ContentPart",
    "ImageUrlInner",
    "ImageUrlPart",
    "LLMClient",
    "LLMMessage",
    "LLMOutputError",
    "StrictOutputModel",
    "TextPart",
    "describe_errors",
    "image_url_part_from_bytes",
    "parse_reply",
    "to_anthropic_response_format",
    "to_gemini_response_format",
    "to_ollama_response_format",
    "to_openai_response_format",
]  # fmt: skip

import logging

from everalgo.llm.errors import LLMOutputError
from everalgo.llm.output import StrictOutputModel, describe_errors, parse_reply
from everalgo.llm.protocols import LLMClient
from everalgo.llm.schema import (
    to_anthropic_response_format,
    to_gemini_response_format,
    to_ollama_response_format,
    to_openai_response_format,
)
from everalgo.llm.types import (
    ContentPart,
    ImageUrlInner,
    ImageUrlPart,
    LLMMessage,
    TextPart,
    image_url_part_from_bytes,
)

# Library logging (ADR-013): silent until the application configures it.
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())
