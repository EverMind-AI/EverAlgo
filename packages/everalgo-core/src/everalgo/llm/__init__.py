"""LLM facade — chat-style abstraction over OpenAI-compatible providers.

Public surface, grouped by role:

- protocol:    LLMClient
- data:        ChatMessage, ChatResponse, ContentPart, ImageUrlInner, ImageUrlPart, TextPart, Usage, LLMConfig
- helpers:     image_url_part_from_bytes
- errors:      LLMError
- factory:     build_client
- format/parse: Lang, format_atomic_fact_time, format_iso_timestamp, format_message_timestamp, format_natural_language_time, parse_llm_json_object

LLM binding is instance-only: every operator accepts ``llm: LLMClient`` at construction time.
There is no global default, no scoped context-var override, and no per-call resolution layer.
This matches the industry convention used by openai-python, anthropic-sdk-python, LangChain,
Instructor, and scikit-learn (5/7 dominant SDKs bind the client at construction time only).
"""

from __future__ import annotations

import logging

from everalgo.llm._filters import SensitiveHeadersFilter
from everalgo.llm.config import LLMConfig
from everalgo.llm.errors import LLMError
from everalgo.llm.factory import build_client
from everalgo.llm.format import (
    Lang,
    format_atomic_fact_time,
    format_iso_timestamp,
    format_message_timestamp,
    format_natural_language_time,
)
from everalgo.llm.parse import parse_llm_json_object
from everalgo.llm.protocols import LLMClient
from everalgo.llm.types import (
    ChatMessage,
    ChatResponse,
    ContentPart,
    ImageUrlInner,
    ImageUrlPart,
    TextPart,
    Usage,
    image_url_part_from_bytes,
)

# Library logging setup (ADR-013). NullHandler suppresses "No handlers" noise
# under Python's library-logging contract; SensitiveHeadersFilter redacts
# authorization-style values from `record.args` mappings, default-on.
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())
_logger.addFilter(SensitiveHeadersFilter())

__all__ = [
    "ChatMessage",
    "ChatResponse",
    "ContentPart",
    "ImageUrlInner",
    "ImageUrlPart",
    "LLMClient",
    "LLMConfig",
    "LLMError",
    "Lang",
    "TextPart",
    "Usage",
    "build_client",
    "format_atomic_fact_time",
    "format_iso_timestamp",
    "format_message_timestamp",
    "format_natural_language_time",
    "image_url_part_from_bytes",
    "parse_llm_json_object",
]  # fmt: skip
