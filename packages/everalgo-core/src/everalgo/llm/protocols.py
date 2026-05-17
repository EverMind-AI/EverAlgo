"""LLM client Protocol — the structural contract every provider satisfies."""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from everalgo.llm.types import ChatMessage, ChatResponse


@runtime_checkable
class LLMClient(Protocol):
    """Async LLM client structural contract (PEP 544). Structural conformance suffices — no subclassing needed."""

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        """Send a chat request and return the structured response.

        Raises:
            LLMError: On any provider-side failure; original SDK exception attached as ``__cause__``.
        """
        ...
