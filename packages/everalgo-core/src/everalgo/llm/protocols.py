"""LLM client Protocol — the structural contract every provider satisfies."""

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

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
        response_format: type[BaseModel] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        """Send a chat request and return the structured response.

        When ``response_format`` is a ``BaseModel`` subclass, providers that support
        the OpenAI Structured Outputs API (``client.beta.chat.completions.parse``) will
        return a ``ChatResponse`` with the ``parsed`` field populated.

        Raises:
            LLMError: On any provider-side failure; original SDK exception attached as ``__cause__``.
        """
        ...
