"""LLM client Protocol — the structural contract every provider satisfies."""

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from everalgo.llm.types import ChatMessage, ChatResponse


@runtime_checkable
class LLMClient(Protocol):
    """Async LLM client structural contract.

    Implementations need not subclass this Protocol; structural conformance
    suffices (PEP 544). The ``@runtime_checkable`` decorator is for sanity
    checks (e.g. inside ``build_client``); production callers rely on static
    typing rather than ``isinstance``.
    """

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
        """Send a chat-style request and await the assistant reply.

        Args:
            messages: Ordered conversation, ending with the latest user turn.
            model: Override the per-config default model for this call.
            temperature: Override per-config default; ``None`` falls back to
                the value baked into the config.
            max_tokens: Override per-config default; ``None`` falls back to
                the value baked into the config.
            response_format: OpenAI-compatible ``response_format`` field
                (e.g. ``{"type": "json_object"}`` for JSON mode).
            **extra: Provider-specific knobs forwarded as kwargs.

        Returns:
            ``ChatResponse`` with structured ``content`` / ``usage`` /
            ``finish_reason`` plus optional ``raw`` for debug.

        Raises:
            LLMError: Any provider-side failure, with the original SDK
                exception attached as ``__cause__`` (PEP 3134).
        """
        ...
