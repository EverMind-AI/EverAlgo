"""Tests for everalgo.llm.protocols.LLMClient — structural conformance."""

from typing import Any

from everalgo.llm.protocols import LLMClient
from everalgo.llm.types import ChatMessage, ChatResponse


class _ConformingClient:
    """Minimal class that structurally satisfies LLMClient."""

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Any = None,
        **extra: Any,
    ) -> ChatResponse:
        return ChatResponse(content="", model=model or "fake")


class _NonConformingClient:
    """Missing the chat method — should not pass isinstance(... , LLMClient)."""

    def something_else(self) -> None:
        return None


def test_conforming_client_is_instance_of_protocol() -> None:
    """@runtime_checkable Protocol uses structural subtyping at runtime."""
    instance = _ConformingClient()
    assert isinstance(instance, LLMClient)


def test_non_conforming_client_is_not_instance_of_protocol() -> None:
    instance = _NonConformingClient()
    assert not isinstance(instance, LLMClient)


def test_protocol_is_runtime_checkable() -> None:
    """isinstance(_, LLMClient) must be supported via @runtime_checkable."""
    # If the decorator is missing, isinstance raises TypeError.
    isinstance(object(), LLMClient)
