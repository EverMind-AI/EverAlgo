"""Tests for ``LLMClient``: its signature, what ``isinstance`` can tell, and static conformance."""

import inspect
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import BaseModel

from everalgo.llm.protocols import LLMClient
from everalgo.llm.types import LLMMessage

if TYPE_CHECKING:
    from everalgo.llm.openai_compat import OpenAICompatClient
    from everalgo.testing import FakeLLMClient


class _ConformingClient:
    """Has an ``achat`` matching the protocol and nothing else."""

    async def achat(
        self,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> str:
        return "{}"


class _NonConformingClient:
    """Has no ``achat`` method."""

    def something_else(self) -> None:
        return None


def test_isinstance_accepts_an_object_with_an_achat_attribute() -> None:
    """``isinstance`` sees only that ``achat`` exists, not its signature; static conformance is checked below."""
    assert isinstance(_ConformingClient(), LLMClient)


def test_isinstance_rejects_an_object_without_achat() -> None:
    assert not isinstance(_NonConformingClient(), LLMClient)


class _PrefixlessClient:
    """Implements the contract under ``chat``, the method's name before it took the ``a`` prefix."""

    async def chat(self, messages: list[LLMMessage], *, response_format: type[BaseModel]) -> str:
        return "{}"


def test_the_prefixless_name_neither_conforms_nor_survives_as_an_alias() -> None:
    """A client offering only the prefix-less ``chat`` does not conform, and the protocol keeps no alias."""
    assert not isinstance(_PrefixlessClient(), LLMClient)
    assert not hasattr(LLMClient, "chat")


if TYPE_CHECKING:

    def _implementations_conform(fake: FakeLLMClient, reference: OpenAICompatClient) -> list[LLMClient]:
        """Checked by mypy, never run: an ``achat`` that drifts from the protocol makes this return type fail."""
        return [fake, reference]


def test_achat_signature_is_the_contract() -> None:
    """Pin the four-item call surface: messages, a mandatory model class, two optional knobs, the escape hatch."""
    parameters = inspect.signature(LLMClient.achat).parameters

    assert list(parameters) == ["self", "messages", "response_format", "temperature", "max_tokens", "extra"]
    assert parameters["response_format"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["response_format"].default is inspect.Parameter.empty
    assert parameters["temperature"].default is None
    assert parameters["max_tokens"].default is None
    assert parameters["extra"].kind is inspect.Parameter.VAR_KEYWORD
    assert "model" not in parameters


def test_achat_returns_plain_text_and_takes_a_model_class() -> None:
    hints = get_type_hints(LLMClient.achat)

    assert hints["return"] is str
    assert hints["response_format"] == type[BaseModel]
    assert hints["messages"] == list[LLMMessage]
