"""In-memory LLMClient double for unit tests."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from everalgo.llm.errors import LLMError
from everalgo.llm.types import ChatMessage, ChatResponse


class CallRecord(BaseModel):
    """Single recorded ``FakeLLMClient.chat`` invocation; public so tests can assert on captured arguments."""

    messages: list[ChatMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: type[BaseModel] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")


# ---- FakeLLMClient (Task 2: constructor only) ----------------------------

_HandlerReturn = ChatResponse | Awaitable[ChatResponse]
_Handler = Callable[..., _HandlerReturn]


class FakeLLMClient:
    """In-memory ``LLMClient`` for unit tests.

    Pass ``responses=[...]`` (popped in order; exhaustion raises ``RuntimeError``) or ``handler=callable``
    (sync or async). Exactly one must be supplied.
    """

    def __init__(
        self,
        responses: Sequence[str | ChatResponse] | None = None,
        *,
        handler: _Handler | None = None,
    ) -> None:
        if (responses is None) == (handler is None):
            raise ValueError("Provide exactly one of `responses` or `handler`")
        self._responses: list[ChatResponse] | None = (
            [_coerce_response(r) for r in responses] if responses is not None else None
        )
        self._initial_response_count: int = len(self._responses) if self._responses is not None else 0
        self._handler: _Handler | None = handler
        self._calls: list[CallRecord] = []

    @property
    def call_count(self) -> int:
        """Number of times ``chat`` has been invoked."""
        return len(self._calls)

    @property
    def calls(self) -> list[CallRecord]:
        """Return all recorded ``chat`` invocations in call order (defensive copy)."""
        return list(self._calls)

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
        """Match ``LLMClient.chat`` Protocol; record + dispatch by mode."""
        self._calls.append(
            CallRecord(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                extra=dict(extra),
            )
        )
        if self._handler is not None:
            base_response = await _invoke_handler(
                self._handler,
                messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **extra,
            )
            if inspect.isclass(response_format) and base_response.parsed is None:
                return _attach_parsed(base_response, response_format)
            return base_response
        # scripted-list mode
        assert self._responses is not None  # narrowed by __init__ invariant
        if not self._responses:
            raise RuntimeError(
                f"FakeLLMClient script exhausted "
                f"(used {self._initial_response_count} of "
                f"{self._initial_response_count} responses)"
            )
        base_response = self._responses.pop(0)
        if inspect.isclass(response_format):
            return _attach_parsed(base_response, response_format)
        return base_response


def _coerce_response(value: str | ChatResponse) -> ChatResponse:
    """Wrap raw str into a default ChatResponse; pass ChatResponse through."""
    if isinstance(value, ChatResponse):
        return value
    if isinstance(value, str):  # pyright: ignore[reportUnnecessaryIsInstance]
        return ChatResponse(
            content=value,
            model="fake",
            usage=None,
            finish_reason="stop",
            raw=None,
        )
    raise TypeError(f"FakeLLMClient `responses` must contain str or ChatResponse, got {type(value).__name__}")


async def _invoke_handler(
    handler: _Handler,
    messages: list[ChatMessage],
    **kwargs: Any,
) -> ChatResponse:
    """Call handler; await if it returned a coroutine; type-check result."""
    result = handler(messages, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, ChatResponse):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"FakeLLMClient handler must return ChatResponse or Awaitable[ChatResponse], got {type(result).__name__}"
        )
    return result


def _attach_parsed(response: ChatResponse, schema: type[BaseModel]) -> ChatResponse:
    """Construct a ``parsed`` instance from ``response.content`` and attach it.

    Mirrors ``OpenAICompatClient``'s parse-branch behaviour so tests written against
    ``FakeLLMClient`` see the same ``ChatResponse.parsed`` field as production code.

    Raises:
        LLMError: If ``response.content`` is not valid JSON or does not conform to ``schema``.
    """
    try:
        parsed = schema.model_validate_json(response.content)
    except Exception as exc:
        raise LLMError(
            f"FakeLLMClient: content {response.content!r} is not valid for schema {schema.__name__}: {exc}"
        ) from exc
    return response.model_copy(update={"parsed": parsed})
