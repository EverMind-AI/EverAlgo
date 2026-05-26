"""OpenAI-compatible provider — wraps openai.AsyncOpenAI."""

import inspect
from typing import TYPE_CHECKING, Any, Literal, cast

import openai
from pydantic import BaseModel

from everalgo.llm.config import LLMConfig
from everalgo.llm.errors import LLMError
from everalgo.llm.types import ChatMessage, ChatResponse, Usage

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion


class OpenAICompatClient:
    """Thin async wrapper over ``openai.AsyncOpenAI``.

    Converts between EverAlgo types and the openai SDK shapes. No retry, rate-limit, or key-rotation logic —
    those are caller concerns.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value(),
            base_url=config.base_url,
            timeout=config.timeout,
        )

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
        """Implement ``LLMClient.chat`` — see protocols.py for contract.

        When ``response_format`` is a ``BaseModel`` subclass, routes to
        ``client.beta.chat.completions.parse`` (OpenAI Structured Outputs strict
        mode). ``None`` calls create without any format hint.
        """
        request_kwargs = _build_request_kwargs(
            messages=messages,
            model=model or self._config.model,
            temperature=temperature if temperature is not None else self._config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._config.max_tokens,
            config_extra=self._config.extra,
            extra=extra,
        )

        if inspect.isclass(response_format):
            return await self._chat_parse(request_kwargs, response_format)
        return await self._chat_create(request_kwargs)

    async def _chat_parse(
        self,
        request_kwargs: dict[str, Any],
        schema: type[BaseModel],
    ) -> ChatResponse:
        """Route to ``client.beta.chat.completions.parse`` for strict Structured Outputs."""
        try:
            completion = await self._client.beta.chat.completions.parse(
                response_format=schema,
                **request_kwargs,
            )
        except openai.OpenAIError as exc:
            raise LLMError(str(exc)) from exc

        if not completion.choices:
            raise LLMError(
                f"upstream returned no choices (model={completion.model!r}); "
                f"likely an upstream rejection or content filter"
            )
        choice = completion.choices[0]
        refusal = choice.message.refusal
        if refusal:
            raise LLMError(f"LLM refused: {refusal}")
        parsed_instance = choice.message.parsed
        if parsed_instance is None:
            raise LLMError("LLM returned no parsed structured output (parsed=None)")

        usage = _extract_usage(completion)
        finish_reason = _normalise_finish_reason(choice.finish_reason)
        return ChatResponse(
            content=choice.message.content or "",
            model=completion.model,
            usage=usage,
            finish_reason=finish_reason,
            parsed=parsed_instance,
            raw=None,
        )

    async def _chat_create(self, request_kwargs: dict[str, Any]) -> ChatResponse:
        """Route to ``client.chat.completions.create`` for standard / legacy calls."""
        try:
            completion = cast("ChatCompletion", await self._client.chat.completions.create(**request_kwargs))
        except openai.OpenAIError as exc:
            raise LLMError(str(exc)) from exc

        # Some OpenAI-compatible upstreams (notably OpenRouter when the upstream
        # model rejects the payload) return a 200 with ``choices=None`` or an
        # empty list rather than a structured HTTP error. Surface as LLMError
        # rather than crashing with ``NoneType`` subscript.
        if not completion.choices:
            raise LLMError(
                f"upstream returned no choices (model={completion.model!r}); "
                f"likely an upstream rejection or content filter"
            )
        choice = completion.choices[0]
        usage = _extract_usage(completion)
        finish_reason = _normalise_finish_reason(choice.finish_reason)
        return ChatResponse(
            content=choice.message.content or "",
            model=completion.model,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
        )


def _build_request_kwargs(
    *,
    messages: list[ChatMessage],
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    config_extra: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the base request kwargs shared by both create and parse paths."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [m.model_dump() for m in messages],
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    kwargs.update(config_extra)
    kwargs.update(extra)
    return kwargs


def _extract_usage(completion: Any) -> Usage | None:
    """Extract token usage from a completion object; returns ``None`` if absent."""
    if completion.usage is None:
        return None
    return Usage(
        prompt_tokens=completion.usage.prompt_tokens,
        completion_tokens=completion.usage.completion_tokens,
    )


def _normalise_finish_reason(
    value: str | None,
) -> Literal["stop", "length", "content_filter"] | None:
    """Collapse provider finish reasons to EverAlgo's 3-value Literal; unknown values map to ``None``."""
    if value in ("stop", "length", "content_filter"):
        return value  # type: ignore[return-value]
    return None
