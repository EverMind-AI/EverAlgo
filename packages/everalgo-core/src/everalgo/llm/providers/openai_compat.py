"""OpenAI-compatible provider — wraps openai.AsyncOpenAI."""

from collections.abc import Mapping
from typing import Any, Literal

import openai

from everalgo.llm.config import LLMConfig
from everalgo.llm.errors import LLMError
from everalgo.llm.types import ChatMessage, ChatResponse, Usage


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
        response_format: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> ChatResponse:
        """Implement ``LLMClient.chat`` — see protocols.py for contract."""
        request_kwargs: dict[str, Any] = {
            "model": model or self._config.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": (temperature if temperature is not None else self._config.temperature),
        }
        max_tokens_val = max_tokens if max_tokens is not None else self._config.max_tokens
        if max_tokens_val is not None:
            request_kwargs["max_tokens"] = max_tokens_val
        if response_format is not None:
            request_kwargs["response_format"] = dict(response_format)
        request_kwargs.update(self._config.extra)
        request_kwargs.update(extra)

        try:
            completion = await self._client.chat.completions.create(**request_kwargs)
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
        usage: Usage | None = None
        if completion.usage is not None:
            usage = Usage(
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
            )

        finish_reason = _normalise_finish_reason(choice.finish_reason)

        return ChatResponse(
            content=choice.message.content or "",
            model=completion.model,
            usage=usage,
            finish_reason=finish_reason,
            raw=None,
        )


def _normalise_finish_reason(
    value: str | None,
) -> Literal["stop", "length", "content_filter"] | None:
    """Collapse provider finish reasons to EverAlgo's 3-value Literal; unknown values map to ``None``."""
    if value in ("stop", "length", "content_filter"):
        return value  # type: ignore[return-value]
    return None
