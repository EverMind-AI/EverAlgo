"""Reference ``LLMClient`` implementation over the OpenAI Chat Completions HTTP API."""

from __future__ import annotations

__all__ = ["LLMError", "OpenAICompatClient"]

import asyncio
import logging
import random
from typing import TYPE_CHECKING, Any, Final, Literal

import httpx
from pydantic import BaseModel, ValidationError

from everalgo.llm.schema import to_openai_response_format

if TYPE_CHECKING:
    from everalgo.llm.types import LLMMessage

_logger = logging.getLogger(__name__)

_CHAT_COMPLETIONS_PATH: Final = "chat/completions"

# The identical request hits the same output cap or the same filter again, so these raise without a resend.
_CUT_SHORT_REASONS: Final = frozenset({"length", "content_filter"})

# Not OpenAI values: vLLM-style gateways report them when the server gave up on a generation it could have
# finished (preemption, an upstream blip), so they are resent like a 5xx.
_ABORTED_REASONS: Final = frozenset({"abort", "error"})

# Retry statuses and backoff constants as in openai SDK 2.35.1, ``_base_client.py``.
_TRANSIENT_STATUSES: Final = frozenset({408, 409, 429})
_BACKOFF_INITIAL_SECONDS: Final = 0.5
_BACKOFF_MAX_SECONDS: Final = 8.0
_RETRY_AFTER_MAX_SECONDS: Final = 60.0


class LLMError(Exception):
    """A call failed, or the upstream refused or cut it short; not part of the ``LLMClient`` contract."""


class _TransientError(LLMError):
    """A failure the identical request may not hit again, so it is resent while transport retries remain.

    ``reason`` is what gets logged; ``detail``, such as an upstream error body, goes into the message only, since a
    body may carry request content that must stay out of logs.
    """

    def __init__(self, reason: str, *, detail: str = "", retry_after: float | None = None) -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.retry_after = retry_after


class _Message(BaseModel):
    content: str | None = None
    refusal: str | None = None


class _Choice(BaseModel):
    message: _Message
    finish_reason: str | None = None


class _Completion(BaseModel):
    """The part of a Chat Completions response this client reads; every other field is ignored."""

    model: str | None = None
    choices: list[_Choice] = []


class OpenAICompatClient:
    """Reference ``LLMClient`` over OpenAI Chat Completions, following every clause and recommendation of the contract.

    It requests the reply format as ``response_format_mode`` says and raises ``LLMError`` on every upstream outcome
    the contract forbids returning. Each reply is validated against ``response_format`` before it is handed over, and
    one that fails validation is retried with the identical request (``docs/concepts/llm-contract.md`` §3, §7).

    The client owns a connection pool; release it with ``aclose()``, or scope it with ``contextlib.aclosing``.

    Args:
        model: Model every request names, unless a call passes ``model`` through ``**extra``.
        api_key: Sent as a bearer token.
        base_url: Base URL of the OpenAI-compatible endpoint.
        temperature: Used when a call passes ``temperature=None``; ``None`` here leaves it out of the request.
        max_tokens: Used when a call passes ``max_tokens=None``; ``None`` here leaves it out of the request.
        timeout: Timeout of each HTTP request, in seconds.
        response_format_mode: How the reply format is requested: ``json_schema`` sends the strict schema of the
            call's ``response_format``, ``json_object`` sends JSON mode, and ``prompt_only`` sends nothing and relies
            on the prompt. The reply is JSON in every case.
        invalid_reply_retries: Extra attempts after a reply that fails validation; a reported refusal or a
            generation cut short (``length``, ``content_filter``) is never retried.
        transport_retries: Extra attempts after a connection error, a timeout, HTTP 408, 409, 429 or 5xx, a reply
            with no choice, or a generation the server aborted (``abort``, ``error``), each after an exponential
            backoff or the upstream's ``Retry-After``.

    Raises:
        ValueError: If a retry count is negative.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        response_format_mode: Literal["json_schema", "json_object", "prompt_only"] = "json_schema",
        invalid_reply_retries: int = 1,
        transport_retries: int = 2,
    ) -> None:
        if invalid_reply_retries < 0 or transport_retries < 0:
            raise ValueError(
                f"retry counts must be non-negative, got invalid_reply_retries={invalid_reply_retries} "
                f"and transport_retries={transport_retries}"
            )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._response_format_mode: Literal["json_schema", "json_object", "prompt_only"] = response_format_mode
        self._invalid_reply_retries = invalid_reply_retries
        self._transport_retries = transport_retries
        self._http = httpx.AsyncClient(
            base_url=base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout
        )

    async def aclose(self) -> None:
        """Close the connection pool; the client cannot send requests afterwards."""
        await self._http.aclose()

    async def achat(
        self,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> str:
        """Send one chat request to the configured model; see ``LLMClient.achat`` for the contract.

        ``**extra`` is sent last and may override any other request parameter, ``model`` included. Every attempt
        sends the same request.

        Returns:
            The reply text, validated against ``response_format``.

        Raises:
            LLMError: If every transport attempt fails transiently, a reply with no choice or an aborted generation
                included; the upstream answers with an error status, a malformed body, a refusal, no content, or a
                generation cut short; or no attempt yields a reply that validates.
            TypeError: If ``response_format_mode`` is ``json_schema`` and ``response_format`` is a ``RootModel``.
            ValueError: If ``response_format_mode`` is ``json_schema`` and ``response_format`` lies outside the
                strict subset in any other way (see ``docs/concepts/llm-contract.md`` §8.3).
        """
        # 1. Build the request once, so every attempt sends it unchanged.
        body = self._build_request_body(
            messages, response_format, temperature=temperature, max_tokens=max_tokens, extra=extra
        )
        # 2. Send it until a reply passes every check.
        return await self._request_valid_reply(body, response_format)

    def _build_request_body(
        self,
        messages: list[LLMMessage],
        response_format: type[BaseModel],
        *,
        temperature: float | None,
        max_tokens: int | None,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the request body: call arguments before the client's defaults, ``None`` left out, ``extra`` last."""
        optional: dict[str, Any] = {
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            "response_format": to_openai_response_format(response_format, self._response_format_mode),
        }
        return {
            "model": self._model,
            "messages": [message.model_dump() for message in messages],
            **{key: value for key, value in optional.items() if value is not None},
            **extra,
        }

    async def _request_valid_reply(self, body: dict[str, Any], response_format: type[BaseModel]) -> str:
        """Send the request until a reply validates, at most ``1 + invalid_reply_retries`` times."""
        attempts = 1 + self._invalid_reply_retries
        failure: ValidationError | None = None
        for attempt in range(1, attempts + 1):
            # 1. Send the request; a transient failure is resent within transport_retries.
            completion = await self._send(body)
            # 2. Check what the upstream reported; a refusal or a cut-short generation raises and is never retried.
            text = _check_upstream_report(completion)
            # 3. Validate the reply against response_format; an invalid one is sent again.
            failure = _validation_error(text, response_format)
            if failure is None:
                return text
            self._log_invalid_reply(response_format, attempt, attempts, failure)
        raise self._no_valid_reply_error(response_format, attempts) from failure

    async def _send(self, body: dict[str, Any]) -> _Completion:
        """Send the request, resending it after a transient failure at most ``transport_retries`` times.

        A transient failure is a connection error, a timeout, HTTP 408, 409, 429 or 5xx, a reply with no choice, or
        a generation the server aborted (finish reason ``abort`` or ``error``).

        Raises:
            LLMError: If the upstream answers with any other error status or a malformed body, or the last attempt
                fails transiently.
        """
        attempts = 1 + self._transport_retries
        attempt = 1
        while True:
            # 1. Send once; anything but a transient failure is final.
            try:
                return await self._send_once(body)
            except _TransientError as failure:
                if attempt == attempts:
                    raise
                delay = failure.retry_after or _backoff_seconds(attempt)
                self._log_resend(failure.reason, delay, attempt, attempts)
            # 2. Wait out the backoff, then send again.
            await asyncio.sleep(delay)
            attempt += 1

    async def _send_once(self, body: dict[str, Any]) -> _Completion:
        """POST the request once and parse the completion.

        Raises:
            _TransientError: On a transport error, a transient status, no choice or an aborted generation.
            LLMError: On any other error status or a malformed body.
        """
        try:
            response = await self._http.post(_CHAT_COMPLETIONS_PATH, json=body)
        except httpx.TransportError as exc:
            raise _TransientError(repr(exc)) from exc
        if _is_transient(response.status_code):
            raise _TransientError(
                f"upstream answered HTTP {response.status_code}",
                detail=response.text,
                retry_after=_retry_after_seconds(response),
            )
        if not response.is_success:
            raise LLMError(f"upstream answered HTTP {response.status_code}: {response.text}")
        completion = _parse_completion(response)
        # OpenRouter answers some upstream rejections with a 200 and no choices; another attempt may route elsewhere.
        if not completion.choices:
            raise _TransientError(f"upstream returned no choices (model={completion.model!r})")
        if (reason := _aborted_reason(completion.choices[0])) is not None:
            raise _TransientError(f"generation aborted (finish_reason={reason!r})")
        return completion

    def _log_invalid_reply(
        self, response_format: type[BaseModel], attempt: int, attempts: int, error: ValidationError
    ) -> None:
        _logger.warning(
            "Reply failed validation against %s (attempt %d of %d, model=%s, response_format_mode=%s, %d errors)",
            response_format.__name__,
            attempt,
            attempts,
            self._model,
            self._response_format_mode,
            error.error_count(),
        )

    def _no_valid_reply_error(self, response_format: type[BaseModel], attempts: int) -> LLMError:
        return LLMError(
            f"no reply validated against {response_format.__name__} in {attempts} attempts "
            f"(model={self._model!r}, response_format_mode={self._response_format_mode!r})"
        )

    def _log_resend(self, reason: str, delay: float, attempt: int, attempts: int) -> None:
        _logger.warning(
            "Resending the request in %.2fs after %s (attempt %d of %d, model=%s)",
            delay,
            reason,
            attempt,
            attempts,
            self._model,
        )


def _check_upstream_report(completion: _Completion) -> str:
    """Return the reply text, raising ``LLMError`` on a refusal, a cut-short generation or absent content.

    ``completion`` comes from ``_send``, which never returns one without a choice.
    """
    choice = completion.choices[0]
    # Checked before the content: a refusal may be phrased as valid JSON that the operator would accept.
    if choice.message.refusal:
        raise LLMError(f"LLM refused (model={completion.model!r}): {choice.message.refusal}")
    # Checked before the content too: a reply cut off exactly after its closing brace still parses.
    if choice.finish_reason in _CUT_SHORT_REASONS:
        raise LLMError(
            f"LLM generation did not finish (model={completion.model!r}, finish_reason={choice.finish_reason!r})"
        )
    if choice.message.content is None:
        raise LLMError(f"LLM returned no content (model={completion.model!r}, finish_reason={choice.finish_reason!r})")
    return choice.message.content


def _validation_error(text: str, response_format: type[BaseModel]) -> ValidationError | None:
    """Return why ``text`` fails validation against ``response_format``, or ``None`` if it passes."""
    try:
        response_format.model_validate_json(text)
    except ValidationError as exc:
        return exc
    return None


def _parse_completion(response: httpx.Response) -> _Completion:
    try:
        return _Completion.model_validate_json(response.content)
    except ValidationError as exc:
        raise LLMError(f"upstream answered HTTP {response.status_code} with a malformed chat completion") from exc


def _aborted_reason(choice: _Choice) -> str | None:
    """Return the finish reason of a generation the server aborted, or ``None``; a refusal is never an abort."""
    if choice.message.refusal or choice.finish_reason not in _ABORTED_REASONS:
        return None
    return choice.finish_reason


def _is_transient(status_code: int) -> bool:
    return status_code in _TRANSIENT_STATUSES or status_code >= 500


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with up to 25% jitter, so concurrent callers do not resend in lockstep."""
    ceiling = min(_BACKOFF_INITIAL_SECONDS * 2.0 ** (attempt - 1), _BACKOFF_MAX_SECONDS)
    return ceiling * random.uniform(0.75, 1.0)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Return the upstream's ``Retry-After`` in seconds, or ``None`` if it is absent, not a number or out of range."""
    try:
        seconds = float(response.headers.get("retry-after", ""))
    except ValueError:
        return None
    return seconds if 0 < seconds <= _RETRY_AFTER_MAX_SECONDS else None
