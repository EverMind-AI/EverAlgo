"""Tenacity retry factories for benchmark stages."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, TypeVar

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)

from everalgo.llm.errors import LLMError

if TYPE_CHECKING:
    from collections.abc import Callable

    from tenacity.wait import WaitBaseT

__all__ = ["answer_retry", "http_retry", "llm_retry"]

logger = logging.getLogger(__name__)

F = TypeVar("F", bound="Callable[..., object]")


def _log_before_sleep(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retry attempt %d/%d for %s (%s); sleeping %.1f s",
        retry_state.attempt_number,
        retry_state.retry_object.stop.max_attempt_number,  # type: ignore[union-attr]
        retry_state.fn.__qualname__ if retry_state.fn else "unknown",
        f"{type(exc).__name__}: {exc}" if exc else "bad result",
        retry_state.next_action.sleep if retry_state.next_action else 0,  # type: ignore[union-attr]
    )


def llm_retry(*, max_attempts: int = 5, wait: WaitBaseT | None = None) -> Callable[[F], F]:
    """LLM call retry: JSONDecodeError / ValueError / LLMError."""
    return retry(  # type: ignore[return-value]
        retry=retry_if_exception_type((json.JSONDecodeError, ValueError, LLMError)),
        stop=stop_after_attempt(max_attempts),
        wait=wait if wait is not None else wait_exponential(multiplier=0.5, min=0.5, max=16),
        before_sleep=_log_before_sleep,
        reraise=True,
    )


def _is_retryable_http(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
        return False
    return isinstance(exc, (httpx.HTTPStatusError, httpx.TransportError))


def http_retry(*, max_attempts: int = 3, wait: WaitBaseT | None = None) -> Callable[[F], F]:
    """HTTP service retry: 5xx / TransportError (4xx errors propagate immediately)."""
    return retry(  # type: ignore[return-value]
        retry=retry_if_exception(_is_retryable_http),
        stop=stop_after_attempt(max_attempts),
        wait=wait if wait is not None else wait_exponential(multiplier=1.0, min=1, max=16),
        before_sleep=_log_before_sleep,
        reraise=True,
    )


def _is_empty_answer(result: object) -> bool:
    return not result


def answer_retry(*, max_attempts: int = 5, wait: WaitBaseT | None = None) -> Callable[[F], F]:
    """Answer generation retry: any exception + empty answer."""
    return retry(  # type: ignore[return-value]
        retry=retry_if_exception_type(Exception) | retry_if_result(_is_empty_answer),
        stop=stop_after_attempt(max_attempts),
        wait=wait if wait is not None else wait_exponential(multiplier=1.0, min=1, max=16),
        before_sleep=_log_before_sleep,
        reraise=True,
    )
