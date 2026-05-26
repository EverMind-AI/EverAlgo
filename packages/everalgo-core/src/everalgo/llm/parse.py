"""Robust LLM JSON-object parsing and answer extraction helpers.

Provides three utilities used on the LLM output path:

- ``parse_llm_json_object`` — multi-strategy JSON object extraction (fence, direct, braces).
- ``retry_on_json_parse_failure`` — decorator factory that retries async callables on JSON/validation
  errors with exponential backoff.
- ``extract_final_answer`` — marker-based final-answer substring slicing.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

__all__ = ["extract_final_answer", "parse_llm_json_object", "retry_on_json_parse_failure"]

logger = logging.getLogger(__name__)


def parse_llm_json_object(raw: str) -> dict[str, Any]:
    """Parse ``raw`` as a JSON object via three strategies in order.

    Tries each strategy until one succeeds:

    1. `` ```json ... ``` `` fenced block
    2. Direct ``json.loads`` on the full string (the JSON-mode happy path)
    3. Outermost ``{ ... }`` substring (rescues prose-wrapped JSON)

    Args:
        raw: Raw string returned by the LLM.

    Returns:
        The parsed dict.

    Raises:
        ValueError: If all three strategies fail or the result is not a JSON object.
    """
    fence_match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, dict):
                return cast("dict[str, Any]", parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, dict):
            return cast("dict[str, Any]", parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return cast("dict[str, Any]", parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError("Failed to parse LLM response as a JSON object")


_BACKOFF_BASE_SECONDS = 0.5


def retry_on_json_parse_failure[T](
    *,
    max_retries: int = 3,
    retryable_exc: tuple[type[Exception], ...] = (json.JSONDecodeError, ValueError),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator that retries an async function on JSON parse or validation errors.

    Implements exponential backoff between retries (0.5 s * 2^attempt). Non-retryable
    exceptions propagate immediately without consuming retry budget.

    Args:
        max_retries: Total attempts including the first try. ``max_retries=3`` means the
            wrapped function may be called up to 3 times before the last exception is re-raised.
        retryable_exc: Tuple of exception types that trigger a retry. Defaults to
            ``(json.JSONDecodeError, ValueError)``, covering raw JSON decode failures and
            post-parse validation errors (e.g. missing required keys after
            ``parse_llm_json_object``).

    Returns:
        A decorator for async callables. The wrapped function re-runs on ``retryable_exc``
        up to ``max_retries`` total attempts, with exponential backoff between attempts.
        If all attempts fail, the last caught exception is re-raised.

    Raises:
        Exception: The last ``retryable_exc`` instance after all attempts are exhausted,
            or any non-retryable exception raised by the wrapped function.

    Example::

        @retry_on_json_parse_failure(max_retries=3)
        async def call_llm() -> dict[str, Any]:
            raw = await llm.achat(messages)
            return parse_llm_json_object(raw)
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: object, **kwargs: object) -> T:
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return await fn(*args, **kwargs)
                except retryable_exc as exc:
                    last_exc = exc
                    if attempt < max_retries - 1:
                        wait = _BACKOFF_BASE_SECONDS * (2**attempt)
                        raw_doc: str | None = getattr(exc, "doc", None)
                        logger.warning(
                            "LLM parse failure on attempt %d/%d (%s); retrying in %.1f s%s",
                            attempt + 1,
                            max_retries,
                            type(exc).__name__,
                            wait,
                            f"\nRaw LLM response: {raw_doc!r}" if raw_doc else "",
                            exc_info=True,
                        )
                        await asyncio.sleep(wait)
            assert last_exc is not None  # loop always assigns before exhausting attempts
            raise last_exc

        return wrapper

    return decorator


def extract_final_answer(raw: str, *, marker: str = "Final answer:") -> str:
    """Extract the substring after ``marker`` from LLM text output.

    Uses ``rsplit`` to take the portion after the **last** occurrence of ``marker``,
    which handles cases where the marker text appears in reasoning prose before the
    actual answer section.

    Args:
        raw: Raw LLM text output (may contain prose / chain-of-thought before the marker).
        marker: Sentinel string preceding the final answer. Default ``"Final answer:"``
            follows the answer-prompt convention with a leading ``Final answer:`` marker.

    Returns:
        The substring after the last occurrence of ``marker``, stripped of surrounding
        whitespace. If ``marker`` is not found, the entire ``raw`` text is returned stripped.
    """
    if marker not in raw:
        return raw.strip()
    return raw.rsplit(marker, 1)[1].strip()
