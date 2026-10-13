"""Tests for everalgo.testing.conformance — every check passes a compliant client and catches its breach.

A check that passes everything proves nothing, so each one is run against a toy client with exactly one
breach switched on and must go red there, while the same toy without the breach stays green.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal

import pytest
from pydantic import BaseModel, ValidationError

from everalgo.llm.types import LLMMessage, TextPart
from everalgo.testing import conformance
from everalgo.testing.conformance import (
    NOT_SENT,
    OPTIONAL_CHECKS,
    REQUIRED_CHECKS,
    ConformanceCheck,
    EffectiveRequest,
    UpstreamStub,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from everalgo.llm.protocols import LLMClient

Violation = Literal[
    "default_wins",
    "falsy_explicit",
    "literal_null",
    "retry_raises_max_tokens",
    "drops_parts",
    "swallows_cancel",
    "returns_refusal",
    "returns_truncated",
    "unbounded_retries",
]

_UNFINISHED = {"length", "content_filter", "abort", "error"}


class _ToyError(Exception):
    """The toy implementation's own error type."""


class _ToyClient:
    """Minimal ``LLMClient`` over an in-memory upstream; ``violation`` switches on one breach of the contract.

    With ``validates`` it also follows the optional recommendations: it validates the reply against
    ``response_format`` and retries a bounded number of times.
    """

    def __init__(
        self,
        upstream: Callable[[dict[str, Any]], Awaitable[UpstreamStub]],
        defaults: dict[str, Any],
        *,
        violation: Violation | None,
        validates: bool,
    ) -> None:
        self._upstream = upstream
        self._defaults = defaults
        self._violation = violation
        self._validates = validates

    async def achat(
        self,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> str:
        request = self._request(messages, temperature=temperature, max_tokens=max_tokens)
        for _ in range(self._attempts()):
            stub = await self._send(request)
            if stub is None:
                continue
            if stub.finish_reason in _UNFINISHED and self._violation == "retry_raises_max_tokens":
                request = {**request, "max_tokens": request["max_tokens"] * 2}
                continue
            text = self._accept(stub)
            if not self._validates and self._violation != "unbounded_retries":
                return text
            try:
                response_format.model_validate_json(text)
            except ValidationError:
                continue
            return text
        raise _ToyError("no valid reply within the retry budget")

    def _attempts(self) -> int:
        if self._violation == "unbounded_retries":
            return 1_000_000
        return 3 if self._validates or self._violation == "retry_raises_max_tokens" else 1

    async def _send(self, request: dict[str, Any]) -> UpstreamStub | None:
        """Send one request; ``None`` means the upstream failed and this toy retries past it."""
        try:
            return await self._upstream(request)
        except asyncio.CancelledError:
            if self._violation == "swallows_cancel":
                return UpstreamStub(content="")
            raise
        except _ToyError:
            if self._violation == "unbounded_retries":
                await asyncio.sleep(0)  # yield, so the check's deadline can fire
                return None
            raise

    def _accept(self, stub: UpstreamStub) -> str:
        """Turn one upstream answer into reply text, raising where the contract says to."""
        if stub.refusal and self._violation != "returns_refusal":
            raise _ToyError("refused")
        if stub.finish_reason in _UNFINISHED and self._violation != "returns_truncated":
            raise _ToyError(f"unfinished: {stub.finish_reason}")
        return stub.content if stub.content is not None else (stub.refusal or "")

    def _request(
        self, messages: list[LLMMessage], *, temperature: float | None, max_tokens: int | None
    ) -> dict[str, Any]:
        request: dict[str, Any] = {"messages": [self._deliver(message) for message in messages]}
        for key, explicit in (("temperature", temperature), ("max_tokens", max_tokens)):
            default = self._defaults.get(key)
            match self._violation:
                case "default_wins":
                    value = default if default is not None else explicit
                case "falsy_explicit":
                    value = explicit or default
                case _:
                    value = explicit if explicit is not None else default
            if value is not None or self._violation == "literal_null":
                request[key] = value
        return request

    def _deliver(self, message: LLMMessage) -> LLMMessage:
        if self._violation != "drops_parts" or isinstance(message.content, str):
            return message
        return LLMMessage(role=message.role, content=[part for part in message.content if isinstance(part, TextPart)])


class _ToyHarness:
    def __init__(self, *, violation: Violation | None = None, validates: bool = False) -> None:
        self._violation: Violation | None = violation
        self._validates = validates
        self._queue: list[UpstreamStub] = []
        self._log: list[EffectiveRequest] = []

    def build(self, **defaults: Any) -> LLMClient:
        return _ToyClient(self._upstream, defaults, violation=self._violation, validates=self._validates)

    def queue_upstream(self, *responses: UpstreamStub) -> None:
        self._queue.extend(responses)

    def sent_requests(self) -> list[EffectiveRequest]:
        return list(self._log)

    async def _upstream(self, request: dict[str, Any]) -> UpstreamStub:
        self._log.append(
            EffectiveRequest(
                messages=request["messages"],
                temperature=request.get("temperature", NOT_SENT),
                max_tokens=request.get("max_tokens", NOT_SENT),
            )
        )
        if not self._queue:
            raise _ToyError("upstream script exhausted")
        stub = self._queue.pop(0)
        if stub.hang:
            await asyncio.Event().wait()
        return stub


def _check_id(check: ConformanceCheck) -> str:
    return getattr(check, "__name__", repr(check))


@pytest.fixture(autouse=True)
def _short_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the unbounded-retry breach from holding a test for the production deadline."""
    monkeypatch.setattr(conformance, "_CALL_TIMEOUT_S", 0.5)


@pytest.mark.parametrize("check", REQUIRED_CHECKS, ids=_check_id)
async def test_compliant_client_passes_every_required_check(check: ConformanceCheck) -> None:
    await check(_ToyHarness())


@pytest.mark.parametrize(
    ("violation", "check"),
    [
        ("default_wins", conformance.check_explicit_values_win_over_defaults),
        ("falsy_explicit", conformance.check_explicit_values_win_over_defaults),
        ("retry_raises_max_tokens", conformance.check_retries_never_raise_max_tokens),
        ("retry_raises_max_tokens", conformance.check_truncation_is_never_returned),
        ("literal_null", conformance.check_none_means_default_or_omitted),
        ("drops_parts", conformance.check_non_text_parts_reach_upstream),
        ("swallows_cancel", conformance.check_cancellation_propagates),
        ("returns_refusal", conformance.check_refusal_raises),
        ("returns_truncated", conformance.check_truncation_is_never_returned),
    ],
)
async def test_each_required_check_catches_its_breach(violation: Violation, check: ConformanceCheck) -> None:
    with pytest.raises(AssertionError):
        await check(_ToyHarness(violation=violation))


def test_every_required_check_has_a_breach_that_turns_it_red() -> None:
    covered = {
        conformance.check_explicit_values_win_over_defaults,
        conformance.check_retries_never_raise_max_tokens,
        conformance.check_none_means_default_or_omitted,
        conformance.check_non_text_parts_reach_upstream,
        conformance.check_cancellation_propagates,
        conformance.check_refusal_raises,
        conformance.check_truncation_is_never_returned,
    }
    assert covered == set(REQUIRED_CHECKS)


@pytest.mark.parametrize("check", OPTIONAL_CHECKS, ids=_check_id)
async def test_validating_client_passes_every_optional_check(check: ConformanceCheck) -> None:
    await check(_ToyHarness(validates=True))


async def test_invalid_reply_check_catches_a_client_that_does_not_validate() -> None:
    with pytest.raises(AssertionError):
        await conformance.check_invalid_replies_are_not_returned(_ToyHarness())


async def test_bounded_retry_check_catches_unbounded_retries() -> None:
    with pytest.raises(AssertionError, match="did not finish"):
        await conformance.check_retries_are_bounded(_ToyHarness(violation="unbounded_retries"))


class _TextOnlyClient:
    """Refuses any non-text content part before sending anything."""

    async def achat(
        self,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> str:
        raise _ToyError("this backend accepts text only")


class _TextOnlyHarness(_ToyHarness):
    def build(self, **defaults: Any) -> LLMClient:
        return _TextOnlyClient()


async def test_a_client_that_raises_on_non_text_parts_passes() -> None:
    """Refusing to deliver loudly is allowed; only dropping silently is a breach."""
    await conformance.check_non_text_parts_reach_upstream(_TextOnlyHarness())
