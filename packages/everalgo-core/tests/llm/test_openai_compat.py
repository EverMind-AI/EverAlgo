"""Tests for ``OpenAICompatClient``: transport, reply handling, reply validation and message delivery.

The transport is mocked with respx, so assertions read the request the client really sent.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, cast

import httpx
import pytest
import respx
from pydantic import ValidationError

from everalgo.llm import StrictOutputModel
from everalgo.llm.openai_compat import LLMError, OpenAICompatClient
from everalgo.llm.types import ImageUrlInner, ImageUrlPart, LLMMessage, TextPart

_BASE_URL = "https://api.openai.com/v1"


class _Answer(StrictOutputModel):
    text: str


def _client(**overrides: Any) -> OpenAICompatClient:
    return OpenAICompatClient(**{"model": "gpt-4o-mini", "api_key": "sk-test", "base_url": _BASE_URL, **overrides})


def _completion(
    *,
    content: str | None = '{"text": "hi there"}',
    refusal: str | None = None,
    finish_reason: str | None = "stop",
) -> dict[str, object]:
    return {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


async def _chat(
    payload: dict[str, object],
    *,
    messages: list[LLMMessage] | None = None,
    client: OpenAICompatClient | None = None,
    **chat_kwargs: Any,
) -> tuple[str, dict[str, Any]]:
    """Run one ``achat`` against a mocked upstream; return the reply and the JSON body that was sent."""
    client = client or _client()
    with respx.mock(base_url=_BASE_URL, assert_all_called=True) as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=payload))
        reply = await client.achat(
            messages or [LLMMessage(role="user", content="hello")], response_format=_Answer, **chat_kwargs
        )
    request = cast("httpx.Request", route.calls[0].request)  # pyright: ignore[reportUnknownMemberType]
    return reply, cast("dict[str, Any]", json.loads(request.content))


async def _chat_against(
    *payloads: dict[str, object], client: OpenAICompatClient | None = None
) -> tuple[str | LLMError, list[dict[str, Any]]]:
    """Run one ``achat`` against an upstream answering with ``payloads`` in turn; return the outcome and bodies sent."""
    client = client or _client()
    with respx.mock(base_url=_BASE_URL, assert_all_called=False) as router:
        responses = [httpx.Response(200, json=payload) for payload in payloads]
        route = router.post("/chat/completions").mock(side_effect=responses)
        try:
            outcome: str | LLMError = await client.achat(
                [LLMMessage(role="user", content="hello")], response_format=_Answer
            )
        except LLMError as exc:
            outcome = exc
    requests = [cast("httpx.Request", call.request) for call in route.calls]  # pyright: ignore[reportUnknownMemberType]
    return outcome, [cast("dict[str, Any]", json.loads(request.content)) for request in requests]


async def test_achat_returns_the_reply_text() -> None:
    reply, _ = await _chat(_completion())

    assert reply == '{"text": "hi there"}'


async def test_the_model_comes_from_the_constructor() -> None:
    _, sent = await _chat(_completion(), client=_client(model="cfg-model"))

    assert sent["model"] == "cfg-model"


# ---------------------------------------------------------------------------
# Transport: authentication, error statuses, bounded resends of transient failures
# ---------------------------------------------------------------------------


@pytest.fixture
def backoff_delays(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record every backoff sleep instead of waiting it out."""
    delays: list[float] = []

    async def _record(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _record)
    return delays


async def _chat_through(
    *responses: httpx.Response | Exception, client: OpenAICompatClient | None = None
) -> list[httpx.Request]:
    """Run one ``achat`` against an upstream answering with ``responses`` in turn; return the requests it received."""
    client = client or _client()
    with respx.mock(base_url=_BASE_URL, assert_all_called=False) as router:
        route = router.post("/chat/completions").mock(side_effect=list(responses))
        try:
            await client.achat([LLMMessage(role="user", content="hello")], response_format=_Answer)
        finally:
            requests = [cast("httpx.Request", call.request) for call in route.calls]  # pyright: ignore[reportUnknownMemberType]
    return requests


def _ok() -> httpx.Response:
    return httpx.Response(200, json=_completion())


async def test_the_api_key_is_sent_as_a_bearer_token() -> None:
    (request,) = await _chat_through(_ok(), client=_client(api_key="sk-secret"))

    assert request.headers["authorization"] == "Bearer sk-secret"


async def test_an_error_status_raises_llm_error_with_status_and_body_and_is_not_resent() -> None:
    error = httpx.Response(400, json={"error": {"message": "bad request"}})
    client = _client()

    with respx.mock(base_url=_BASE_URL) as router:
        route = router.post("/chat/completions").mock(return_value=error)
        with pytest.raises(LLMError, match=r"HTTP 400: .*bad request"):
            await client.achat([LLMMessage(role="user", content="hi")], response_format=_Answer)

    assert route.call_count == 1


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
async def test_a_transient_status_is_resent_with_the_identical_body(status: int, backoff_delays: list[float]) -> None:
    first, second = await _chat_through(httpx.Response(status), _ok())

    assert first.content == second.content
    assert len(backoff_delays) == 1


async def test_a_transport_error_is_resent(backoff_delays: list[float]) -> None:
    requests = await _chat_through(httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), _ok())

    assert len(requests) == 3
    assert len(backoff_delays) == 2


async def test_a_reply_with_no_choice_is_resent(backoff_delays: list[float]) -> None:
    """OpenRouter answers some upstream rejections with a 200 and ``choices=[]``; another attempt may route elsewhere."""
    no_choice = httpx.Response(200, json={**_completion(), "choices": []})

    requests = await _chat_through(no_choice, _ok())

    assert len(requests) == 2
    assert len(backoff_delays) == 1


async def test_replies_with_no_choice_raise_once_the_transport_retries_run_out(backoff_delays: list[float]) -> None:
    no_choice = httpx.Response(200, json={**_completion(), "choices": []})

    with pytest.raises(LLMError, match="no choices"):
        await _chat_through(no_choice, no_choice, _ok(), client=_client(transport_retries=1))

    assert len(backoff_delays) == 1


async def test_transient_failures_raise_once_the_transport_retries_run_out(backoff_delays: list[float]) -> None:
    overloaded = httpx.Response(503, text="upstream overloaded")

    with pytest.raises(LLMError, match="HTTP 503: upstream overloaded"):
        await _chat_through(overloaded, overloaded, _ok(), client=_client(transport_retries=1))

    assert len(backoff_delays) == 1


async def test_a_resend_is_logged_without_the_upstream_body(
    caplog: pytest.LogCaptureFixture, backoff_delays: list[float]
) -> None:
    """An error body may echo request content, so it reaches the raised message but never a log record."""
    caplog.set_level(logging.WARNING, logger="everalgo.llm.openai_compat")

    await _chat_through(httpx.Response(503, text="echo of the user's prompt"), _ok())

    (record,) = caplog.records
    assert "HTTP 503" in record.getMessage()
    assert "echo of the user's prompt" not in record.getMessage()


async def test_a_last_transport_error_raises_llm_error_with_it_as_cause(backoff_delays: list[float]) -> None:
    client = _client(transport_retries=0)

    with pytest.raises(LLMError) as caught:
        await _chat_through(httpx.ConnectError("refused"), client=client)

    assert isinstance(caught.value.__cause__, httpx.ConnectError)
    assert backoff_delays == []


async def test_backoff_doubles_from_half_a_second_with_jitter(backoff_delays: list[float]) -> None:
    await _chat_through(httpx.Response(500), httpx.Response(500), _ok())

    first, second = backoff_delays
    assert 0.375 <= first <= 0.5
    assert 0.75 <= second <= 1.0


async def test_retry_after_within_a_minute_replaces_the_backoff(backoff_delays: list[float]) -> None:
    await _chat_through(httpx.Response(429, headers={"retry-after": "3"}), _ok())

    assert backoff_delays == [3.0]


@pytest.mark.parametrize("retry_after", ["0", "61", "soon"])
async def test_retry_after_out_of_range_or_not_a_number_falls_back_to_the_backoff(
    retry_after: str, backoff_delays: list[float]
) -> None:
    await _chat_through(httpx.Response(429, headers={"retry-after": retry_after}), _ok())

    (delay,) = backoff_delays
    assert 0.375 <= delay <= 0.5


async def test_a_body_that_is_not_a_chat_completion_raises_llm_error() -> None:
    with pytest.raises(LLMError, match="malformed chat completion") as caught:
        await _chat_through(httpx.Response(200, text="<html>gateway</html>"))

    assert isinstance(caught.value.__cause__, ValidationError)


@pytest.mark.parametrize("retries", [{"invalid_reply_retries": -1}, {"transport_retries": -1}])
def test_a_negative_retry_count_is_rejected(retries: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _client(**retries)


async def test_aclose_releases_the_connection_pool() -> None:
    async with contextlib.aclosing(_client()) as client:
        pass

    with pytest.raises(RuntimeError, match="closed"):
        await client.achat([LLMMessage(role="user", content="hi")], response_format=_Answer)


# ---------------------------------------------------------------------------
# Refusal and absent content — never returned as a reply
# ---------------------------------------------------------------------------


async def test_refusal_without_content_raises_with_the_reason() -> None:
    with pytest.raises(LLMError, match="I cannot help with that request"):
        await _chat(_completion(content=None, refusal="I cannot help with that request."))


async def test_refusal_raises_even_when_the_content_is_valid_json() -> None:
    """A refusal phrased as a schema-valid object would pass the operator's parse; only the flag tells."""
    with pytest.raises(LLMError, match="refused"):
        await _chat(_completion(content='{"text": "I cannot help."}', refusal="I cannot help."))


async def test_explicit_null_refusal_returns_content() -> None:
    """The refusal check must not disturb an ordinary reply that reports ``refusal: null``."""
    reply, _ = await _chat(_completion(refusal=None))

    assert reply == '{"text": "hi there"}'


async def test_null_content_without_refusal_raises() -> None:
    """Absent content is not an empty reply: returning ``""`` would send the operator after a parse error."""
    with pytest.raises(LLMError, match="no content"):
        await _chat(_completion(content=None))


# ---------------------------------------------------------------------------
# Reported truncation or abnormal stop — never returned, even as a closed valid object
# ---------------------------------------------------------------------------

_CLOSED_AND_VALID = '{"text": "partial"}'


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
async def test_a_cut_short_generation_raises_even_with_closed_valid_json(finish_reason: str) -> None:
    assert _Answer.model_validate_json(_CLOSED_AND_VALID).text == "partial"

    with pytest.raises(LLMError, match=f"finish_reason='{finish_reason}'"):
        await _chat(_completion(content=_CLOSED_AND_VALID, finish_reason=finish_reason))


@pytest.mark.parametrize("finish_reason", ["abort", "error"])
async def test_an_aborted_generation_is_resent_with_the_identical_body(
    finish_reason: str, backoff_delays: list[float]
) -> None:
    aborted = httpx.Response(200, json=_completion(content=_CLOSED_AND_VALID, finish_reason=finish_reason))

    first, second = await _chat_through(aborted, _ok())

    assert first.content == second.content
    assert len(backoff_delays) == 1


@pytest.mark.parametrize("finish_reason", ["abort", "error"])
async def test_aborted_generations_raise_once_the_transport_retries_run_out(
    finish_reason: str, backoff_delays: list[float]
) -> None:
    aborted = httpx.Response(200, json=_completion(content=_CLOSED_AND_VALID, finish_reason=finish_reason))

    with pytest.raises(LLMError, match=f"finish_reason='{finish_reason}'"):
        await _chat_through(aborted, aborted, _ok(), client=_client(transport_retries=1))

    assert len(backoff_delays) == 1


async def test_an_aborted_refusal_raises_without_a_resend(backoff_delays: list[float]) -> None:
    """A refusal must raise (§3.4); the abort flag beside it does not turn it into a transient failure."""
    refusal = httpx.Response(200, json=_completion(content=None, refusal="I cannot help.", finish_reason="abort"))

    with pytest.raises(LLMError, match="refused"):
        await _chat_through(refusal, _ok())

    assert backoff_delays == []


@pytest.mark.parametrize("finish_reason", ["stop", None, "tool_calls"])
async def test_other_or_missing_finish_reasons_return_the_reply(finish_reason: str | None) -> None:
    """A missing finish reason needs no inference; validating the reply decides."""
    reply, _ = await _chat(_completion(finish_reason=finish_reason))

    assert reply == '{"text": "hi there"}'


# ---------------------------------------------------------------------------
# Reply validation, the contract's recommendations (§7) — followed so the reference client is a complete example
# ---------------------------------------------------------------------------


async def test_an_invalid_reply_is_retried_with_the_identical_request() -> None:
    outcome, bodies = await _chat_against(_completion(content="not json"), _completion(), client=_client(max_tokens=64))

    assert outcome == '{"text": "hi there"}'
    assert len(bodies) == 2
    assert bodies[0] == bodies[1], "a retry must not alter the request, max_tokens included"


async def test_invalid_replies_raise_once_the_retries_run_out() -> None:
    client = _client(invalid_reply_retries=2)

    outcome, bodies = await _chat_against(*(_completion(content="{}") for _ in range(3)), client=client)

    assert isinstance(outcome, LLMError)
    assert isinstance(outcome.__cause__, ValidationError)
    assert len(bodies) == 3


async def test_zero_retries_sends_a_single_request() -> None:
    outcome, bodies = await _chat_against(_completion(content="{}"), client=_client(invalid_reply_retries=0))

    assert isinstance(outcome, LLMError)
    assert len(bodies) == 1


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(_completion(content=None, refusal="I cannot help."), id="refusal"),
        pytest.param(_completion(content='{"text": "cut', finish_reason="length"), id="truncation"),
    ],
)
async def test_a_reported_refusal_or_cut_short_generation_is_not_retried(payload: dict[str, object]) -> None:
    """Retrying either would spend the caller's budget on a request that fails for the same reason."""
    outcome, bodies = await _chat_against(payload, _completion())

    assert isinstance(outcome, LLMError)
    assert len(bodies) == 1


async def test_each_failed_validation_is_logged_with_its_attempt(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="everalgo.llm.openai_compat")

    await _chat_against(_completion(content="not json"), _completion())

    assert [record.getMessage() for record in caplog.records] == [
        "Reply failed validation against _Answer (attempt 1 of 2, model=gpt-4o-mini, response_format_mode=json_schema, 1 errors)"
    ]


# ---------------------------------------------------------------------------
# Messages reach the wire whole, and cancellation propagates
# ---------------------------------------------------------------------------


async def test_non_text_parts_reach_the_request() -> None:
    parts: list[TextPart | ImageUrlPart] = [
        TextPart(text="describe"),
        ImageUrlPart(image_url=ImageUrlInner(url="data:image/png;base64,AAA")),
        ImageUrlPart(image_url=ImageUrlInner(url="data:application/pdf;base64,BBB")),
    ]

    _, sent = await _chat(_completion(), messages=[LLMMessage(role="user", content=parts)])

    content = sent["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "describe"}
    assert [part["image_url"]["url"] for part in content[1:]] == [
        "data:image/png;base64,AAA",
        "data:application/pdf;base64,BBB",
    ]


async def test_cancellation_propagates_as_cancelled_error() -> None:
    client = _client()
    reached = asyncio.Event()

    async def hang(request: httpx.Request) -> httpx.Response:
        reached.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    # A cancelled call never completes, so respx does not count the route as called.
    with respx.mock(base_url=_BASE_URL, assert_all_called=False) as router:
        router.post("/chat/completions").mock(side_effect=hang)
        task = asyncio.create_task(client.achat([LLMMessage(role="user", content="hi")], response_format=_Answer))
        await asyncio.wait_for(reached.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
