"""Tests for benchmarks.common.retry — tenacity factory functions."""

import json

import httpx
import pytest
from tenacity import wait_none

from benchmarks.common.retry import answer_retry, http_retry, llm_retry


async def test_llm_retry_succeeds_on_first_try():
    call_count = 0

    @llm_retry(max_attempts=3, wait=wait_none())
    async def ok():
        nonlocal call_count
        call_count += 1
        return {"key": "value"}

    result = await ok()
    assert result == {"key": "value"}
    assert call_count == 1


async def test_llm_retry_recovers_after_json_error():
    call_count = 0

    @llm_retry(max_attempts=3, wait=wait_none())
    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise json.JSONDecodeError("bad", "", 0)
        return {"ok": True}

    result = await flaky()
    assert result == {"ok": True}
    assert call_count == 2


async def test_llm_retry_exhausts_then_raises():
    @llm_retry(max_attempts=2, wait=wait_none())
    async def always_fails():
        raise ValueError("parse error")

    with pytest.raises(ValueError, match="parse error"):
        await always_fails()


async def test_llm_retry_does_not_catch_unrelated_exceptions():
    @llm_retry(max_attempts=3, wait=wait_none())
    async def type_error():
        raise TypeError("not retryable")

    with pytest.raises(TypeError, match="not retryable"):
        await type_error()


async def test_http_retry_retries_on_5xx():
    call_count = 0

    @http_retry(max_attempts=3, wait=wait_none())
    async def flaky_http():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            response = httpx.Response(502, request=httpx.Request("POST", "https://x"))
            raise httpx.HTTPStatusError("bad gateway", request=response.request, response=response)
        return "ok"

    result = await flaky_http()
    assert result == "ok"
    assert call_count == 2


async def test_http_retry_does_not_retry_4xx():
    @http_retry(max_attempts=3, wait=wait_none())
    async def client_error():
        response = httpx.Response(422, request=httpx.Request("POST", "https://x"))
        raise httpx.HTTPStatusError("unprocessable", request=response.request, response=response)

    with pytest.raises(httpx.HTTPStatusError):
        await client_error()


async def test_answer_retry_retries_on_empty_result():
    call_count = 0

    @answer_retry(max_attempts=3, wait=wait_none())
    async def sometimes_empty():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return ""
        return "the answer"

    result = await sometimes_empty()
    assert result == "the answer"
    assert call_count == 3


async def test_answer_retry_retries_on_exception():
    call_count = 0

    @answer_retry(max_attempts=3, wait=wait_none())
    async def flaky_answer():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("transient")
        return "recovered"

    result = await flaky_answer()
    assert result == "recovered"
    assert call_count == 2
