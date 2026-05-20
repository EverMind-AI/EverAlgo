"""Tests for LLMClient and Services bundle."""

from unittest.mock import patch

import httpx
import pytest
import respx
from pytest import MonkeyPatch

from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.services import LLMClient, Services


@pytest.fixture
def cfg() -> BenchmarkConfig:
    return BenchmarkConfig()


@pytest.mark.asyncio
async def test_services_bundle_holds_three_clients(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    svcs = Services.from_config(cfg)
    try:
        assert svcs.llm is not None
        assert svcs.embedding is not None
        assert svcs.rerank is not None
    finally:
        await svcs.llm.close()


def test_llm_from_config_requires_api_key(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        LLMClient.from_config(cfg)


@respx.mock
@pytest.mark.asyncio
async def test_llm_chat_passes_temperature_zero(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "openai/gpt-4.1-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )
    )
    llm = LLMClient.from_config(cfg)
    resp = await llm.chat([{"role": "user", "content": "ping"}])
    assert resp.content == "hi"
    assert resp.prompt_tokens == 5
    assert resp.completion_tokens == 1
    assert route.called
    body: bytes = route.calls[0].request.read()  # type: ignore[union-attr]
    assert b'"temperature":0' in body or b'"temperature": 0' in body
    await llm.close()


@respx.mock
@pytest.mark.asyncio
async def test_llm_chat_per_call_temperature_override(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    """Caller can override the bound temperature via per-call temperature kwarg."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "model": "openai/gpt-4.1-mini",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
    llm = LLMClient.from_config(cfg)  # bound temperature is 0.3
    await llm.chat([{"role": "user", "content": "ping"}], temperature=0.0)
    body: bytes = route.calls[0].request.read()  # type: ignore[union-attr]
    # temperature=0.0 must be serialized in the payload, not the default 0.3
    assert (
        b'"temperature": 0.0' in body
        or b'"temperature":0.0' in body
        or b'"temperature": 0' in body
        or b'"temperature":0' in body
    )
    assert b'"temperature": 0.3' not in body and b'"temperature":0.3' not in body
    await llm.close()


@respx.mock
@pytest.mark.asyncio
async def test_llm_retry_on_transient_error(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    # First two calls fail, third succeeds
    responses = [
        httpx.Response(500),
        httpx.Response(500),
        httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "model": "openai/gpt-4.1-mini",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        ),
    ]
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(side_effect=responses)
    llm = LLMClient.from_config(cfg)
    with patch("benchmarks.common.services.asyncio.sleep") as mock_sleep:
        resp = await llm.chat([{"role": "user", "content": "ping"}])
    assert resp.content == "ok"
    assert route.call_count == 3
    # Verify sleep was called with expected backoff sequence
    assert [call.args[0] for call in mock_sleep.call_args_list] == [1.0, 2.0]
    await llm.close()
