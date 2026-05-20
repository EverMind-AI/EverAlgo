"""Tests for DeepInfra RerankClient."""

from unittest.mock import patch

import httpx
import pytest
import respx
from pytest import MonkeyPatch

from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.services import RerankClient


@pytest.fixture
def cfg() -> BenchmarkConfig:
    return BenchmarkConfig()


def test_rerank_from_config_requires_api_key(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    """Missing DEEPINFRA_API_KEY must raise."""
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPINFRA_API_KEY"):
        RerankClient.from_config(cfg)


@respx.mock
@pytest.mark.asyncio
async def test_rerank_results_shape_returns_sorted_index_score(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    """{'results': [{index, relevance_score}, ...]} shape — return (idx, score) sorted desc."""
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    respx.post("https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-4B").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"index": 0, "relevance_score": 0.1},
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 2, "relevance_score": 0.5},
                ],
            },
        )
    )
    client = RerankClient.from_config(cfg)
    try:
        out = await client.rerank("q", ["a", "b", "c"], instruction="match")
        assert out == [(1, 0.9), (2, 0.5), (0, 0.1)]
    finally:
        await client.close()


@respx.mock
@pytest.mark.asyncio
async def test_rerank_scores_shape_returns_sorted_index_score(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    """{'scores': [0.1, 0.9, 0.5]} shape — return (idx, score) sorted desc."""
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    respx.post("https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-4B").mock(
        return_value=httpx.Response(200, json={"scores": [0.1, 0.9, 0.5]})
    )
    client = RerankClient.from_config(cfg)
    try:
        out = await client.rerank("q", ["a", "b", "c"])
        assert out == [(1, 0.9), (2, 0.5), (0, 0.1)]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_rerank_empty_documents_returns_empty(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    """Empty documents must short-circuit (no HTTP)."""
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    client = RerankClient.from_config(cfg)
    try:
        assert await client.rerank("q", []) == []
    finally:
        await client.close()


@respx.mock
@pytest.mark.asyncio
async def test_rerank_request_includes_qwen3_template(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    """Verify query+docs are formatted with Qwen3 chat template before sending."""
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    route = respx.post("https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-4B").mock(
        return_value=httpx.Response(200, json={"scores": [0.5]})
    )
    client = RerankClient.from_config(cfg)
    try:
        await client.rerank("what is X?", ["doc body"], instruction="match facts")
        body: bytes = route.calls[0].request.read()  # type: ignore[union-attr,index]
        # Qwen3 template markers
        assert b"<|im_start|>system" in body
        assert b"<Instruct>: match facts" in body
        assert b"<Query>: what is X?" in body
        assert b"<Document>: doc body" in body
        assert b"<|im_end|>" in body
    finally:
        await client.close()


@respx.mock
@pytest.mark.asyncio
async def test_rerank_retries_on_5xx(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    """Transient 5xx is retried; backoff sleep mocked."""
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    responses = [
        httpx.Response(500),
        httpx.Response(200, json={"scores": [0.7]}),
    ]
    route = respx.post("https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-4B").mock(side_effect=responses)
    client = RerankClient.from_config(cfg)
    try:
        with patch("benchmarks.common.services.asyncio.sleep") as mock_sleep:
            out = await client.rerank("q", ["x"])
        assert out == [(0, 0.7)]
        assert route.call_count == 2
        assert [c.args[0] for c in mock_sleep.call_args_list] == [1.0]
    finally:
        await client.close()
