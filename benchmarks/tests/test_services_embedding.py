"""Tests for DeepInfra EmbeddingClient."""

from unittest.mock import patch

import httpx
import pytest
import respx
from pytest import MonkeyPatch

from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.services import EmbeddingClient


@pytest.fixture
def cfg() -> BenchmarkConfig:
    return BenchmarkConfig()


def test_embedding_from_config_requires_api_key(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    """Missing DEEPINFRA_API_KEY must raise."""
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPINFRA_API_KEY"):
        EmbeddingClient.from_config(cfg)


@respx.mock
@pytest.mark.asyncio
async def test_embed_returns_vectors_in_input_order(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    """Returned vectors must align with the order of input texts."""
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    respx.post("https://api.deepinfra.com/v1/openai/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.1, 0.2], "index": 0},
                    {"embedding": [0.3, 0.4], "index": 1},
                ],
                "model": "Qwen/Qwen3-Embedding-4B",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )
    )
    client = EmbeddingClient.from_config(cfg)
    try:
        out = await client.embed(["hello", "world"])
        assert len(out) == 2
        assert out[0] == [0.1, 0.2]
        assert out[1] == [0.3, 0.4]
    finally:
        await client.close()


@respx.mock
@pytest.mark.asyncio
async def test_embed_sorts_by_index_when_api_returns_out_of_order(
    cfg: BenchmarkConfig, monkeypatch: MonkeyPatch
) -> None:
    """If DeepInfra returns embeddings out of input order, sort by .index."""
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    respx.post("https://api.deepinfra.com/v1/openai/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"embedding": [0.3, 0.4], "index": 1},  # out of order
                    {"embedding": [0.1, 0.2], "index": 0},
                ],
                "model": "Qwen/Qwen3-Embedding-4B",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )
    )
    client = EmbeddingClient.from_config(cfg)
    try:
        out = await client.embed(["hello", "world"])
        assert out[0] == [0.1, 0.2]
        assert out[1] == [0.3, 0.4]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_embed_empty_input_returns_empty(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    """Empty input list must short-circuit without an HTTP call."""
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    client = EmbeddingClient.from_config(cfg)
    try:
        out = await client.embed([])
        assert out == []
    finally:
        await client.close()


@respx.mock
@pytest.mark.asyncio
async def test_embed_retries_on_5xx(cfg: BenchmarkConfig, monkeypatch: MonkeyPatch) -> None:
    """Transient 5xx must be retried; verify exponential backoff sleep sequence."""
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    responses = [
        httpx.Response(500),
        httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.5], "index": 0}],
                "model": "Qwen/Qwen3-Embedding-4B",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        ),
    ]
    route = respx.post("https://api.deepinfra.com/v1/openai/embeddings").mock(side_effect=responses)
    client = EmbeddingClient.from_config(cfg)
    try:
        with patch("benchmarks.common.services.asyncio.sleep") as mock_sleep:
            out = await client.embed(["x"])
        assert out == [[0.5]]
        assert route.call_count == 2
        # First retry waits base_delay * 2**0 = 1.0
        assert [c.args[0] for c in mock_sleep.call_args_list] == [1.0]
    finally:
        await client.close()
