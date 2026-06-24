"""External service clients: LLM (OpenRouter), Embedding (DeepInfra), Rerank (DeepInfra).

Thin minimal clients with exponential-backoff retry. Not production-grade --
sized for one-shot benchmark runs.

``LLMClient`` is the algo ``OpenAICompatClient`` (via ``everalgo.llm.providers.openai_compat``).
``EmbeddingClient`` and ``RerankClient`` are benchmark-private model clients; they use the same
DeepInfra HTTP surface and are not in the algo surface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import SecretStr

from benchmarks.common.retry import http_retry
from everalgo.llm.config import LLMConfig
from everalgo.llm.providers.openai_compat import OpenAICompatClient

if TYPE_CHECKING:
    from benchmarks.common.config import BenchmarkConfig


def build_llm_client(cfg: BenchmarkConfig, *, model: str | None = None) -> OpenAICompatClient:
    """Construct the algo OpenAI-compatible LLM client from BenchmarkConfig.

    Args:
        cfg: Benchmark configuration supplying base URL, temperature, and token limits.
        model: Model identifier override. When ``None``, defaults to ``cfg.extract_model``
            (covers stages 1-3 and agentic search). Pass ``cfg.answer_model`` explicitly
            for the stage-4 answer client.

    Raises:
        RuntimeError: When ``OPENROUTER_API_KEY`` is not set.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return OpenAICompatClient(
        LLMConfig(
            api_key=SecretStr(key),
            base_url=cfg.llm_base_url,
            model=model if model is not None else cfg.extract_model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            timeout=cfg.llm_timeout,
        )
    )


# ---------------------------------------------------------------------------
# Benchmark-private model clients — not in the algo surface
# ---------------------------------------------------------------------------


class EmbeddingClient:
    """DeepInfra embedding HTTP client (OpenAI-compatible /v1/openai/embeddings endpoint).

    Wraps the Qwen/Qwen3-Embedding-4B model exposed by DeepInfra's OpenAI-compat
    surface. Single request batches up to N texts; preserves input order via the
    response's ``index`` field.

    Args:
        api_key: DeepInfra API key.
        base_url: Base URL for the embedding endpoint.
        model: Model identifier (e.g. ``"Qwen/Qwen3-Embedding-4B"``).
        timeout: HTTP timeout in seconds.
        dimensions: Matryoshka truncation dimension (0 = full output).
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
        dimensions: int = 1024,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._model = model
        self._dimensions = dimensions

    @classmethod
    def from_config(cls, cfg: BenchmarkConfig) -> EmbeddingClient:
        """Build client from benchmark config; raises if DEEPINFRA_API_KEY missing."""
        key = os.environ.get("DEEPINFRA_API_KEY")
        if not key:
            raise RuntimeError("DEEPINFRA_API_KEY not set")
        return cls(
            api_key=key,
            base_url=cfg.embedding_base_url,
            model=cfg.embedding_model,
            timeout=cfg.llm_timeout,
            dimensions=cfg.embedding_dimensions,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns vectors in the input order."""
        if not texts:
            return []

        @http_retry()
        async def _call() -> dict[str, Any]:
            # ``dimensions`` is the Matryoshka truncation knob — Qwen3-Embedding-4B's full
            # output is 2560-dim; DeepInfra truncates server-side when ``dimensions`` is passed.
            # ``encoding_format="float"`` is semantically a no-op (OpenAI /embeddings defaults to
            # ``"float"`` when omitted) but kept explicit for consistent request format.
            payload: dict[str, Any] = {
                "input": texts,
                "model": self._model,
                "encoding_format": "float",
            }
            if self._dimensions > 0:
                payload["dimensions"] = self._dimensions
            r = await self._client.post("/embeddings", json=payload)
            r.raise_for_status()
            return r.json()  # type: ignore[no-any-return]

        data: dict[str, Any] = await _call()
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


_QWEN3_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_DEFAULT_RERANK_INSTRUCTION = (
    "Given a question and a passage, determine if the passage contains information relevant to answering the question."
)


def _format_rerank_inputs(query: str, documents: list[str], instruction: str | None) -> tuple[list[str], list[str]]:
    """Format query and documents with the Qwen3-Reranker chat template."""
    instr = instruction or _DEFAULT_RERANK_INSTRUCTION
    formatted_query = f"{_QWEN3_PREFIX}<Instruct>: {instr}\n<Query>: {query}\n"
    formatted_docs = [f"<Document>: {doc}{_QWEN3_SUFFIX}" for doc in documents]
    return [formatted_query], formatted_docs


def _extract_scores(json_body: dict[str, Any], *, expected_len: int) -> list[float]:
    """Normalize the two DeepInfra response shapes into a flat list of scores.

    Ports the dual-shape parser from the upstream reference's ``_parse_response``.
    """
    if "results" in json_body:
        results = list(json_body["results"])
        results.sort(key=lambda x: x.get("index", 0))
        scores = [float(item.get("relevance_score", 0.0)) for item in results]
    elif "scores" in json_body:
        scores = [float(s) for s in json_body["scores"]]
    else:
        scores = []
    # Pad if API returned fewer scores than expected (defensive)
    if len(scores) < expected_len:
        scores = scores + [0.0] * (expected_len - len(scores))
    return scores[:expected_len]


class RerankClient:
    """DeepInfra Qwen3-Reranker-4B client.

    Posts to ``<base_url>/<model>`` (e.g.
    ``https://api.deepinfra.com/v1/inference/Qwen/Qwen3-Reranker-4B``).
    Query and documents are wrapped with the Qwen3 chat-template prefix/suffix;
    response shape is normalized across the two DeepInfra variants
    (``{"results": [...]}`` vs ``{"scores": [...]}``).

    Args:
        api_key: DeepInfra API key.
        base_url: Base URL for the inference endpoint.
        model: Model identifier (e.g. ``"Qwen/Qwen3-Reranker-4B"``).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        # Build full URL once: base_url + "/" + model
        self._url = f"{base_url.rstrip('/')}/{model}"
        self._model = model

    @classmethod
    def from_config(cls, cfg: BenchmarkConfig) -> RerankClient:
        """Build client from config; raises if DEEPINFRA_API_KEY missing."""
        key = os.environ.get("DEEPINFRA_API_KEY")
        if not key:
            raise RuntimeError("DEEPINFRA_API_KEY not set")
        return cls(
            api_key=key,
            base_url=cfg.deepinfra_base_url,
            model=cfg.reranker_model,
            timeout=cfg.reranker_timeout,
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        instruction: str | None = None,
    ) -> list[tuple[int, float]]:
        """Score documents by query relevance. Returns (orig_index, score) sorted descending."""
        if not documents:
            return []

        queries, formatted_docs = _format_rerank_inputs(query, documents, instruction)
        payload = {"queries": queries, "documents": formatted_docs}

        @http_retry()
        async def _call() -> dict[str, Any]:
            r = await self._client.post(self._url, json=payload)
            r.raise_for_status()
            return r.json()  # type: ignore[no-any-return]

        data: dict[str, Any] = await _call()
        scores = _extract_scores(data, expected_len=len(documents))
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


@dataclass(frozen=True)
class Services:
    """Bundle of external service clients."""

    llm: OpenAICompatClient
    embedding: EmbeddingClient
    rerank: RerankClient

    @classmethod
    def from_config(cls, cfg: BenchmarkConfig) -> Services:
        """Construct all three clients from a shared BenchmarkConfig."""
        return cls(
            llm=build_llm_client(cfg),
            embedding=EmbeddingClient.from_config(cfg),
            rerank=RerankClient.from_config(cfg),
        )

    async def close(self) -> None:
        """Close all underlying HTTP clients."""
        await self.llm._client.close()
        await self.embedding.close()
        await self.rerank.close()
