"""External service clients: LLM (OpenRouter), Embedding (DeepInfra), Rerank (DeepInfra).

Thin minimal clients with exponential-backoff retry. Not production-grade --
sized for one-shot benchmark runs.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from benchmarks.common.config import BenchmarkConfig


class ChatResponse(BaseModel):
    """Minimal chat completion response shape."""

    model_config = ConfigDict(frozen=True)

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    parsed: BaseModel | None = None


async def _retry_with_backoff[T](
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Exponential backoff retry: 1s, 2s, 4s.

    Retries on httpx.HTTPError (network) and httpx.HTTPStatusError (5xx).
    Lets non-retryable errors (4xx) propagate immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await fn()
        except httpx.HTTPStatusError as exc:
            # Only retry server errors (5xx); 4xx is caller's fault, fail fast
            if exc.response.status_code < 500:
                raise
            last_exc = exc
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            last_exc = exc
        if attempt < max_retries - 1:
            await asyncio.sleep(base_delay * (2**attempt))
    raise RuntimeError(f"Retries exhausted after {max_retries} attempts") from last_exc


class LLMClient:
    """OpenRouter chat-completion client (raw httpx; no openai SDK dependency)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        max_retries: int = 3,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    @classmethod
    def from_config(cls, cfg: BenchmarkConfig) -> LLMClient:
        """Construct from BenchmarkConfig; raises RuntimeError if API key is missing."""
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        return cls(
            api_key=key,
            base_url=cfg.llm_base_url,
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            timeout=cfg.llm_timeout,
            max_retries=cfg.llm_max_retries,
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> ChatResponse:
        """Send a chat completion request and return the assistant response.

        When ``response_format`` is a Pydantic ``BaseModel`` subclass, the request is sent
        to OpenRouter with the ``json_schema`` structured-outputs format, and the response
        content is validated and deserialized into ``ChatResponse.parsed``.
        """

        async def call() -> dict[str, Any]:
            payload: dict[str, Any] = {
                "model": model or self._model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self._temperature,
                "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
            }
            if response_format is not None:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "strict": True,
                        "schema": response_format.model_json_schema(),
                    },
                }
            r = await self._client.post("/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()  # type: ignore[no-any-return]

        data = await _retry_with_backoff(call, max_retries=self._max_retries)
        choice = data["choices"][0]
        usage = data.get("usage", {})
        content = choice["message"]["content"] or ""

        parsed: BaseModel | None = None
        if response_format is not None:
            parsed = response_format.model_validate_json(content)

        return ChatResponse(
            content=content,
            model=data.get("model", self._model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            parsed=parsed,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


class EmbeddingClient:
    """DeepInfra embedding HTTP client (OpenAI-compatible /v1/openai/embeddings endpoint).

    Wraps the Qwen/Qwen3-Embedding-4B model exposed by DeepInfra's OpenAI-compat
    surface. Single request batches up to N texts; preserves input order via the
    response's ``index`` field.
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

        async def call() -> dict[str, Any]:
            # ``dimensions`` is the Matryoshka truncation knob — Qwen3-Embedding-4B's full
            # output is 2560-dim; DeepInfra truncates server-side when ``dimensions`` is
            # passed. Mirrors EverCore main ``vectorize_base.py:117-118`` which always
            # forwards ``HybridVectorizeConfig.dimensions=1024`` to DeepInfra.
            payload: dict[str, Any] = {"input": texts, "model": self._model}
            if self._dimensions > 0:
                payload["dimensions"] = self._dimensions
            r = await self._client.post("/embeddings", json=payload)
            r.raise_for_status()
            return r.json()  # type: ignore[no-any-return]

        data = await _retry_with_backoff(call)
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
    """Format query and documents with the Qwen3-Reranker chat template.

    Ported verbatim from EverCore's ``agentic_layer.rerank_deepinfra._format_rerank_texts``
    to preserve the scoring behavior the EverCore 92.32% LoCoMo baseline depends on.
    """
    instr = instruction or _DEFAULT_RERANK_INSTRUCTION
    formatted_query = f"{_QWEN3_PREFIX}<Instruct>: {instr}\n<Query>: {query}\n"
    formatted_docs = [f"<Document>: {doc}{_QWEN3_SUFFIX}" for doc in documents]
    return [formatted_query], formatted_docs


def _extract_scores(json_body: dict[str, Any], *, expected_len: int) -> list[float]:
    """Normalize the two DeepInfra response shapes into a flat list of scores.

    Ports the dual-shape parser from EverCore's ``_parse_response``.
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

        async def call() -> dict[str, Any]:
            r = await self._client.post(self._url, json=payload)
            r.raise_for_status()
            return r.json()  # type: ignore[no-any-return]

        data = await _retry_with_backoff(call)
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

    llm: LLMClient
    embedding: EmbeddingClient
    rerank: RerankClient

    @classmethod
    def from_config(cls, cfg: BenchmarkConfig) -> Services:
        """Construct all three clients from a shared BenchmarkConfig."""
        return cls(
            llm=LLMClient.from_config(cfg),
            embedding=EmbeddingClient.from_config(cfg),
            rerank=RerankClient.from_config(cfg),
        )
