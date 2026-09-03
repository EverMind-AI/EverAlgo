"""Stage 5 — Retrieval primitives + agentic loop + main entry.

Provides pure scoring/fusion primitives (BM25, embedding MaxSim, RRF) and a
unified retrieval path that delegates to the algo facade:
- ``ahybrid_retrieve`` — dense+sparse RRF base.
- ``acluster_retrieve`` — cluster-scoped R1 narrowed on top of hybrid.
- ``aagentic_retrieve`` — LLM-guided sufficiency + multi-query/refined-query loop.

Both branches (cluster-enabled and plain-hybrid) share one ``_attempt_single_qa``
implementation; the cluster branch passes ``cluster_scoped`` as ``base_retrieve``
and ``hybrid_full`` as ``round2_retrieve`` so Round 2 spills out to the full corpus.

Security: this module uses ``pickle`` to load benchmark index artifacts
(``bm25_conv_*.pkl`` / ``emb_conv_*.pkl`` / ``cluster_index_conv_*.pkl``)
produced by Stage 4 in the same trusted local workspace. **Never point the
``input_dir`` at untrusted / network-shared paths** — a tampered ``.pkl`` can
execute arbitrary code on load. The ``benchmarks/`` package is internal-only
(``[tool.uv] package = false``) and is not published to PyPI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import time
import traceback
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from nltk.corpus import stopwords  # type: ignore[import-untyped]
from nltk.stem import PorterStemmer  # type: ignore[import-untyped]

from benchmarks.common.retry import http_retry
from benchmarks.common.stages._tokenize import ensure_nltk as _ensure_nltk
from benchmarks.common.stages._tokenize import tokenize as _tokenize
from everalgo.clustering import Cluster
from everalgo.rank import aagentic_retrieve, acluster_retrieve, ahybrid_retrieve, amaxsim_retrieve
from everalgo.types import Candidate as _Candidate

if TYPE_CHECKING:
    from benchmarks.common.services import EmbeddingClient, RerankClient
    from benchmarks.common.stages.types import StageContext, StageStats
    from everalgo.llm.providers.openai_compat import OpenAICompatClient
    from everalgo.llm.types import ChatMessage, ChatResponse
    from everalgo.rank.protocols import RerankFn, RetrieveFn

logger = logging.getLogger(__name__)

# Convenience alias — every document dict is untyped at runtime.
_Doc = dict[str, Any]
_Scored = tuple[_Doc, float]


# ---------------------------------------------------------------------------
# Candidate ↔ (doc, score) conversion helpers
# ---------------------------------------------------------------------------


def _doc_to_candidate(doc: _Doc, score: float) -> _Candidate:
    """Wrap a ``(doc, score)`` pair as an algo ``Candidate``.

    The original doc dict is stored in ``metadata["_doc"]`` for round-trip recovery.
    The ``episode`` field is restructured into a nested dict matching the contract
    expected by ``everalgo.rank.agentic._format_docs`` (``episode.subject`` + ``episode.content``).
    """
    doc_id = str(doc.get("id", "")) if doc.get("id") is not None else ""
    meta: dict[str, Any] = {"_doc": doc, **doc}
    # Algo's _format_docs expects metadata["episode"] = {"subject": ..., "content": ...}
    # but the entity-split model stores episode text as a flat string field.
    if isinstance(meta.get("episode"), str):
        meta["episode"] = {"subject": meta.get("subject", ""), "content": meta["episode"]}
    return _Candidate(id=doc_id, score=float(score), metadata=meta)


def _candidate_to_scored(c: _Candidate) -> _Scored:
    """Unwrap a ``Candidate`` back to ``(doc, score)``.

    Prefers the original doc from ``metadata["_doc"]`` (round-trip fidelity).
    Falls back to reconstructing a minimal doc from the candidate's metadata when
    ``_doc`` is absent (e.g. algo internally created candidates).
    """
    raw_doc = c.metadata.get("_doc")
    if isinstance(raw_doc, dict):
        return cast("_Doc", raw_doc), float(c.score)
    # Fallback: reconstruct from metadata (minus internal keys)
    doc: _Doc = {k: v for k, v in c.metadata.items() if not k.startswith("_")}
    doc.setdefault("id", c.id)
    return doc, float(c.score)


def _scored_to_candidates(results: list[_Scored]) -> list[_Candidate]:
    """Convert a ``[(doc, score), ...]`` list to ``list[Candidate]``."""
    return [_doc_to_candidate(doc, score) for doc, score in results]


def _candidates_to_scored(candidates: list[_Candidate]) -> list[_Scored]:
    """Convert a ``list[Candidate]`` to ``[(doc, score), ...]``."""
    return [_candidate_to_scored(c) for c in candidates]


# ---------------------------------------------------------------------------
# Token-counting LLM proxy
# ---------------------------------------------------------------------------


class _TokenCountingLLM:
    """Thin proxy around ``OpenAICompatClient`` that accumulates token usage.

    Wraps every ``chat`` call to sum up ``usage.prompt_tokens`` and
    ``usage.completion_tokens`` into ``self.prompt_tokens`` / ``self.completion_tokens``.
    This lets ``_process_single_qa`` report token totals for the retrieval stage
    while still delegating all logic to the algo.

    Args:
        inner: The underlying LLM client to proxy.
    """

    def __init__(self, inner: OpenAICompatClient) -> None:
        self._inner = inner
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Forward to the inner client and accumulate token counts."""
        resp = await self._inner.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        if resp.usage is not None:
            self.prompt_tokens += resp.usage.prompt_tokens or 0
            self.completion_tokens += resp.usage.completion_tokens or 0
        return resp


def compute_maxsim_score(
    query_emb: np.ndarray,
    atomic_fact_embs: list[np.ndarray],
) -> float:
    """Compute maximum similarity between query and multiple atomic_fact embeddings.

    Uses the MaxSim strategy: find the single most relevant atomic_fact to the
    query. Vectorized matrix operations give a 2-3x speed boost over a naive loop.

    Args:
        query_emb: Query embedding vector (1-D numpy array).
        atomic_fact_embs: List of atomic_fact embedding vectors.

    Returns:
        Maximum cosine similarity score (float, range [-1, 1], typically [0, 1]).
        Returns 0.0 when the list is empty or the query vector is zero.
    """
    if not atomic_fact_embs:
        return 0.0

    query_norm = np.linalg.norm(query_emb)
    if query_norm == 0:
        return 0.0

    try:
        fact_matrix = np.array(atomic_fact_embs)
        fact_norms = np.linalg.norm(fact_matrix, axis=1)

        valid_mask = fact_norms > 0
        if not np.any(valid_mask):
            return 0.0

        dot_products = np.dot(fact_matrix[valid_mask], query_emb)
        sims = dot_products / (query_norm * fact_norms[valid_mask])
        return float(np.max(sims))

    except Exception:
        # Fall back to loop (compatibility guarantee)
        similarities: list[float] = []
        for fact_emb in atomic_fact_embs:
            fact_norm = np.linalg.norm(fact_emb)
            if fact_norm == 0:
                continue
            sim = np.dot(query_emb, fact_emb) / (float(query_norm) * float(fact_norm))
            similarities.append(float(sim))
        return max(similarities) if similarities else 0.0


def search_with_bm25_index(
    query: str,
    bm25_index: dict[str, Any],
    top_n: int = 5,
) -> list[_Scored]:
    """Fact-level BM25 search with MaxSim aggregation to doc level.

    Reads the ``bm25_index`` payload
    (``{"bm25": BM25Okapi, "docs": [...], "fact_to_doc_idx": [int...]}``).
    ``bm25.get_scores`` returns one score per fact-row in the corpus; we map
    each fact back to its parent memcell and keep the maximum across the
    doc's facts (MaxSim).

    Args:
        query: Raw query text (tokenized internally).
        bm25_index: Fact-level BM25 payload as persisted by Stage 4.
        top_n: Maximum number of results to return.

    Returns:
        List of ``(doc, max_score)`` pairs sorted by score descending,
        length <= ``top_n``.
    """
    bm25 = bm25_index["bm25"]
    docs: list[_Doc] = bm25_index["docs"]
    fact_to_doc_idx: list[int] = bm25_index["fact_to_doc_idx"]

    _ensure_nltk()
    stemmer = PorterStemmer()
    stop_words: set[str] = set(stopwords.words("english"))  # type: ignore[no-untyped-call]
    tokenized_query = _tokenize(query, stemmer, stop_words)

    if not tokenized_query:
        return []

    fact_scores: Any = bm25.get_scores(tokenized_query)

    doc_max_score: dict[int, float] = {}
    for fact_idx, raw_score in enumerate(fact_scores):
        doc_idx = fact_to_doc_idx[fact_idx]
        score = float(raw_score)
        if score > doc_max_score.get(doc_idx, float("-inf")):
            doc_max_score[doc_idx] = score

    sorted_items = sorted(doc_max_score.items(), key=lambda kv: kv[1], reverse=True)
    return [(docs[d], s) for d, s in sorted_items[:top_n]]


def _score_emb_item(
    item: dict[str, Any],
    query_vec: np.ndarray,
) -> _Scored | None:
    """Score an embedding index entry via MaxSim — atomic_facts + subject only.

    Entity-split model: every episode is guaranteed to have atomic_facts embeddings. The score
    is the max cosine similarity over those facts. The ``subject`` vector provides a supplementary
    topic-level signal added to the MaxSim pool. No summary or episode content fallback.
    """
    doc_id: str = str(item.get("doc_id") or "")
    embeddings: dict[str, Any] = item.get("embeddings") or {}
    if not embeddings:
        return None

    pool: list[np.ndarray] = []

    atomic_fact_embs = embeddings.get("atomic_facts")
    if isinstance(atomic_fact_embs, list) and atomic_fact_embs:
        pool.extend(cast("list[np.ndarray]", atomic_fact_embs))

    subject_emb = embeddings.get("subject")
    if subject_emb is not None:
        pool.append(subject_emb)

    if not pool:
        return None

    doc: _Doc = {"id": doc_id}
    return (doc, compute_maxsim_score(query_vec, pool))


async def _resolve_query_vec(
    query: str,
    embedding_client: EmbeddingClient,
    query_embedding: np.ndarray | None,
) -> np.ndarray:
    """Return the query embedding, fetching via ``embedding_client`` if needed."""
    if query_embedding is not None:
        return query_embedding
    raw: list[list[float]] = await embedding_client.embed([query])
    return np.array(raw[0])


async def search_with_emb_index(
    query: str,
    emb_index: list[dict[str, Any]],
    *,
    top_n: int = 5,
    embedding_client: EmbeddingClient,
    query_embedding: np.ndarray | None = None,
) -> list[_Scored]:
    """Execute embedding retrieval using the MaxSim strategy.

    Scores each entry by MaxSim over ``atomic_facts`` embeddings + ``subject`` vector.
    No summary or episode content fallback.

    Args:
        query: Raw query text.
        emb_index: Pre-built embedding index; each entry is
            ``{"doc_id": str, "embeddings": {"atomic_facts": [...], "subject": vec, ...}}``.
        top_n: Maximum number of results to return.
        embedding_client: Service used to embed the query when ``query_embedding`` is not pre-provided.
        query_embedding: Optional pre-computed query embedding.

    Returns:
        List of (doc, score) pairs sorted by score descending, length <= top_n.
    """
    query_vec = await _resolve_query_vec(query, embedding_client, query_embedding)
    if float(np.linalg.norm(query_vec)) == 0:
        return []

    doc_scores: list[_Scored] = [
        result for item in emb_index if (result := _score_emb_item(item, query_vec)) is not None
    ]

    if not doc_scores:
        return []

    sorted_results = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


def _format_doc_for_rerank(doc: dict[str, Any]) -> str:
    """Format a doc for reranker input — ``episode`` text field, no fallback.

    In the entity-split model, ``doc["episode"]`` is a flat string field (the episode narrative).

    Raises:
        ValueError: If ``episode`` text is missing or empty — fail-loud so upstream
            schema regressions surface immediately.
    """
    content = doc.get("episode")
    if isinstance(content, str) and content.strip():
        return content.strip()
    raise ValueError(f"doc has no episode text for reranker: id={doc.get('id', 'unknown')}")


def _trace_scored(results: list[_Scored], *, limit: int) -> list[dict[str, Any]]:
    """Reduce a ``[(doc, score), ...]`` list to a compact trace shape for diagnostics.

    Stored shape per entry: ``{"id": <memcell_id_str>, "score": <float, 4 dp>}``.
    Limits length to bound dump size — typical limits: 30 for raw branch (BM25/Emb)
    top-N, 20 for hybrid/rerank top-N, 10 for final top-N. Used only for offline
    analysis of "did the gold memcell make it into round 1 / why did rerank drop it".
    """
    return [{"id": str(doc.get("id", "")), "score": round(float(score), 4)} for doc, score in results[:limit]]


# ---------------------------------------------------------------------------
# Reranker — batch cross-encoder with retry and fallback
# ---------------------------------------------------------------------------


async def _rerank_batch(
    query: str,
    start: int,
    batch_texts: list[str],
    *,
    rerank_client: RerankClient,
    reranker_instruction: str | None,
    max_retries: int,
    timeout: float,
) -> list[tuple[int, float]]:
    """Score a single reranker batch with tenacity retry.

    Args:
        query: Raw query text forwarded to the reranker.
        start: Offset of this batch in the full ``docs_with_text`` list.
        batch_texts: Document texts for this batch.
        rerank_client: RerankClient instance.
        reranker_instruction: Optional task instruction.
        max_retries: Maximum attempts before giving up.
        timeout: Per-attempt asyncio timeout (seconds).

    Returns:
        List of ``(global_index, score)`` on success.

    Raises:
        Exception: Propagated after all retries are exhausted.
    """

    @http_retry(max_attempts=max_retries)
    async def _call() -> list[tuple[int, float]]:
        scored: list[tuple[int, float]] = await asyncio.wait_for(
            rerank_client.rerank(query, batch_texts, instruction=reranker_instruction),
            timeout=timeout,
        )
        return [(start + idx, score) for idx, score in scored]

    return await _call()


def _format_docs_for_rerank(results: list[_Scored]) -> tuple[list[str], list[int]]:
    """Format candidate docs for reranker input, preserving original indices.

    Returns:
        Tuple of ``(doc_texts, original_indices)`` where each entry corresponds
        to a successfully formatted document.

    Raises:
        ValueError: Propagated from ``_format_doc_for_rerank`` when a doc lacks episode text.
    """
    doc_texts: list[str] = []
    original_indices: list[int] = []
    for orig_idx, (doc, _score) in enumerate(results):
        text = _format_doc_for_rerank(doc)
        if text:
            doc_texts.append(text)
            original_indices.append(orig_idx)
    return doc_texts, original_indices


async def _execute_rerank_batches(
    query: str,
    doc_texts: list[str],
    *,
    rerank_client: RerankClient,
    reranker_instruction: str | None,
    batch_size: int,
    concurrent_batches: int,
    max_retries: int,
    timeout: float,
    fallback_threshold: float,
) -> list[tuple[int, float]]:
    """Run reranker batches with concurrency control; fail-loud on low success rate.

    Raises:
        RuntimeError: When batch success rate falls below ``fallback_threshold``.
    """
    batches: list[tuple[int, list[str]]] = [
        (i, doc_texts[i : i + batch_size]) for i in range(0, len(doc_texts), batch_size)
    ]
    all_scored: list[tuple[int, float]] = []
    successful = 0

    for group_start in range(0, len(batches), concurrent_batches):
        group = batches[group_start : group_start + concurrent_batches]
        group_results = await asyncio.gather(
            *(
                _rerank_batch(
                    query,
                    start,
                    batch,
                    rerank_client=rerank_client,
                    reranker_instruction=reranker_instruction,
                    max_retries=max_retries,
                    timeout=timeout,
                )
                for start, batch in group
            ),
            return_exceptions=True,
        )
        for r in group_results:
            if isinstance(r, BaseException):
                logger.warning("Rerank batch failed after retries: %s", r)
            else:
                all_scored.extend(r)
                successful += 1
        if group_start + concurrent_batches < len(batches):
            await asyncio.sleep(0.3)

    success_rate = successful / len(batches) if batches else 0.0
    if not all_scored or success_rate < fallback_threshold:
        raise RuntimeError(
            f"reranker batch success rate {success_rate:.2f} below threshold {fallback_threshold:.2f} "
            f"({successful}/{len(batches)} batches succeeded)"
        )
    return all_scored


async def reranker_search(
    query: str,
    *,
    results: list[_Scored],
    rerank_client: RerankClient,
    top_n: int = 20,
    reranker_instruction: str | None = None,
    batch_size: int = 20,
    concurrent_batches: int = 5,
    max_retries: int = 3,
    timeout: float = 60.0,
    fallback_threshold: float = 0.3,
) -> list[_Scored]:
    """Rerank candidate docs with Qwen3 reranker; return top-n.

    Fails loud (raises ``RuntimeError``) when the reranker success rate drops below
    ``fallback_threshold`` — silent fallback is forbidden because cluster-path upstream
    scores are all 0.0 and would degrade to dict-insertion order.

    Raises:
        RuntimeError: When batch success rate falls below ``fallback_threshold``.
    """
    if not results:
        return []

    doc_texts, original_indices = _format_docs_for_rerank(results)
    if not doc_texts:
        return []

    all_scored = await _execute_rerank_batches(
        query,
        doc_texts,
        rerank_client=rerank_client,
        reranker_instruction=reranker_instruction,
        batch_size=batch_size,
        concurrent_batches=concurrent_batches,
        max_retries=max_retries,
        timeout=timeout,
        fallback_threshold=fallback_threshold,
    )

    all_scored.sort(key=lambda x: x[1], reverse=True)
    top_scored = all_scored[:top_n]
    return [(results[original_indices[idx]][0], score) for idx, score in top_scored]


# ---------------------------------------------------------------------------
# Algo bridge — RerankFn adapter for aagentic_retrieve
# ---------------------------------------------------------------------------


def _build_rerank_fn(
    rerank_client: RerankClient,
    *,
    reranker_instruction: str | None,
    batch_size: int,
    concurrent_batches: int,
    max_retries: int,
    timeout: float,
    fallback_threshold: float,
) -> RerankFn:
    """Wrap ``RerankClient`` as an algo ``RerankFn`` compatible with ``aagentic_retrieve``.

    The returned callable converts ``list[Candidate]`` to ``list[_Scored]``,
    calls ``reranker_search`` with ``top_n=len(candidates)`` (no truncation;
    caller slices the result), and converts back to ``list[Candidate]``.
    """

    async def rerank_fn(query: str, candidates: list[_Candidate]) -> list[_Candidate]:
        scored_inputs = _candidates_to_scored(candidates)
        scored_outputs = await reranker_search(
            query,
            results=scored_inputs,
            rerank_client=rerank_client,
            top_n=len(scored_inputs),  # no truncation — caller slices
            reranker_instruction=reranker_instruction,
            batch_size=batch_size,
            concurrent_batches=concurrent_batches,
            max_retries=max_retries,
            timeout=timeout,
            fallback_threshold=fallback_threshold,
        )
        return _scored_to_candidates(scored_outputs)

    return rerank_fn


def _reranker_kwargs(config: Any) -> dict[str, Any]:
    """Collect reranker call kwargs from config — avoids repeating 7 args twice."""
    return {
        "reranker_instruction": config.reranker_instruction,
        "batch_size": config.reranker_batch_size,
        "concurrent_batches": config.reranker_concurrent_batches,
        "max_retries": config.reranker_max_retries,
        "timeout": config.reranker_timeout,
        "fallback_threshold": config.reranker_fallback_threshold,
    }


# ---------------------------------------------------------------------------
# amaxsim_retrieve helpers — child/parent closure factories
# ---------------------------------------------------------------------------


def _make_bm25_amaxsim_pair(
    bm25_index: dict[str, Any],
) -> tuple[Any, Any]:
    """Return ``(child_retrieve, parent_fetch)`` closures for BM25 MaxSim via ``amaxsim_retrieve``.

    ``child_retrieve`` emits one Candidate per fact-row (mirrors ``search_with_bm25_index``
    tokenisation verbatim). ``parent_fetch`` batch-hydrates parent docs by doc-index string.
    Max-pool + top-n are handled inside ``amaxsim_retrieve`` — result is bit-for-bit identical
    to the inline loop in ``search_with_bm25_index`` because:
      - max-pool uses ``float("-inf")`` sentinel + strict ``>`` (same as the old loop)
      - sort is ``sorted(reverse=True)[:top_n]`` (same as old ``sorted_items[:top_n]``)
    """
    bm25 = bm25_index["bm25"]
    docs: list[_Doc] = bm25_index["docs"]
    fact_to_doc_idx: list[int] = bm25_index["fact_to_doc_idx"]

    async def child_retrieve(q: str, _k: int) -> list[_Candidate]:
        _ensure_nltk()
        stemmer = PorterStemmer()
        stop_words: set[str] = set(stopwords.words("english"))  # type: ignore[no-untyped-call]
        tokenized_query = _tokenize(q, stemmer, stop_words)
        if not tokenized_query:
            return []
        fact_scores: Any = bm25.get_scores(tokenized_query)
        return [
            _Candidate(
                id=f"f{i}",
                score=float(s),
                source="keyword",
                metadata={"parent_id": str(fact_to_doc_idx[i])},
            )
            for i, s in enumerate(fact_scores)
        ]

    async def parent_fetch(parent_ids: list[str]) -> list[_Candidate]:
        return [
            _Candidate(
                id=d_idx,
                score=0.0,
                source="keyword",
                metadata={"doc": docs[int(d_idx)]},
            )
            for d_idx in parent_ids
        ]

    return child_retrieve, parent_fetch


def _emb_doc_to_children(
    doc_idx: int,
    item: dict[str, Any],
    query_vec: np.ndarray,
    query_norm: float,
) -> list[_Candidate]:
    """Score one embedding index entry and return its child Candidates for max-pool.

    Entity-split model: uses atomic_facts + subject only (no summary/episode content fallback).
    Returns an empty list when no embeddings exist, or a list of scored Candidates.
    """
    embeddings: dict[str, Any] = item.get("embeddings") or {}
    if not embeddings:
        return []

    pool: list[Any] = []
    atomic_fact_embs = embeddings.get("atomic_facts")
    if isinstance(atomic_fact_embs, list) and atomic_fact_embs:
        pool.extend(atomic_fact_embs)

    subject_emb = embeddings.get("subject")
    if subject_emb is not None:
        pool.append(subject_emb)

    if not pool:
        return []

    fact_matrix = np.array(pool)
    fact_norms = np.linalg.norm(fact_matrix, axis=1)
    valid_mask = fact_norms > 0
    if not np.any(valid_mask):
        return [_Candidate(id=f"d{doc_idx}_invalid", score=0.0, source="vector", metadata={"parent_id": str(doc_idx)})]

    dot_products = np.dot(fact_matrix[valid_mask], query_vec)
    sims = dot_products / (query_norm * fact_norms[valid_mask])
    return [
        _Candidate(id=f"d{doc_idx}f{i}", score=float(s), source="vector", metadata={"parent_id": str(doc_idx)})
        for i, s in enumerate(sims)
    ]


def _make_emb_amaxsim_pair(
    emb_index: list[dict[str, Any]],
    embedding_client: Any,
    episode_docs: list[_Doc] | None = None,
) -> tuple[Any, Any]:
    """Return ``(child_retrieve, parent_fetch)`` closures for embedding MaxSim via ``amaxsim_retrieve``.

    ``child_retrieve`` mirrors ``search_with_emb_index`` corner cases A (query_norm==0 short-circuit)
    and delegates per-doc scoring to ``_emb_doc_to_children``. Max-pool + top-n happen
    inside ``amaxsim_retrieve``.

    Args:
        emb_index: Embedding index entries (``doc_id`` + ``embeddings``).
        embedding_client: Client for embedding the query.
        episode_docs: Optional list of episode dicts aligned by index with ``emb_index``.
            When provided, ``parent_fetch`` returns the full episode dict (needed by the reranker).
            When ``None``, returns a minimal ``{"id": doc_id}`` dict.
    """

    async def child_retrieve(q: str, _k: int) -> list[_Candidate]:
        query_vec = await _resolve_query_vec(q, embedding_client, None)
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm == 0:
            return []
        children: list[_Candidate] = []
        for doc_idx, item in enumerate(emb_index):
            children.extend(_emb_doc_to_children(doc_idx, item, query_vec, query_norm))
        return children

    async def parent_fetch(parent_ids: list[str]) -> list[_Candidate]:
        results: list[_Candidate] = []
        for d_idx in parent_ids:
            idx = int(d_idx)
            if episode_docs is not None and idx < len(episode_docs):
                doc = episode_docs[idx]
            else:
                doc = {"id": emb_index[idx].get("doc_id", d_idx)}
            results.append(_Candidate(id=d_idx, score=0.0, source="vector", metadata={"doc": doc}))
        return results

    return child_retrieve, parent_fetch


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


async def run_search_stage(ctx: StageContext) -> StageStats:
    """Stage 5 — agentic retrieval for every (conv, question).

    Loads per-conversation BM25 + embedding indices written by Stage 4
    (``index.py``), runs agentic retrieval for every QA pair, and writes
    ``search_results.json`` to ``ctx.output_dir``.

    Args:
        ctx: ``StageContext`` instance (typed as ``Any`` to avoid a circular
            import; the actual type is ``benchmarks.common.stages.types.StageContext``).

    Returns:
        ``StageStats`` with ``stage_name="search"`` and success / failed counts.
    """
    from benchmarks.common.stages.types import StageStats

    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    convs = _load_conversations_for_search(ctx)
    filter_cats = ctx.dataset.filter_categories()

    qa_sem = asyncio.Semaphore(ctx.config.max_concurrent_qa)

    started = time.monotonic()
    search_results: dict[str, list[dict[str, Any]]] = {}
    success_total = 0

    for conv_num, conv in enumerate(convs, start=1):
        conv_results = await _search_one_conversation(
            conv, ctx, filter_cats, qa_sem=qa_sem, conv_num=conv_num, total=len(convs)
        )
        if conv_results is None:
            continue
        search_results[conv.id] = conv_results
        success_total += len(conv_results)

    _write_search_results(search_results, ctx.output_dir)
    prompt_tokens, completion_tokens = _aggregate_retrieval_tokens(search_results)

    return StageStats(
        stage_name="search",
        duration_seconds=time.monotonic() - started,
        success=success_total,
        failed=0,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _load_conversations_for_search(ctx: StageContext) -> list[Any]:
    """Load conversations from the dataset, applying smoke-mode truncation.

    Sorts by numeric suffix so ``locomo_exp_user_10`` lands after ``..._9``
    instead of after ``..._1`` (alphabetical). LoCoMo only has 10 convs so the
    order coincides with the raw JSON yield order, but the explicit sort keeps the
    contract dataset-agnostic.
    """
    convs = sorted(ctx.dataset.load_conversations(), key=_conv_sort_key)
    if ctx.conv_indices is not None:
        allowed = set(ctx.conv_indices)
        convs = [c for i, c in enumerate(convs) if i in allowed]
    elif ctx.smoke:
        convs = convs[: ctx.smoke_conv_limit]
    return convs


def _conv_sort_key(conv: Any) -> tuple[str, int]:
    """Numeric-suffix sort key for conv_id."""
    parts = conv.id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (parts[0], int(parts[1]))
    return (conv.id, 0)


async def _search_one_conversation(
    conv: Any,
    ctx: StageContext,
    filter_cats: set[str],
    *,
    qa_sem: asyncio.Semaphore,
    conv_num: int,
    total: int,
) -> list[dict[str, Any]] | None:
    """Run agentic retrieval for every QA pair of a single conversation.

    Returns the list of successful retrieval result dicts on success, or
    ``None`` when the per-conv BM25 / embedding indices are missing (soft skip).
    Retrieval failures propagate (caller fast-fails the whole stage per
    Stage 1-4 artifact contract); ``None`` entries in the gathered list represent
    filtered-out categories (e.g. adversarial), not failures.
    """
    conv_idx = _conv_index(conv.id)
    indices = _load_per_conv_indices(ctx.input_dir, conv_idx, ctx.config)
    if indices is None:
        logger.warning("Skipping %s: index files not found in %s", conv.id, ctx.input_dir)
        return None

    qa_pairs = _load_qa_pairs_for_conv(ctx, conv.id)
    raw_results = await _gather_qa_retrieval(
        qa_pairs, indices, ctx, filter_cats, qa_sem=qa_sem, conv_num=conv_num, total=total
    )
    return [item for item in raw_results if item is not None]


def _load_per_conv_indices(input_dir: Any, conv_idx: int, config: Any) -> dict[str, Any] | None:
    """Load BM25 + embedding + cluster indices for one conversation.

    Returns ``{"bm25": ..., "emb": ..., "cluster": list[Cluster]}`` or ``None``
    when the BM25 or embedding pkl is missing (soft-skip path). Cluster loading
    always runs in agentic mode; a missing cluster pkl raises ``FileNotFoundError``.
    """
    bm25_path = input_dir / f"bm25_conv_{conv_idx}.pkl"
    emb_path = input_dir / f"emb_conv_{conv_idx}.pkl"

    if not bm25_path.exists() or not emb_path.exists():
        return None

    with bm25_path.open("rb") as fh:
        bm25_index: dict[str, Any] = pickle.load(fh)
    with emb_path.open("rb") as fh:
        emb_index: list[dict[str, Any]] = pickle.load(fh)

    cluster_index = _load_cluster_index(input_dir, conv_idx)
    return {"bm25": bm25_index, "emb": emb_index, "cluster": cluster_index}


def _load_qa_pairs_for_conv(ctx: StageContext, conv_id: str) -> list[Any]:
    """Load QA pairs for one conversation, applying smoke-mode truncation."""
    qa_pairs = list(ctx.dataset.load_qa_pairs(conv_id))
    if ctx.smoke:
        qa_pairs = qa_pairs[: ctx.smoke_qa_limit]
    return qa_pairs


async def _gather_qa_retrieval(
    qa_pairs: list[Any],
    indices: dict[str, Any],
    ctx: StageContext,
    filter_cats: set[str],
    *,
    qa_sem: asyncio.Semaphore,
    conv_num: int,
    total: int,
) -> list[dict[str, Any] | None]:
    """Run ``_process_single_qa`` concurrently for every QA pair under the shared semaphore."""
    from benchmarks.common._progress import gather_with_progress

    return await gather_with_progress(
        *(
            _process_single_qa(
                qa,
                ctx=ctx,
                sem=qa_sem,
                emb_index=indices["emb"],
                bm25_index=indices["bm25"],
                cluster_index=indices["cluster"],
                filter_cats=filter_cats,
            )
            for qa in qa_pairs
        ),
        desc=f"search {conv_num}/{total}",
        unit="q",
    )


def _write_search_results(search_results: dict[str, list[dict[str, Any]]], output_dir: Any) -> None:
    """Dump aggregated per-conv search results to ``search_results.json``."""
    output_path = output_dir / "search_results.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(search_results, fh, indent=2, ensure_ascii=False)


def _aggregate_retrieval_tokens(
    search_results: dict[str, list[dict[str, Any]]],
) -> tuple[int, int]:
    """Sum prompt / completion tokens across every QA's ``retrieval_metadata``."""
    total_prompt = 0
    total_completion = 0
    for conv_results in search_results.values():
        for item in conv_results:
            meta: dict[str, Any] = item.get("retrieval_metadata", {})
            total_prompt += meta.get("prompt_tokens", 0)
            total_completion += meta.get("completion_tokens", 0)
    return total_prompt, total_completion


def _write_search_error(ctx: StageContext, question_id: str, max_retries: int) -> None:
    """Write error traceback to ``search_<question_id>.error.txt`` and log."""
    err_path = ctx.output_dir / f"search_{question_id}.error.txt"
    err_path.write_text(traceback.format_exc())
    logger.exception(
        "question_id=%s retrieval failed after %d attempts; full traceback in %s",
        question_id,
        max_retries,
        err_path,
    )


async def _process_single_qa(
    qa: Any,
    *,
    ctx: StageContext,
    sem: asyncio.Semaphore,
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
    cluster_index: list[Cluster],
    filter_cats: set[str],
) -> dict[str, Any] | None:
    """Run agentic retrieval for one QA pair with retry + fail-loud.

    Returns ``None`` only for filtered-out categories.
    """
    if qa.category in filter_cats:
        return None

    max_retries = max(1, int(ctx.config.llm_max_retries))

    for attempt in range(max_retries):
        try:
            return await asyncio.wait_for(
                _attempt_single_qa(
                    qa,
                    ctx=ctx,
                    sem=sem,
                    emb_index=emb_index,
                    bm25_index=bm25_index,
                    cluster_index=cluster_index,
                ),
                timeout=120.0,
            )
        except Exception:
            if attempt >= max_retries - 1:
                _write_search_error(ctx, qa.question_id, max_retries)
                raise
            logger.warning(
                "Retrieval attempt %d/%d failed for question_id=%s; retrying",
                attempt + 1,
                max_retries,
                qa.question_id,
                exc_info=True,
            )
            await asyncio.sleep(1.0 * (2**attempt))

    raise RuntimeError(f"_process_single_qa: exhausted retry loop without return (qa={qa.question_id})")


def _build_retrieval_closures(
    cfg: Any,
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
    cluster_index: list[Cluster],
    embedding_client: EmbeddingClient,
    rerank_fn: RerankFn,
) -> tuple[RetrieveFn, RetrieveFn, int, RerankFn]:
    """Build the retrieval closure stack (dense → sparse → hybrid → cluster).

    Returns:
        Tuple of ``(base_retrieve, round2_retrieve, round2_cap, rerank_fn)``.
    """
    episode_docs: list[_Doc] = bm25_index["docs"]

    async def _dense(q: str, k: int) -> list[_Candidate]:
        child_retrieve, parent_fetch = _make_emb_amaxsim_pair(emb_index, embedding_client, episode_docs=episode_docs)
        results = await amaxsim_retrieve(
            q,
            child_retrieve=child_retrieve,
            parent_fetch=parent_fetch,
            top_n=k,
            child_candidates=len(emb_index),
        )
        return [_doc_to_candidate(c.metadata["doc"], c.score) for c in results]

    async def _sparse(q: str, k: int) -> list[_Candidate]:
        child_retrieve, parent_fetch = _make_bm25_amaxsim_pair(bm25_index)
        results = await amaxsim_retrieve(
            q,
            child_retrieve=child_retrieve,
            parent_fetch=parent_fetch,
            top_n=k,
            child_candidates=len(bm25_index["fact_to_doc_idx"]),
        )
        return [_doc_to_candidate(c.metadata["doc"], c.score) for c in results]

    async def hybrid_full(q: str, k: int) -> list[_Candidate]:
        return await ahybrid_retrieve(
            q,
            dense_retrieve=_dense,
            sparse_retrieve=_sparse,
            top_n=k,
            dense_candidates=cfg.hybrid_emb_candidates,
            sparse_candidates=cfg.hybrid_bm25_candidates,
            rrf_k=cfg.hybrid_rrf_k,
        )

    return _build_cluster_closures(cfg, bm25_index, cluster_index, hybrid_full, rerank_fn)


def _build_cluster_closures(
    cfg: Any,
    bm25_index: dict[str, Any],
    cluster_index: list[Cluster],
    hybrid_full: RetrieveFn,
    rerank_fn: RerankFn,
) -> tuple[RetrieveFn, RetrieveFn, int, RerankFn]:
    """Build the cluster-path closures when ``cluster_index`` is available."""
    all_docs: list[_Candidate] = [_doc_to_candidate(d, 0.0) for d in bm25_index["docs"]]

    async def cluster_scoped(q: str, _k: int) -> list[_Candidate]:
        return await acluster_retrieve(
            q,
            base_retrieve=hybrid_full,
            base_candidates=None,
            clusters=cluster_index,
            all_docs=all_docs,
            cluster_top_k=cfg.cluster_top_k,
        )

    return cluster_scoped, hybrid_full, 40, rerank_fn


async def _run_agentic_retrieval(
    query: str,
    *,
    cfg: Any,
    base_retrieve: RetrieveFn,
    round2_retrieve: RetrieveFn,
    round2_cap: int,
    rerank_fn: RerankFn,
    llm_proxy: _TokenCountingLLM,
    sem: asyncio.Semaphore,
) -> tuple[list[_Candidate], Any]:
    """Execute ``aagentic_retrieve`` under the concurrency semaphore.

    Returns:
        Tuple of ``(final_candidates, decision)``.
    """
    async with sem:
        return await aagentic_retrieve(
            query,
            base_retrieve=base_retrieve,
            round2_retrieve=round2_retrieve,
            round2_cap=round2_cap,
            rerank_fn=rerank_fn,
            llm=llm_proxy,  # type: ignore[arg-type]
            top_n=cfg.response_top_k,
            round1_top_n=50,
            round1_rerank_top_n=cfg.round1_rerank_top_n,
            refinement_strategy="multi_query",
            multi_query_count=cfg.multi_query_num,
            rrf_k=cfg.hybrid_rrf_k,
        )


def _build_retrieval_result(
    qa: Any,
    top_results: list[_Scored],
    decision: Any,
    llm_proxy: _TokenCountingLLM,
) -> dict[str, Any]:
    """Assemble the final search result dict for one QA pair."""
    members = [doc["id"] for doc, _ in top_results if doc.get("id") is not None]
    retrieval_metadata: dict[str, Any] = {
        "is_multi_round": decision.is_multi_round,
        "is_sufficient": decision.is_sufficient,
        "reasoning": decision.reasoning,
        "missing_info": list(decision.missing_info),
        "key_information_found": list(decision.key_information_found),
        "refined_queries": list(decision.refined_queries),
        "query_strategy": decision.query_strategy,
        "final_count": len(top_results),
        "prompt_tokens": llm_proxy.prompt_tokens,
        "completion_tokens": llm_proxy.completion_tokens,
        "trace": {"final_top": _trace_scored(top_results, limit=20)},
    }
    return {
        "question_id": qa.question_id,
        "query": qa.question,
        "members": members,
        "original_qa": {
            "question_id": qa.question_id,
            "conv_id": qa.conv_id,
            "question": qa.question,
            "golden_answer": qa.golden_answer,
            "category": qa.category,
        },
        "retrieval_metadata": retrieval_metadata,
    }


async def _attempt_single_qa(
    qa: Any,
    *,
    ctx: StageContext,
    sem: asyncio.Semaphore,
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
    cluster_index: list[Cluster],
) -> dict[str, Any]:
    """Single retrieval attempt — caller wraps with retry + error.txt + re-raise."""
    cfg = ctx.config
    llm_proxy = _TokenCountingLLM(ctx.services.llm)
    rerank_fn = _build_rerank_fn(ctx.services.rerank, **_reranker_kwargs(cfg))

    base_retrieve, round2_retrieve, round2_cap, rerank_fn = _build_retrieval_closures(
        cfg, emb_index, bm25_index, cluster_index, ctx.services.embedding, rerank_fn
    )

    final_candidates, decision = await _run_agentic_retrieval(
        qa.question,
        cfg=cfg,
        base_retrieve=base_retrieve,
        round2_retrieve=round2_retrieve,
        round2_cap=round2_cap,
        rerank_fn=rerank_fn,
        llm_proxy=llm_proxy,
        sem=sem,
    )

    top_results = _candidates_to_scored(final_candidates)
    return _build_retrieval_result(qa, top_results, decision, llm_proxy)


def _load_cluster_index(input_dir: Any, conv_idx: int) -> list[Cluster]:
    """Load cluster snapshot as ``list[Cluster]``.

    Stage 4 writes the pkl as ``list[Cluster.model_dump()]`` (one entry per cluster);
    we round-trip via ``Cluster.model_validate`` so downstream gets a typed view.

    Always loads in agentic mode — the cluster index MUST exist. Stage 1 builds the
    source clusters and Stage 4 builds this index, so an absent pkl indicates an incomplete run.
    """
    cluster_path = input_dir / f"cluster_index_conv_{conv_idx}.pkl"
    if not cluster_path.exists():
        raise FileNotFoundError(
            f"Cluster index not found for conv_{conv_idx}; expected: {cluster_path}. Re-run stage 4."
        )
    with cluster_path.open("rb") as fh:
        raw = pickle.load(fh)
    return [Cluster.model_validate(d) for d in raw]


def _conv_index(conv_id: str) -> int:
    """Extract the numeric suffix from a ``locomo_exp_user_<N>`` conv_id.

    Raises ``ValueError`` on a malformed ``conv_id`` — fail-loud rather than
    silently defaulting to 0 (which would route a broken conv to conv_0's
    indices and corrupt Stage 5 metrics).
    """
    prefix = "locomo_exp_user_"
    try:
        return int(conv_id.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(f"Cannot parse conv index from conv_id={conv_id!r}; expected '{prefix}<N>'") from exc
