"""Stage 3 — Retrieval primitives + agentic loop + main entry.

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
produced by Stage 2 in the same trusted local workspace. **Never point the
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
from typing import TYPE_CHECKING, Any, Literal, cast

import nltk  # type: ignore[import-untyped]
import numpy as np
from nltk.corpus import stopwords  # type: ignore[import-untyped]
from nltk.stem import PorterStemmer  # type: ignore[import-untyped]
from nltk.tokenize import word_tokenize  # type: ignore[import-untyped]

from everalgo.clustering import Cluster
from everalgo.rank import aagentic_retrieve, acluster_retrieve, ahybrid_retrieve, amaxsim_retrieve
from everalgo.types import Candidate as _Candidate

if TYPE_CHECKING:
    from benchmarks.common.services import EmbeddingClient, RerankClient
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
    ``id`` is read from ``doc["id"]``; absent ids fall back to an empty string (caller
    dedup-by-id degrades gracefully for such items).
    """
    doc_id = str(doc.get("id", "")) if doc.get("id") is not None else ""
    return _Candidate(id=doc_id, score=float(score), metadata={"_doc": doc, **doc})


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


# ---------------------------------------------------------------------------
# NLTK helpers
# ---------------------------------------------------------------------------


def _ensure_nltk() -> None:
    """Download required NLTK data if not present."""
    for find_path, download_id in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]:
        try:
            nltk.data.find(find_path)  # type: ignore[no-untyped-call]
        except LookupError:
            nltk.download(download_id, quiet=True)  # type: ignore[no-untyped-call]


def _tokenize(text: str, stemmer: Any, stop_words: set[str]) -> list[str]:
    """Lower -> tokenize -> keep alpha words len>=2 not stopword -> stem.

    Must be identical to the tokenization used during indexing (index.py).
    """
    if not text:
        return []
    tokens: list[str] = word_tokenize(text.lower())  # type: ignore[no-untyped-call]
    return [str(stemmer.stem(t)) for t in tokens if t.isalpha() and len(t) >= 2 and t not in stop_words]


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
        bm25_index: Fact-level BM25 payload as persisted by stage 2.
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
    """Score an embedding index entry via MaxSim — atomic_facts short-circuit + fallback.

    When the doc carries non-empty ``atomic_facts`` embeddings, the score is the max cosine
    similarity over just those facts; the ``subject`` / ``summary`` / ``episode`` field embeddings
    only participate as a fallback when ``atomic_facts`` is missing or empty. Atomic facts are
    LLM-condensed semantic summaries derived from the episode body, so they typically dominate the
    cosine similarity against the query — including the parent-field embeddings would dilute the
    MaxSim pool with redundant signal.
    """
    doc: _Doc = item.get("doc") or {}
    embeddings: dict[str, Any] = item.get("embeddings") or {}
    if not embeddings:
        return None

    atomic_fact_embs = embeddings.get("atomic_facts")
    if isinstance(atomic_fact_embs, list) and atomic_fact_embs:
        return (doc, compute_maxsim_score(query_vec, cast("list[np.ndarray]", atomic_fact_embs)))

    field_vecs: list[np.ndarray] = []
    for field in ("subject", "summary", "episode"):
        field_emb = embeddings.get(field)
        if field_emb is not None:
            field_vecs.append(field_emb)
    if not field_vecs:
        return None
    return (doc, compute_maxsim_score(query_vec, field_vecs))


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

    For documents that contain ``atomic_facts`` embeddings:
    - Compute cosine similarity between the query and each atomic_fact.
    - Take the maximum similarity as the document score (MaxSim strategy).

    For documents without ``atomic_facts``:
    - Fall back to scoring against ``subject``, ``summary``, and ``episode``
      field embeddings; take the maximum across those fields.

    Args:
        query: Raw query text.
        emb_index: Pre-built embedding index; each entry is
            ``{"doc": dict, "embeddings": {"atomic_facts": [...], ...}}``.
        top_n: Maximum number of results to return.
        embedding_client: Service used to embed the query when
            ``query_embedding`` is not pre-provided.
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


def _episode_field(doc: dict[str, Any], field: str) -> str:
    """Read ``doc.episode.<field>`` as a non-empty string, else empty.

    Tolerates the legacy schema where ``doc.episode`` was a plain string —
    such docs do not expose nested fields, so always return empty.
    """
    episode = doc.get("episode")
    if not isinstance(episode, dict):
        return ""
    value = cast("dict[str, Any]", episode).get(field) or ""
    return value if isinstance(value, str) else ""


def _format_doc_for_rerank(doc: dict[str, Any]) -> str:
    """Format a doc for reranker input — ``episode.content`` only, no fallback.

    Mirrors the 93-version behavior (``stage3_memory_retrivel.py:127-130``) which
    reads ``doc["episode"]`` as a single string. In our nested schema the equivalent
    is ``doc["episode"]["content"]``.

    Raises:
        ValueError: If ``episode.content`` is missing or empty — fail-loud so Stage 1
            schema regressions surface immediately.
    """
    content = _episode_field(doc, "content")
    if content:
        return content
    raise ValueError(f"doc has no episode.content for reranker: id={doc.get('id', 'unknown')}")


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


async def _rerank_batch_with_retry(
    query: str,
    start: int,
    batch_texts: list[str],
    *,
    rerank_client: RerankClient,
    reranker_instruction: str | None,
    max_retries: int,
    retry_delay: float,
    timeout: float,
) -> list[tuple[int, float]] | None:
    """Attempt a single reranker batch up to ``max_retries`` times.

    Args:
        query: Raw query text forwarded to the reranker.
        start: Offset of this batch in the full ``docs_with_text`` list.
        batch_texts: Document texts for this batch.
        rerank_client: RerankClient instance.
        reranker_instruction: Optional task instruction.
        max_retries: Maximum attempts before giving up.
        retry_delay: Base delay for exponential backoff (seconds).
        timeout: Per-attempt asyncio timeout (seconds).

    Returns:
        List of ``(global_index, score)`` on success, or ``None`` after all
        retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            scored: list[tuple[int, float]] = await asyncio.wait_for(
                rerank_client.rerank(query, batch_texts, instruction=reranker_instruction),
                timeout=timeout,
            )
            return [(start + idx, score) for idx, score in scored]
        except Exception:  # intentional broad catch — retry on any failure
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2**attempt))
    return None


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
    retry_delay: float = 0.8,
    timeout: float = 60.0,
    fallback_threshold: float = 0.3,
) -> list[_Scored]:
    """Rerank candidate docs with Qwen3 reranker; return top-n.

    Documents containing atomic facts are formatted as multi-line text
    (timestamp + one fact per line) to match the upstream reference scoring format.
    Documents without usable text are silently skipped.

    Batches are processed with controlled concurrency and per-batch retry
    with exponential backoff. Fails loud (raises ``RuntimeError``) when the
    reranker success rate drops below ``fallback_threshold`` — silent fallback
    to the original ranking is forbidden because on the cluster path the upstream
    scores are all 0.0 (see ``acluster_retrieve``), so a fallback would degrade
    to dict-insertion order and silently regress accuracy. The outer
    ``_attempt_single_qa`` retry loop catches this; 5 retries exhausted means
    a real problem and the stage aborts (global ``唯一兜底 = 重试`` principle).

    Args:
        query: Raw query text.
        results: Candidate (doc, score) pairs from prior retrieval.
        rerank_client: RerankClient instance (DeepInfra Qwen3-Reranker).
        top_n: Number of results to return after reranking.
        reranker_instruction: Optional task instruction for the reranker.
        batch_size: Documents per API call.
        concurrent_batches: Max batches processed simultaneously per group.
        max_retries: Per-batch retry limit.
        retry_delay: Base delay for exponential backoff (seconds).
        timeout: Per-batch asyncio timeout (seconds).
        fallback_threshold: Minimum batch success rate; below this the call raises.

    Returns:
        List of ``(doc, reranker_score)`` pairs sorted by score descending.

    Raises:
        RuntimeError: When batch success rate falls below ``fallback_threshold``.
    """
    if not results:
        return []

    # Step 1: Format docs, preserving original index for round-trip mapping.
    docs_with_text: list[tuple[int, _Doc, str]] = []
    for orig_idx, (doc, _score) in enumerate(results):
        text = _format_doc_for_rerank(doc)
        if text:
            docs_with_text.append((orig_idx, doc, text))

    if not docs_with_text:
        return []

    doc_texts = [t for _, _, t in docs_with_text]
    original_indices = [oi for oi, _, _ in docs_with_text]

    # Step 2: Partition into fixed-size batches.
    batches: list[tuple[int, list[str]]] = [
        (i, doc_texts[i : i + batch_size]) for i in range(0, len(doc_texts), batch_size)
    ]

    # Step 3: Run batches in groups of concurrent_batches with inter-group delay.
    all_scored: list[tuple[int, float]] = []
    successful = 0

    for group_start in range(0, len(batches), concurrent_batches):
        group = batches[group_start : group_start + concurrent_batches]
        group_results = await asyncio.gather(
            *(
                _rerank_batch_with_retry(
                    query,
                    start,
                    batch,
                    rerank_client=rerank_client,
                    reranker_instruction=reranker_instruction,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    timeout=timeout,
                )
                for start, batch in group
            )
        )
        for r in group_results:
            if r is not None:
                all_scored.extend(r)
                successful += 1
        if group_start + concurrent_batches < len(batches):
            await asyncio.sleep(0.3)

    success_rate = successful / len(batches) if batches else 0.0

    # Fail-loud when too many reranker batches failed. Silent fallback to the original
    # ranking would degenerate to dict-insertion order on the cluster path (upstream
    # scores are 0.0) and mask real reranker outages.
    if not all_scored or success_rate < fallback_threshold:
        raise RuntimeError(
            f"reranker batch success rate {success_rate:.2f} below threshold {fallback_threshold:.2f} "
            f"({successful}/{len(batches)} batches succeeded)"
        )

    # Step 4: Sort by reranker score and map back to original documents.
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
    retry_delay: float,
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
            retry_delay=retry_delay,
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
        "retry_delay": config.reranker_retry_delay,
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

    Mirrors ``_score_emb_item`` + ``compute_maxsim_score`` corner cases (B-H) exactly.
    Returns an empty list to skip the doc (B/E), or a list of scored Candidates (H)
    — including a score=0.0 placeholder when valid_mask is all-false (G).
    """
    embeddings: dict[str, Any] = item.get("embeddings") or {}
    if not embeddings:
        return []  # B

    # C: atomic_facts non-empty → use exclusively (mirrors _score_emb_item short-circuit)
    atomic_fact_embs = embeddings.get("atomic_facts")
    if isinstance(atomic_fact_embs, list) and atomic_fact_embs:
        fact_embs: list[Any] = atomic_fact_embs
    else:
        # D: fallback chain — fixed field order mirrors _score_emb_item
        field_vecs: list[Any] = [
            embeddings[f] for f in ("episode", "subject", "summary") if embeddings.get(f) is not None
        ]
        if not field_vecs:
            return []  # E
        fact_embs = field_vecs

    # H: vectorised MaxSim — mirrors compute_maxsim_score matrix path exactly
    fact_matrix = np.array(fact_embs)
    fact_norms = np.linalg.norm(fact_matrix, axis=1)
    valid_mask = fact_norms > 0
    if not np.any(valid_mask):
        # G: score=0.0 placeholder ensures parent still enters max-pool (mirrors return 0.0)
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
) -> tuple[Any, Any]:
    """Return ``(child_retrieve, parent_fetch)`` closures for embedding MaxSim via ``amaxsim_retrieve``.

    ``child_retrieve`` mirrors ``search_with_emb_index`` corner cases A (query_norm==0 short-circuit)
    and delegates per-doc scoring to ``_emb_doc_to_children`` (cases B-H). Max-pool + top-n happen
    inside ``amaxsim_retrieve`` using the same sentinel/sort as the old inline loop.
    """

    async def child_retrieve(q: str, _k: int) -> list[_Candidate]:
        query_vec = await _resolve_query_vec(q, embedding_client, None)
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm == 0:
            return []  # A
        children: list[_Candidate] = []
        for doc_idx, item in enumerate(emb_index):
            children.extend(_emb_doc_to_children(doc_idx, item, query_vec, query_norm))
        return children

    async def parent_fetch(parent_ids: list[str]) -> list[_Candidate]:
        return [
            _Candidate(
                id=d_idx,
                score=0.0,
                source="vector",
                metadata={"doc": emb_index[int(d_idx)].get("doc") or {}},
            )
            for d_idx in parent_ids
        ]

    return child_retrieve, parent_fetch


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


async def run_search_stage(ctx: Any) -> Any:
    """Stage 3 — agentic retrieval for every (conv, question).

    Loads per-conversation BM25 + embedding indices written by Stage 2
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


def _load_conversations_for_search(ctx: Any) -> list[Any]:
    """Load conversations from the dataset, applying smoke-mode truncation.

    Sorts by numeric suffix so ``locomo_exp_user_10`` lands after ``..._9``
    instead of after ``..._1`` (alphabetical). LoCoMo only has 10 convs so the
    order coincides with the raw JSON yield order, but the explicit sort keeps the
    contract dataset-agnostic.
    """
    convs = sorted(ctx.dataset.load_conversations(), key=_conv_sort_key)
    if ctx.smoke:
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
    ctx: Any,
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
    Stage 1/2 contract); ``None`` entries in the gathered list represent
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

    Returns ``{"bm25": ..., "emb": ..., "cluster": list[Cluster] | None}`` or ``None``
    when the BM25 or embedding pkl is missing (soft-skip path).  Cluster loading
    delegates to ``_load_cluster_index`` and preserves its fast-fail contract:
    when ``enable_cluster_retrieval`` is True but the cluster pkl is missing,
    the exception propagates to the caller.
    """
    bm25_path = input_dir / f"bm25_conv_{conv_idx}.pkl"
    emb_path = input_dir / f"emb_conv_{conv_idx}.pkl"

    if not bm25_path.exists() or not emb_path.exists():
        return None

    with bm25_path.open("rb") as fh:
        bm25_index: dict[str, Any] = pickle.load(fh)
    with emb_path.open("rb") as fh:
        emb_index: list[dict[str, Any]] = pickle.load(fh)

    cluster_index = _load_cluster_index(input_dir, conv_idx, enable=config.enable_cluster_retrieval)
    return {"bm25": bm25_index, "emb": emb_index, "cluster": cluster_index}


def _load_qa_pairs_for_conv(ctx: Any, conv_id: str) -> list[Any]:
    """Load QA pairs for one conversation, applying smoke-mode truncation."""
    qa_pairs = list(ctx.dataset.load_qa_pairs(conv_id))
    if ctx.smoke:
        qa_pairs = qa_pairs[: ctx.smoke_qa_limit]
    return qa_pairs


async def _gather_qa_retrieval(
    qa_pairs: list[Any],
    indices: dict[str, Any],
    ctx: Any,
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


async def _process_single_qa(
    qa: Any,
    *,
    ctx: Any,
    sem: asyncio.Semaphore,
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
    cluster_index: list[Cluster] | None,
    filter_cats: set[str],
) -> dict[str, Any] | None:
    """Run agentic retrieval for a single QA pair with retry + fail-loud.

    Returns ``None`` only for filtered-out categories. On retrieval failure the
    call is retried up to ``cfg.llm_max_retries`` times with exponential backoff;
    if all retries are exhausted, writes ``search_<question_id>.error.txt`` with
    the traceback and re-raises so ``asyncio.gather`` terminates the stage. This
    mirrors the Stage 2 fast-fail contract (``index.py:466-480``).
    """
    if qa.category in filter_cats:
        return None

    cfg = ctx.config
    max_retries = max(1, int(cfg.llm_max_retries))

    for attempt in range(max_retries):
        try:
            # 120 s per-attempt cap. LLM client / embedding / BM25 / reranker each carry their own
            # timeouts, but a stuck branch on any of them could otherwise hang the whole stage
            # waiting on ``asyncio.gather``. ``TimeoutError`` falls through to the retry handler below.
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
                err_path = ctx.output_dir / f"search_{qa.question_id}.error.txt"
                err_path.write_text(traceback.format_exc())
                logger.exception(
                    "question_id=%s retrieval failed after %d attempts; full traceback in %s",
                    qa.question_id,
                    max_retries,
                    err_path,
                )
                raise
            logger.warning(
                "Retrieval attempt %d/%d failed for question_id=%s; retrying",
                attempt + 1,
                max_retries,
                qa.question_id,
                exc_info=True,
            )
            await asyncio.sleep(1.0 * (2**attempt))

    # Unreachable: loop either returns or raises.
    raise RuntimeError(f"_process_single_qa: exhausted retry loop without return (qa={qa.question_id})")


async def _attempt_single_qa(
    qa: Any,
    *,
    ctx: Any,
    sem: asyncio.Semaphore,
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
    cluster_index: list[Cluster] | None,
) -> dict[str, Any]:
    """Single retrieval attempt — caller wraps with retry + error.txt + re-raise.

    Builds three caller-side closures wrapping the per-conv BM25 + embedding indices:
    ``_dense`` (vector search), ``_sparse`` (BM25 search), and ``hybrid_full`` (dense+sparse
    RRF via ``ahybrid_retrieve``). When cluster retrieval is enabled, also builds a
    ``cluster_scoped`` closure wrapping ``acluster_retrieve`` and passes it as
    ``aagentic_retrieve.base_retrieve`` with ``hybrid_full`` as ``round2_retrieve``
    (Round 2 spills out of cluster scope to the full corpus). Cap merged R1+R2 at 40.
    """
    cfg = ctx.config
    embedding_client: EmbeddingClient = ctx.services.embedding
    rerank_client: RerankClient = ctx.services.rerank
    llm_proxy = _TokenCountingLLM(ctx.services.llm)

    rerank_fn = _build_rerank_fn(
        rerank_client,
        **_reranker_kwargs(cfg),
    )

    async def _dense(q: str, k: int) -> list[_Candidate]:
        child_retrieve, parent_fetch = _make_emb_amaxsim_pair(emb_index, embedding_client)
        results = await amaxsim_retrieve(
            q,
            child_retrieve=child_retrieve,
            parent_fetch=parent_fetch,
            top_n=k,
            child_candidates=len(emb_index),  # exhaustive — mirror search_with_emb_index full scan
        )
        return [_doc_to_candidate(c.metadata["doc"], c.score) for c in results]

    async def _sparse(q: str, k: int) -> list[_Candidate]:
        child_retrieve, parent_fetch = _make_bm25_amaxsim_pair(bm25_index)
        results = await amaxsim_retrieve(
            q,
            child_retrieve=child_retrieve,
            parent_fetch=parent_fetch,
            top_n=k,
            child_candidates=len(bm25_index["fact_to_doc_idx"]),  # exhaustive — all fact rows
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

    base_retrieve: RetrieveFn
    round2_retrieve: RetrieveFn | None
    round2_cap: int | None
    agentic_rerank_fn: RerankFn | None
    if cluster_index is not None:
        all_docs: list[_Candidate] = [_doc_to_candidate(d, 0.0) for d in bm25_index["docs"]]

        async def cluster_scoped(q: str, _k: int) -> list[_Candidate]:
            # ``_k`` (aagentic's ``round1_top_n``) is ignored on the cluster path: the Level-1
            # candidate pool size is controlled by ``cluster_base_candidates``, and
            # ``acluster_retrieve`` returns the entire cluster expansion unreranked so aagentic
            # can rerank it.
            return await acluster_retrieve(
                q,
                base_retrieve=hybrid_full,
                base_candidates=cfg.cluster_base_candidates,
                clusters=cluster_index,
                all_docs=all_docs,
                cluster_top_k=cfg.cluster_top_k,
            )

        base_retrieve = cluster_scoped
        round2_retrieve = hybrid_full
        round2_cap = 40
        # aagentic reranks twice on the cluster path: once over the full Round-1 expansion (Level 2)
        # and once over the merged Round-1 + Round-2 set (Round-2 final rerank,
        # ``scene_retrieval.py:331-356``). Both passes share this ``rerank_fn``.
        agentic_rerank_fn = rerank_fn
    else:
        base_retrieve = hybrid_full
        round2_retrieve = None
        round2_cap = None
        # Hybrid path: rerank gated by config (mirrors old _hybrid_agentic_retrieval).
        agentic_rerank_fn = rerank_fn if cfg.use_reranker else None

    refinement_strategy: Literal["multi_query", "refined_query"] = (
        "multi_query" if getattr(cfg, "use_multi_query", True) else "refined_query"
    )

    async with sem:
        final_candidates, decision = await aagentic_retrieve(
            qa.question,
            base_retrieve=base_retrieve,
            round2_retrieve=round2_retrieve,
            round2_cap=round2_cap,
            rerank_fn=agentic_rerank_fn,
            llm=llm_proxy,  # type: ignore[arg-type]
            top_n=cfg.response_top_k,
            # 50 = Round-2 sub-query candidate pool (``hybrid_emb_candidates+hybrid_bm25_candidates``
            # from ``scene_retrieval.py:302-324``). On the cluster path the Round-1
            # ``cluster_scoped`` closure ignores this hint and uses ``cfg.cluster_base_candidates``
            # instead, so this value only governs Round-2 per-query retrieval and the hybrid
            # fallback path.
            round1_top_n=50,
            round1_rerank_top_n=cfg.round1_rerank_top_n,
            refinement_strategy=refinement_strategy,
            multi_query_count=cfg.multi_query_num,
            rrf_k=cfg.hybrid_rrf_k,
        )

    top_results = _candidates_to_scored(final_candidates)
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


def _load_cluster_index(input_dir: Any, conv_idx: int, *, enable: bool) -> list[Cluster] | None:
    """Load cluster snapshot as ``list[Cluster]`` when ``enable`` is True.

    Stage 2 writes the pkl as ``list[Cluster.model_dump()]`` (one entry per cluster);
    we round-trip via ``Cluster.model_validate`` so downstream gets a typed view.

    Returns ``None`` only when ``enable`` is False. When enabled, the file MUST exist —
    stage 2 fast-fails on cluster build, so an absent pkl indicates an out-of-band skip:
    re-run stage 2 or set ``enable_cluster_retrieval=False``.
    """
    if not enable:
        return None
    cluster_path = input_dir / f"cluster_index_conv_{conv_idx}.pkl"
    if not cluster_path.exists():
        raise FileNotFoundError(
            f"enable_cluster_retrieval=True but cluster index not found for conv_{conv_idx}; "
            f"expected: {cluster_path}. Re-run stage 2 or set enable_cluster_retrieval=False."
        )
    with cluster_path.open("rb") as fh:
        raw = pickle.load(fh)
    return [Cluster.model_validate(d) for d in raw]


def _conv_index(conv_id: str) -> int:
    """Extract the numeric suffix from a ``locomo_exp_user_<N>`` conv_id.

    Raises ``ValueError`` on a malformed ``conv_id`` — fail-loud rather than
    silently defaulting to 0 (which would route a broken conv to conv_0's
    indices and corrupt Stage 3 metrics).
    """
    prefix = "locomo_exp_user_"
    try:
        return int(conv_id.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(f"Cannot parse conv index from conv_id={conv_id!r}; expected '{prefix}<N>'") from exc
