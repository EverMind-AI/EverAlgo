"""Stage 3 — Retrieval primitives + agentic loop + main entry.

Ports the pure scoring/fusion functions from EverCore's stage3_memory_retrivel.py
verbatim to preserve benchmark parity. T14b adds hybrid_search_with_rrf +
reranker_search; T14c adds agentic_retrieval (LLM-guided multi-round loop) and
run_search_stage (main pipeline entry point).
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import time
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import nltk  # type: ignore[import-untyped]
import numpy as np
from nltk.corpus import stopwords  # type: ignore[import-untyped]
from nltk.stem import PorterStemmer  # type: ignore[import-untyped]
from nltk.tokenize import word_tokenize  # type: ignore[import-untyped]

from everalgo.rank.fusion import rrf as _algo_rrf
from everalgo.types import Candidate as _Candidate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarks.common.services import EmbeddingClient, LLMClient, RerankClient

logger = logging.getLogger(__name__)

# Convenience alias — every document dict is untyped at runtime.
_Doc = dict[str, Any]
_Scored = tuple[_Doc, float]


def _fuse_with_algo_rrf(sources: Sequence[Sequence[_Scored]], *, k: int) -> list[_Scored]:
    """Adapter: fuse ``(doc, score)`` ranked lists via ``everalgo.rank.fusion.rrf``.

    Wraps each doc as a ``Candidate`` (carrying the doc reference in
    ``metadata["_doc"]``), delegates fusion to the algo primitive, then unwraps
    back into ``(doc, rrf_score)`` tuples. Sources with an empty doc.id are
    skipped (rrf needs a stable id for cross-source dedup).
    """
    cand_sources: list[list[_Candidate]] = []
    for src in sources:
        cands: list[_Candidate] = []
        for doc, score in src:
            doc_id = doc.get("id")
            if not isinstance(doc_id, str) or not doc_id:
                continue
            cands.append(_Candidate(id=doc_id, score=float(score), metadata={"_doc": doc}))
        cand_sources.append(cands)
    fused = _algo_rrf(*cand_sources, k=k)
    out: list[_Scored] = []
    for c in fused:
        raw_doc = c.metadata.get("_doc")
        if isinstance(raw_doc, dict):
            out.append((raw_doc, float(c.score)))
    return out


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

    Reads the locomo-benchmark-style ``bm25_index`` payload
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
    query_norm: float,
) -> _Scored | None:
    """Score an embedding index entry via MaxSim across all stored embeddings.

    Collects every available vector (each ``atomic_fact``, plus ``subject`` /
    ``summary`` / ``content`` when present) into one list and takes the max
    cosine similarity against the query. Mirror locomo-benchmark
    ``stage2_index_building.py:332-379`` + ``stage3_memory_retrivel`` semantics:
    no short-circuit — facts, subject and summary all participate.

    The unused ``query_norm`` parameter is kept for signature stability with
    legacy call sites; ``compute_maxsim_score`` recomputes norms internally.
    """
    del query_norm  # see docstring
    doc: _Doc = item.get("doc") or {}
    embeddings: dict[str, Any] = item.get("embeddings") or {}
    if not embeddings:
        return None

    all_vecs: list[np.ndarray] = []
    atomic_fact_embs = embeddings.get("atomic_facts")
    if isinstance(atomic_fact_embs, list):
        all_vecs.extend(atomic_fact_embs)
    for field in ("subject", "summary", "content"):
        field_emb = embeddings.get(field)
        if field_emb is not None:
            all_vecs.append(field_emb)

    if not all_vecs:
        return None

    return (doc, compute_maxsim_score(query_vec, all_vecs))


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
    query_norm = float(np.linalg.norm(query_vec))
    if query_norm == 0:
        return []

    doc_scores: list[_Scored] = [
        result for item in emb_index if (result := _score_emb_item(item, query_vec, query_norm)) is not None
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


def _first_atomic_fact(doc: dict[str, Any]) -> str:
    """Return the first atomic_fact text in the doc, or empty if absent."""
    facts = doc.get("atomic_facts") or []
    if not facts:
        return ""
    first = facts[0]
    if isinstance(first, dict):
        fact = cast("dict[str, Any]", first).get("fact") or ""
        return fact if isinstance(fact, str) else ""
    return first if isinstance(first, str) else ""


def _format_doc_for_rerank(doc: dict[str, Any]) -> str | None:
    """Format a doc for reranker input — mirror locomo-benchmark.

    Reference: ``rerank_deepinfra.py:219-255`` ``_extract_text_from_hit`` generic
    fallback chain. LoCoMo memcells carry no ``memory_type`` field, so 93 walks
    ``episode → atomic_fact → foresight → content → summary → subject`` and returns
    the first non-empty match. For our nested-episode schema that resolves to
    ``episode.content → first atomic_fact → episode.summary → episode.subject``
    (foresight is unused; ``content`` already lives inside our ``episode`` dict).

    No timestamp prefix, no atomic_facts concatenation — the previous
    EverAlgo-self-invented format was neither in main nor in 93.
    """
    episode = doc.get("episode")
    if isinstance(episode, str) and episode:
        return episode

    for source in (
        _episode_field(doc, "content"),
        _first_atomic_fact(doc),
        _episode_field(doc, "summary"),
        _episode_field(doc, "subject"),
    ):
        if source:
            return source
    return None


@overload
async def hybrid_search_with_rrf(
    query: str,
    *,
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
    embedding_client: EmbeddingClient,
    top_n: int = ...,
    emb_candidates: int = ...,
    bm25_candidates: int = ...,
    rrf_k: int = ...,
    query_embedding: np.ndarray | None = ...,
    return_components: Literal[False] = ...,
) -> list[_Scored]: ...


@overload
async def hybrid_search_with_rrf(
    query: str,
    *,
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
    embedding_client: EmbeddingClient,
    top_n: int = ...,
    emb_candidates: int = ...,
    bm25_candidates: int = ...,
    rrf_k: int = ...,
    query_embedding: np.ndarray | None = ...,
    return_components: Literal[True],
) -> tuple[list[_Scored], list[_Scored], list[_Scored]]: ...


async def hybrid_search_with_rrf(
    query: str,
    *,
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
    embedding_client: EmbeddingClient,
    top_n: int = 40,
    emb_candidates: int = 50,
    bm25_candidates: int = 50,
    rrf_k: int = 60,
    query_embedding: np.ndarray | None = None,
    return_components: bool = False,
) -> list[_Scored] | tuple[list[_Scored], list[_Scored], list[_Scored]]:
    """Run Embedding + BM25 in parallel, fuse with RRF, return top-n.

    Executes both retrieval branches concurrently via ``asyncio.gather``,
    then fuses with Reciprocal Rank Fusion. Falls back to the populated
    branch when the other returns nothing.

    Args:
        query: Raw query text.
        emb_index: Pre-built embedding index entries.
        bm25_index: Fact-level BM25 payload dict
            (``{"bm25", "docs", "fact_to_doc_idx", "index_type"}``) consumed by
            ``search_with_bm25_index``.
        embedding_client: Service used to embed the query.
        top_n: Maximum number of fused results to return.
        emb_candidates: Candidates fetched from embedding search.
        bm25_candidates: Candidates fetched from BM25 search.
        rrf_k: RRF smoothing constant (empirical optimum: 60).
        query_embedding: Optional pre-computed query embedding.
        return_components: When True, also return the pre-fusion embedding and
            BM25 result lists for diagnostic tracing. Default False preserves
            the simple list return.

    Returns:
        If ``return_components=False``: fused list of ``(doc, rrf_score)`` pairs.
        If ``return_components=True``: tuple of ``(fused, emb_results, bm25_results)``
        where ``emb_results`` / ``bm25_results`` hold the raw per-branch scores
        before RRF.
    """
    emb_task = search_with_emb_index(
        query,
        emb_index,
        top_n=emb_candidates,
        embedding_client=embedding_client,
        query_embedding=query_embedding,
    )
    bm25_task = asyncio.to_thread(search_with_bm25_index, query, bm25_index, bm25_candidates)
    emb_results, bm25_results = await asyncio.gather(emb_task, bm25_task)

    if not emb_results and not bm25_results:
        fused: list[_Scored] = []
    elif not emb_results:
        fused = bm25_results[:top_n]
    elif not bm25_results:
        fused = emb_results[:top_n]
    else:
        fused = _fuse_with_algo_rrf([emb_results, bm25_results], k=rrf_k)[:top_n]

    if return_components:
        return fused, emb_results, bm25_results
    return fused


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
    (timestamp + one fact per line) to match EverCore scoring format.
    Documents without usable text are silently skipped.

    Batches are processed with controlled concurrency and per-batch retry
    with exponential backoff. Falls back to the original ranking when the
    reranker success rate drops below ``fallback_threshold``.

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
        fallback_threshold: Minimum success rate before falling back.

    Returns:
        List of ``(doc, reranker_score)`` pairs sorted by score descending.
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

    # Fallback: all failed or success rate too low → return original ranking.
    if not all_scored or success_rate < fallback_threshold:
        return results[:top_n]

    # Step 4: Sort by reranker score and map back to original documents.
    all_scored.sort(key=lambda x: x[1], reverse=True)
    top_scored = all_scored[:top_n]
    return [(results[original_indices[idx]][0], score) for idx, score in top_scored]


# ---------------------------------------------------------------------------
# T14c — Agentic retrieval loop + pipeline entry point
# ---------------------------------------------------------------------------


def _trace_scored(results: list[_Scored], *, limit: int) -> list[dict[str, Any]]:
    """Reduce a ``[(doc, score), ...]`` list to a compact trace shape for diagnostics.

    Stored shape per entry: ``{"id": <memcell_id_str>, "score": <float, 4 dp>}``.
    Limits length to bound dump size — typical limits: 30 for raw branch (BM25/Emb)
    top-N, 20 for hybrid/rerank top-N, 10 for final top-N. Used only for offline
    analysis of "did the gold memcell make it into round 1 / why did rerank drop it".
    """
    return [{"id": str(doc.get("id", "")), "score": round(float(score), 4)} for doc, score in results[:limit]]


async def agentic_retrieval(
    query: str,
    *,
    config: Any,
    llm: LLMClient,
    embedding_client: EmbeddingClient,
    rerank_client: RerankClient,
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
) -> tuple[list[_Scored], dict[str, Any]]:
    """Multi-round LLM-guided retrieval.

    Implements EverCore's agentic_retrieval logic (stage3_memory_retrivel.py
    lines ~591-892).  Round 1 produces top-20 hybrid candidates, a reranker
    trims to top-10, and an LLM judge decides whether they are sufficient.  If
    not, up to 3 refined queries are generated, run in parallel, fused with
    multi-RRF, merged with Round 1, and reranked to top-20.

    Args:
        query: Raw user question.
        config: ``BenchmarkConfig`` instance (provides hybrid / reranker knobs).
        llm: ``LLMClient`` used for the sufficiency check and query refinement.
        embedding_client: Client used to embed queries.
        rerank_client: Qwen3-Reranker client.
        emb_index: Per-conversation embedding index.
        bm25_index: Per-conversation fact-level BM25 payload dict
            (``{"bm25", "docs", "fact_to_doc_idx", "index_type"}``) — the
            ``(query, top_n) -> list[(doc, score)]`` retriever wrapper resolves
            fact-level scores back to doc-level via MaxSim.

    Returns:
        ``(final_results, metadata)`` — ``final_results`` is a list of
        ``(doc, score)`` pairs; ``metadata`` is a diagnostic dict.
    """
    from benchmarks.common.stages import _agentic_utils

    start_time = time.monotonic()

    metadata: dict[str, Any] = {
        "is_multi_round": False,
        "round1_count": 0,
        "round1_reranked_count": 0,
        "round2_count": 0,
        "is_sufficient": None,
        "reasoning": None,
        "key_information_found": [],
        "refined_queries": [],
        "final_count": 0,
        "total_latency_ms": 0.0,
        # Per-stage diagnostic trace — see _trace_scored docstring for shape.
        "trace": {
            "r1_emb_top": [],
            "r1_bm25_top": [],
            "r1_rrf_top": [],
            "r1_rerank_top": [],
            "r2_subqueries": [],
            "r2_fused_top": [],
            "final_top": [],
        },
    }

    # ---- Round 1: hybrid → top 20 ----------------------------------------
    round1_top20, r1_emb_results, r1_bm25_results = await hybrid_search_with_rrf(
        query,
        emb_index=emb_index,
        bm25_index=bm25_index,
        embedding_client=embedding_client,
        top_n=20,
        emb_candidates=config.hybrid_emb_candidates,
        bm25_candidates=config.hybrid_bm25_candidates,
        rrf_k=config.hybrid_rrf_k,
        return_components=True,
    )
    metadata["round1_count"] = len(round1_top20)
    metadata["trace"]["r1_emb_top"] = _trace_scored(r1_emb_results, limit=30)
    metadata["trace"]["r1_bm25_top"] = _trace_scored(r1_bm25_results, limit=30)
    metadata["trace"]["r1_rrf_top"] = _trace_scored(round1_top20, limit=20)

    if not round1_top20:
        metadata["total_latency_ms"] = (time.monotonic() - start_time) * 1000
        return [], metadata

    # ---- Rerank → top ``round1_rerank_top_n`` (for sufficiency check) ----
    rerank_top_n: int = config.round1_rerank_top_n
    if config.use_reranker:
        reranked_topn = await reranker_search(
            query,
            results=round1_top20,
            rerank_client=rerank_client,
            top_n=rerank_top_n,
            reranker_instruction=config.reranker_instruction,
            batch_size=config.reranker_batch_size,
            concurrent_batches=config.reranker_concurrent_batches,
            max_retries=config.reranker_max_retries,
            retry_delay=config.reranker_retry_delay,
            timeout=config.reranker_timeout,
            fallback_threshold=config.reranker_fallback_threshold,
        )
    else:
        reranked_topn = round1_top20[:rerank_top_n]

    metadata["round1_reranked_count"] = len(reranked_topn)
    metadata["trace"]["r1_rerank_top"] = _trace_scored(reranked_topn, limit=rerank_top_n)

    if not reranked_topn:
        metadata["trace"]["final_top"] = _trace_scored(round1_top20, limit=20)
        metadata["total_latency_ms"] = (time.monotonic() - start_time) * 1000
        return round1_top20, metadata

    # ---- LLM sufficiency check -------------------------------------------
    is_sufficient, reasoning, missing_info, key_info, sufficiency_tokens = await _agentic_utils.check_sufficiency(
        query,
        reranked_topn,
        llm=llm,
        judge_model=getattr(config, "judge_model", None),
        max_docs=rerank_top_n,
    )
    metadata["prompt_tokens"] = metadata.get("prompt_tokens", 0) + sufficiency_tokens.get("prompt_tokens", 0)
    metadata["completion_tokens"] = metadata.get("completion_tokens", 0) + sufficiency_tokens.get(
        "completion_tokens", 0
    )

    metadata["is_sufficient"] = is_sufficient
    metadata["reasoning"] = reasoning
    metadata["key_information_found"] = key_info

    if is_sufficient:
        # Mirror locomo-benchmark ``retrieval_utils.py:676-684``: when the LLM
        # judges the round-1 rerank top-N as sufficient, return the *Round-1
        # RRF-fused top 20* — not the reranker's top-N. The reranker only
        # selects the docs the sufficiency-check LLM sees; final results stay
        # in hybrid-RRF order. The downstream answer stage applies
        # ``response_top_k`` to truncate this list.
        metadata["final_count"] = len(round1_top20)
        metadata["trace"]["final_top"] = _trace_scored(round1_top20, limit=20)
        metadata["total_latency_ms"] = (time.monotonic() - start_time) * 1000
        return round1_top20, metadata

    # ---- Round 2: query refinement + parallel hybrid ----------------------
    metadata["is_multi_round"] = True
    metadata["missing_info"] = missing_info

    use_multi_query: bool = getattr(config, "use_multi_query", True)

    if use_multi_query:
        refined_queries, query_strategy, multi_query_tokens = await _agentic_utils.generate_multi_queries(
            original_query=query,
            results=reranked_topn,
            missing_info=missing_info,
            llm=llm,
            judge_model=getattr(config, "judge_model", None),
            key_info=key_info,
            max_docs=rerank_top_n,
            num_queries=3,
        )
        metadata["prompt_tokens"] = metadata.get("prompt_tokens", 0) + multi_query_tokens.get("prompt_tokens", 0)
        metadata["completion_tokens"] = metadata.get("completion_tokens", 0) + multi_query_tokens.get(
            "completion_tokens", 0
        )
        metadata["refined_queries"] = refined_queries
        metadata["query_strategy"] = query_strategy
        metadata["num_queries"] = len(refined_queries)

        multi_query_tasks = [
            hybrid_search_with_rrf(
                q,
                emb_index=emb_index,
                bm25_index=bm25_index,
                embedding_client=embedding_client,
                top_n=50,
                emb_candidates=config.hybrid_emb_candidates,
                bm25_candidates=config.hybrid_bm25_candidates,
                rrf_k=config.hybrid_rrf_k,
                return_components=True,
            )
            for q in refined_queries
        ]
        multi_query_components = await asyncio.gather(*multi_query_tasks)
        multi_query_fused = [comp[0] for comp in multi_query_components]
        if len(multi_query_fused) == 1:
            round2_results = list(multi_query_fused[0])[:40]
        else:
            round2_results = _fuse_with_algo_rrf(multi_query_fused, k=config.hybrid_rrf_k)[:40]
        metadata["multi_query_total_docs"] = sum(len(r) for r in multi_query_fused)
        metadata["trace"]["r2_subqueries"] = [
            {
                "query": q,
                "emb_top": _trace_scored(emb_r, limit=30),
                "bm25_top": _trace_scored(bm25_r, limit=30),
                "rrf_top": _trace_scored(fused, limit=30),
            }
            for q, (fused, emb_r, bm25_r) in zip(refined_queries, multi_query_components, strict=True)
        ]

    else:
        refined_query, refined_tokens = await _agentic_utils.generate_refined_query(
            original_query=query,
            results=reranked_topn,
            missing_info=missing_info,
            llm=llm,
            judge_model=getattr(config, "judge_model", None),
            key_info=key_info,
            max_docs=rerank_top_n,
        )
        metadata["prompt_tokens"] = metadata.get("prompt_tokens", 0) + refined_tokens.get("prompt_tokens", 0)
        metadata["completion_tokens"] = metadata.get("completion_tokens", 0) + refined_tokens.get(
            "completion_tokens", 0
        )
        metadata["refined_query"] = refined_query
        round2_results, r2_emb_results, r2_bm25_results = await hybrid_search_with_rrf(
            refined_query,
            emb_index=emb_index,
            bm25_index=bm25_index,
            embedding_client=embedding_client,
            top_n=40,
            emb_candidates=config.hybrid_emb_candidates,
            bm25_candidates=config.hybrid_bm25_candidates,
            rrf_k=config.hybrid_rrf_k,
            return_components=True,
        )
        metadata["trace"]["r2_subqueries"] = [
            {
                "query": refined_query,
                "emb_top": _trace_scored(r2_emb_results, limit=30),
                "bm25_top": _trace_scored(r2_bm25_results, limit=30),
                "rrf_top": _trace_scored(round2_results, limit=30),
            }
        ]

    metadata["round2_count"] = len(round2_results)
    metadata["trace"]["r2_fused_top"] = _trace_scored(round2_results, limit=40)

    # ---- Merge Round 1 + Round 2 (dedup, target 40 total) ----------------
    round1_ids = {doc.get("id", id(doc)) for doc, _ in round1_top20}
    round2_unique = [(doc, score) for doc, score in round2_results if doc.get("id", id(doc)) not in round1_ids]
    combined: list[_Scored] = list(round1_top20)
    needed = 40 - len(combined)
    combined.extend(round2_unique[:needed])

    # ---- Final rerank → top 20 -------------------------------------------
    if config.use_reranker and combined:
        final_results = await reranker_search(
            query,
            results=combined,
            rerank_client=rerank_client,
            top_n=20,
            reranker_instruction=config.reranker_instruction,
            batch_size=config.reranker_batch_size,
            concurrent_batches=config.reranker_concurrent_batches,
            max_retries=config.reranker_max_retries,
            retry_delay=config.reranker_retry_delay,
            timeout=config.reranker_timeout,
            fallback_threshold=config.reranker_fallback_threshold,
        )
    else:
        final_results = combined[:20]

    metadata["final_count"] = len(final_results)
    metadata["trace"]["final_top"] = _trace_scored(final_results, limit=20)
    metadata["total_latency_ms"] = (time.monotonic() - start_time) * 1000
    return final_results, metadata


async def run_search_stage(ctx: Any) -> Any:
    """Stage 3 — agentic retrieval for every (conv, question).

    Loads per-conversation BM25 + embedding indices written by Stage 2
    (``index.py``), runs ``agentic_retrieval`` for every QA pair, and writes
    ``search_results.json`` to ``ctx.output_dir``.

    Args:
        ctx: ``StageContext`` instance (typed as ``Any`` to avoid a circular
            import; the actual type is ``benchmarks.common.stages.types.StageContext``).

    Returns:
        ``StageStats`` with ``stage_name="search"`` and success / failed counts.
    """
    from benchmarks.common.stages.types import StageStats

    _ensure_nltk()
    ctx.output_dir.mkdir(parents=True, exist_ok=True)

    start_wall = time.monotonic()
    search_results: dict[str, list[dict[str, Any]]] = {}
    success_total = 0
    failed_total = 0

    convs = list(ctx.dataset.load_conversations())
    if ctx.smoke:
        convs = convs[: ctx.smoke_conv_limit]

    filter_cats = ctx.dataset.filter_categories()

    for conv_num, conv in enumerate(convs, start=1):
        conv_id = conv.id
        # Derive per-conversation index index (strip "locomo_exp_user_" prefix)
        conv_idx = _conv_index(conv_id)

        bm25_path = ctx.input_dir / f"bm25_conv_{conv_idx}.pkl"
        emb_path = ctx.input_dir / f"emb_conv_{conv_idx}.pkl"

        if not bm25_path.exists() or not emb_path.exists():
            logger.warning("Skipping %s: index files not found in %s", conv_id, ctx.input_dir)
            continue

        with bm25_path.open("rb") as fh:
            bm25_index: dict[str, Any] = pickle.load(fh)

        with emb_path.open("rb") as fh:
            emb_index: list[dict[str, Any]] = pickle.load(fh)

        scene_index = _load_scene_index(
            ctx.input_dir, conv_idx, enable=getattr(ctx.config, "enable_scene_retrieval", False)
        )

        qa_pairs = list(ctx.dataset.load_qa_pairs(conv_id))
        if ctx.smoke:
            qa_pairs = qa_pairs[: ctx.smoke_qa_limit]

        total_qa = len(qa_pairs)

        sem = asyncio.Semaphore(ctx.config.max_concurrent_qa)

        from tqdm.asyncio import tqdm as async_tqdm  # type: ignore[import-untyped]

        raw_results: list[dict[str, Any] | None] = await async_tqdm.gather(  # type: ignore[attr-defined]
            *(
                _process_single_qa(
                    qa,
                    qa_idx=qa_idx,
                    total_qa=total_qa,
                    ctx=ctx,
                    sem=sem,
                    emb_index=emb_index,
                    bm25_index=bm25_index,
                    scene_index=scene_index,
                    filter_cats=filter_cats,
                )
                for qa_idx, qa in enumerate(qa_pairs, start=1)
            ),
            desc=f"search {conv_num}/{len(convs)}",
            unit="q",
            dynamic_ncols=True,
        )

        conv_results: list[dict[str, Any]] = []
        for item in raw_results:
            if item is None:
                failed_total += 1
            else:
                conv_results.append(item)
                success_total += 1

        search_results[conv_id] = conv_results

    output_path = ctx.output_dir / "search_results.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(search_results, fh, indent=2, ensure_ascii=False)

    # Aggregate token counts from per-QA retrieval metadata
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for conv_results in search_results.values():
        for item in conv_results:
            meta: dict[str, Any] = item.get("retrieval_metadata", {})
            total_prompt_tokens += meta.get("prompt_tokens", 0)
            total_completion_tokens += meta.get("completion_tokens", 0)

    duration = time.monotonic() - start_wall
    return StageStats(
        stage_name="search",
        duration_seconds=duration,
        success=success_total,
        failed=failed_total,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
    )


async def _process_single_qa(
    qa: Any,
    *,
    qa_idx: int,
    total_qa: int,
    ctx: Any,
    sem: asyncio.Semaphore,
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
    scene_index: dict[str, Any] | None,
    filter_cats: set[str],
) -> dict[str, Any] | None:
    """Run agentic retrieval for a single QA pair.

    Returns a populated result dict on success, or ``None`` when the question
    is filtered out or retrieval raises an exception.  All per-conv loop
    variables are passed as explicit arguments to avoid B023 closure capture.

    Args:
        qa: QAPair dataclass.
        qa_idx: 1-based question index within the conversation (for progress display).
        total_qa: Total questions in this conversation (for progress display).
        ctx: ``StageContext`` providing config, services, and output paths.
        sem: Per-conversation concurrency semaphore.
        emb_index: Per-conversation embedding index.
        bm25_index: Per-conversation fact-level BM25 payload dict
            (``{"bm25", "docs", "fact_to_doc_idx", "index_type"}``).
        scene_index: Per-conversation scene index (``None`` when unavailable or disabled).
        filter_cats: Category codes to skip (e.g. ``{"5"}`` for adversarial).

    Returns:
        Result dict or ``None``.
    """
    if qa.category in filter_cats:
        return None
    try:
        async with sem:
            if scene_index is not None and getattr(ctx.config, "enable_scene_retrieval", False):
                from benchmarks.common.stages.scene_search import scene_agentic_retrieval

                top_results, retrieval_metadata = await scene_agentic_retrieval(
                    qa.question,
                    scene_index=scene_index,
                    emb_index=emb_index,
                    bm25_index=bm25_index,
                    config=ctx.config,
                    llm=ctx.services.llm,
                    embedding_client=ctx.services.embedding,
                    rerank_client=ctx.services.rerank,
                )
            else:
                top_results, retrieval_metadata = await agentic_retrieval(
                    qa.question,
                    config=ctx.config,
                    llm=ctx.services.llm,
                    embedding_client=ctx.services.embedding,
                    rerank_client=ctx.services.rerank,
                    emb_index=emb_index,
                    bm25_index=bm25_index,
                )
    except Exception:
        logger.exception("Retrieval failed for question_id=%s", qa.question_id)
        return None

    memcell_ids = [doc["id"] for doc, _ in top_results if doc.get("id") is not None]
    return {
        "question_id": qa.question_id,
        "query": qa.question,
        "memcell_ids": memcell_ids,
        "original_qa": {
            "question_id": qa.question_id,
            "conv_id": qa.conv_id,
            "question": qa.question,
            "golden_answer": qa.golden_answer,
            "category": qa.category,
        },
        "retrieval_metadata": retrieval_metadata,
    }


def _load_scene_index(input_dir: Any, conv_idx: int, *, enable: bool) -> dict[str, Any] | None:
    """Load scene index pickle if ``enable`` is True and the file exists.

    Returns None (with a warning) when the file is absent or scene retrieval is disabled.
    """
    if not enable:
        return None
    scene_path = input_dir / f"scene_index_conv_{conv_idx}.pkl"
    if scene_path.exists():
        with scene_path.open("rb") as fh:
            return cast("dict[str, Any]", pickle.load(fh))
    logger.warning("enable_scene_retrieval=True but scene index not found: %s", scene_path)
    return None


def _conv_index(conv_id: str) -> int:
    """Extract the numeric suffix from a ``locomo_exp_user_<N>`` conv_id.

    Falls back to 0 on parse failure rather than raising, so a malformed
    conv_id only skips that conversation instead of aborting the stage.
    """
    prefix = "locomo_exp_user_"
    try:
        return int(conv_id.removeprefix(prefix))
    except ValueError:
        logger.warning("Cannot parse conv index from %r, defaulting to 0", conv_id)
        return 0
