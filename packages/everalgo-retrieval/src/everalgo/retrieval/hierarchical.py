"""Hierarchical retrieval — generic parent-child expansion (MRAG-style).

Generalizes the previous ``rank.fusion.expand`` which was hard-coded to
episode→atomic_fact. Now any (parent_kind, child_kind) pair works as long as
caller supplies the three retrieve callables.

Algorithm overview
------------------
Phase 1  — dual fusion: dense + sparse results are combined with both RRF and LR
           so the full score signal is available for later phases.
Phase 2  — child prefetch: for the top ``expand_limit`` parents, fetch their
           children via ``child_retrieve_for_parent``.
Phase 3  — heap convergence: max-heap over parents; each popped parent's children
           compete to enter the top-N set, evicting their parent if they win.
Phase 4  — convergence check: stop after ``max_convergence_rounds`` stable
           iterations (top-N unchanged) or when the heap is exhausted.

Returns a single ``list[Candidate]``. Surviving children carry
``metadata["source_kind"] = "child"`` and ``metadata["parent_id"] = <parent.id>``.
Parents carry ``metadata["source_kind"] = "parent"``.
Convergence diagnostics are emitted to ``logger.debug`` only — no metadata dict.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.rank.fusion import cosine_to_lr_score, lr, rrf
from everalgo.types import Candidate

if TYPE_CHECKING:
    from everalgo.rank.weight import LRCoefs
    from everalgo.retrieval.protocols import RetrieveFn

__all__ = ["ahierarchical_retrieve", "hierarchical_retrieve"]

logger = logging.getLogger(__name__)

ChildRetrieveFn = Callable[[str, Candidate, int], Awaitable[list[Candidate]]]
"""``retrieve(query, parent, child_top_n) -> list[Candidate]`` — child recall for a given parent."""

# Internal type alias: (candidate, float score, "parent"|"child", parent_id)
_TopNEntry = tuple[Candidate, float, str, str]


async def ahierarchical_retrieve(
    query: str,
    *,
    parent_dense_retrieve: RetrieveFn,
    parent_sparse_retrieve: RetrieveFn,
    child_retrieve_for_parent: ChildRetrieveFn,
    response_top_k: int = 20,
    parent_candidates: int = 50,
    child_top_n_per_parent: int = 10,
    children_per_parent: int | None = None,
    rrf_k: int = 60,
    lr_coefs: LRCoefs | None = None,
    expand_limit: int | None = None,
    alpha: float | None = None,
    max_convergence_rounds: int | None = None,
    min_score: float | None = None,
) -> list[Candidate]:
    """Generic parent-child hierarchical retrieval (Phase 1 RRF+LR + Phase 2-4 heap convergence).

    Args:
        query: The retrieval query string.
        parent_dense_retrieve: Async callable ``(query, k) -> list[Candidate]`` for dense (embedding) recall.
        parent_sparse_retrieve: Async callable ``(query, k) -> list[Candidate]`` for sparse (BM25/keyword) recall.
        child_retrieve_for_parent: Async callable ``(query, parent, child_top_n) -> list[Candidate]`` — given
            a parent candidate, returns its top-k child candidates. Caller binds the storage index inside.
        response_top_k: Maximum number of candidates to return in the final result.
        parent_candidates: How many parent candidates to fetch from each retrieve route (dense + sparse).
        child_top_n_per_parent: How many children to fetch per parent during Phase 2 prefetch.
        children_per_parent: Maximum number of children per parent that compete in the top-N during Phase 3-4
            heap convergence. When ``None``, resolves to ``DEFAULT_RANK_CONFIG.expand_limit`` (default 3).
        rrf_k: Reciprocal Rank Fusion smoothing constant (default 60 — standard Cormack et al. 2009).
        lr_coefs: Optional LR weighting coefficients for Phase 1 LR fusion and child score calibration.
            When ``None``, defers to ``weight.default_lr_coefs()``.
        expand_limit: Maximum number of parents to expand (prefetch children for). When ``None``,
            resolves to ``DEFAULT_RANK_CONFIG.expand_limit``.
        alpha: Parent-child score blend weight. ``final = alpha*child + (1-alpha)*parent``.
            When ``None``, resolves to ``DEFAULT_RANK_CONFIG.alpha``.
        max_convergence_rounds: Stop heap convergence after this many stable iterations (top-N unchanged).
            When ``None``, resolves to ``DEFAULT_RANK_CONFIG.max_convergence_rounds``.
        min_score: When set, candidates whose score is strictly below this threshold are filtered from
            the final result after Phase 4 convergence (default ``None`` — no filtering).

    Returns:
        Up to ``response_top_k`` candidates sorted descending by score. Parents carry
        ``metadata["source_kind"] = "parent"``; surviving children carry
        ``metadata["source_kind"] = "child"`` and ``metadata["parent_id"] = <parent.id>``.

    Note:
        Resource contract: this function does I/O (LLM/index calls via the three retrieve callables)
        and must be awaited. It is async-first; use ``asgiref.async_to_sync`` for sync callers outside
        an event loop.
    """
    from everalgo.rank import DEFAULT_RANK_CONFIG

    effective_expand_limit = expand_limit if expand_limit is not None else DEFAULT_RANK_CONFIG.expand_limit
    effective_alpha = alpha if alpha is not None else DEFAULT_RANK_CONFIG.alpha
    effective_max_rounds = (
        max_convergence_rounds if max_convergence_rounds is not None else DEFAULT_RANK_CONFIG.max_convergence_rounds
    )
    effective_children_per_parent = (
        children_per_parent if children_per_parent is not None else DEFAULT_RANK_CONFIG.expand_limit
    )

    dense = await parent_dense_retrieve(query, parent_candidates)
    sparse = await parent_sparse_retrieve(query, parent_candidates)

    if not dense and not sparse:
        return []
    if not dense:
        return list(sparse)[:response_top_k]
    if not sparse:
        return list(dense)[:response_top_k]

    # Phase 1: dual fusion — keep both columns for LR weighting signal
    lr_results = lr(dense, sparse, coefs=lr_coefs)
    parent_scores: dict[str, float] = {c.id: c.score for c in lr_results}
    fused = rrf(dense, sparse, k=rrf_k)
    bm25_scores: dict[str, float] = {c.id: c.score for c in sparse}

    if not fused:
        return []

    # Phase 2: prefetch children for top ``expand_limit`` parents (parallel gather)
    parents_to_expand = fused[:effective_expand_limit]
    children_lists = await asyncio.gather(
        *(child_retrieve_for_parent(query, parent, child_top_n_per_parent) for parent in parents_to_expand)
    )
    prefetched_children: dict[str, list[Candidate]] = {
        parent.id: children for parent, children in zip(parents_to_expand, children_lists, strict=True)
    }

    # Phase 3-4: heap convergence — returns flat list[Candidate]
    result = _expand_heap(
        fused_results=fused,
        parent_scores=parent_scores,
        prefetched_children=prefetched_children,
        bm25_scores=bm25_scores,
        response_top_k=response_top_k,
        alpha=effective_alpha,
        max_convergence_rounds=effective_max_rounds,
        lr_coefs=lr_coefs,
        children_per_parent=effective_children_per_parent,
    )
    if min_score is not None:
        result = [c for c in result if c.score >= min_score]
    return result


hierarchical_retrieve = async_to_sync(ahierarchical_retrieve)
"""Sync bridge for non-event-loop contexts (CLI / pytest). Per ADR 010."""


def _expand_one_parent(
    *,
    parent_id: str,
    topn: dict[str, _TopNEntry],
    prefetched_children: dict[str, list[Candidate]],
    parent_scores: dict[str, float],
    bm25_map: dict[str, float],
    alpha: float,
    children_per_parent: int,
    response_top_k: int,
    lr_coefs: LRCoefs | None,
) -> None:
    """Phase 3 iteration: score one parent's children and let them compete in top-N.

    Mutates ``topn`` in place. No-ops when the parent has no pre-fetched children.
    When any child enters top-N, the parent is evicted from top-N.

    Args:
        parent_id: ID of the parent being expanded.
        topn: Current top-N dict (mutated in place). Keys are slot identifiers.
        prefetched_children: Maps parent IDs to their pre-fetched child candidates.
        parent_scores: LR-fused parent scores (Phase 1 output), used for alpha-blend.
        bm25_map: Sparse (BM25) scores for each parent, used for LR child calibration.
        alpha: Parent-child blend weight. ``final = alpha*child_lr + (1-alpha)*parent``.
        children_per_parent: Cap on how many children to score per expansion.
        response_top_k: Top-N size limit.
        lr_coefs: LR weighting coefficients for child cosine calibration.
    """
    children = prefetched_children.get(parent_id, [])
    if not children:
        return

    parent_score = parent_scores.get(parent_id, 0.0)
    parent_bm25 = bm25_map.get(parent_id, 0.0)

    # Score children: calibrate cosine to LR probability, then blend with parent
    scored_children: list[tuple[int, float]] = []
    for i, child in enumerate(children[: children_per_parent * 2]):
        child_cosine = child.score
        child_lr_score = cosine_to_lr_score(child_cosine, parent_bm25, coefs=lr_coefs)
        final_score = alpha * child_lr_score + (1.0 - alpha) * parent_score
        scored_children.append((i, final_score))

    scored_children.sort(key=lambda kv: kv[1], reverse=True)
    top_scored = scored_children[:children_per_parent]

    min_topn_score = min((v[1] for v in topn.values()), default=-1.0)

    any_child_entered = False
    for idx, final_score in top_scored:
        if final_score <= 0:
            continue
        if len(topn) < response_top_k or final_score > min_topn_score:
            child = children[idx]
            child_key = f"child_{child.id}" if child.id else f"child_{parent_id}__{idx}"
            child_with_metadata = child.model_copy(
                update={
                    "score": final_score,
                    "metadata": {**child.metadata, "source_kind": "child", "parent_id": parent_id},
                }
            )
            topn[child_key] = (child_with_metadata, final_score, "child", parent_id)
            any_child_entered = True

            while len(topn) > response_top_k:
                worst_key = min(topn, key=lambda k: topn[k][1])
                del topn[worst_key]

            min_topn_score = min((v[1] for v in topn.values()), default=-1.0)

    if any_child_entered and parent_id in topn:
        del topn[parent_id]


def _expand_heap(
    fused_results: list[Candidate],
    parent_scores: dict[str, float],
    prefetched_children: dict[str, list[Candidate]],
    *,
    bm25_scores: dict[str, float] | None = None,
    response_top_k: int,
    alpha: float = 1.0,
    max_convergence_rounds: int = 10,
    lr_coefs: LRCoefs | None = None,
    children_per_parent: int = 3,
) -> list[Candidate]:
    """Phase 3-4 heap convergence — top-N selection with child eviction.

    Max-heap pops parents by fused score; each parent's children compete with
    current top-N via cosine_to_lr_score weighting; iterates until stable for
    max_convergence_rounds or heap is exhausted.

    Args:
        fused_results: Phase 1 RRF-fused parent candidates (sorted desc by fused score).
        parent_scores: LR-fused score per parent ID (Phase 1 LR output).
        prefetched_children: Phase 2 output — maps parent IDs to child candidates.
        bm25_scores: Sparse (BM25) score per parent ID for child LR calibration.
        response_top_k: Maximum result set size.
        alpha: Parent-child blend weight.
        max_convergence_rounds: Stop after this many consecutive stable top-N iterations.
        lr_coefs: LR weighting coefficients for child cosine calibration.
        children_per_parent: Max children per parent that compete in top-N during expansion.

    Returns:
        Up to ``response_top_k`` candidates sorted descending by score.
    """
    bm25_map = bm25_scores or {}

    heap: list[tuple[float, str]] = []

    for doc in fused_results:
        if not doc.id:
            continue
        heapq.heappush(heap, (-doc.score, doc.id))

    # Seed top-N with initial fused results, using LR-fused scores for ranking
    topn: dict[str, _TopNEntry] = {}
    for doc in fused_results[:response_top_k]:
        if doc.id:
            score = parent_scores.get(doc.id, 0.0)
            parent_with_metadata = doc.model_copy(
                update={"score": score, "metadata": {**doc.metadata, "source_kind": "parent"}}
            )
            topn[doc.id] = (parent_with_metadata, score, "parent", doc.id)

    prev_topn_keys = frozenset(topn.keys())
    convergence_count = 0
    expansions = 0
    total_iterations = 0

    logger.debug("Phase 2: heap=%d, initial top-N=%d", len(heap), len(topn))

    while heap and convergence_count < max_convergence_rounds:
        total_iterations += 1
        _neg_fused, parent_id = heapq.heappop(heap)

        _expand_one_parent(
            parent_id=parent_id,
            topn=topn,
            prefetched_children=prefetched_children,
            parent_scores=parent_scores,
            bm25_map=bm25_map,
            alpha=alpha,
            children_per_parent=children_per_parent,
            response_top_k=response_top_k,
            lr_coefs=lr_coefs,
        )
        expansions += 1

        current_keys = frozenset(topn.keys())
        if current_keys == prev_topn_keys:
            convergence_count += 1
        else:
            convergence_count = 0
            prev_topn_keys = current_keys

    stop_reason = "convergence" if convergence_count >= max_convergence_rounds else "heap_exhausted"
    logger.debug(
        "Phase 3: iterations=%d expansions=%d convergence=%d/%d stop=%s",
        total_iterations,
        expansions,
        convergence_count,
        max_convergence_rounds,
        stop_reason,
    )

    sorted_entries = sorted(topn.values(), key=lambda v: v[1], reverse=True)
    return [entry_doc.model_copy(update={"score": score}) for entry_doc, score, _kind, _src in sorted_entries]
