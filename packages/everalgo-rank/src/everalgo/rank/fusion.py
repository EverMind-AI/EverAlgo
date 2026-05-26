"""Fusion algorithms — RRF / LR / cosine→LR / propagation / hierarchical expand / agentic LLM-guided rank."""

from __future__ import annotations

import asyncio
import heapq
import logging
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field

from everalgo.llm.types import ChatMessage
from everalgo.rank import weight as _weight
from everalgo.rank.prompts.en.agentic import (
    AGENTIC_MULTI_QUERY_PROMPT_EN,
    AGENTIC_SUFFICIENCY_CHECK_PROMPT_EN,
)
from everalgo.types import Candidate, FactCandidate

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from everalgo.llm.protocols import LLMClient
    from everalgo.rank import RankConfig
    from everalgo.rank.weight import LRCoefs

__all__ = [
    "DEFAULT_AGENTIC_CONFIG",
    "AgenticConfig",
    "RerankFn",
    "RetrieveFn",
    "aagentic_rank",
    "cosine_to_lr_score",
    "expand",
    "lr",
    "rrf",
    "score_propagation",
    "vector_anchored",
]

logger = logging.getLogger(__name__)


class SufficiencyCheckResponse(BaseModel):
    """LLM sufficiency-check response schema."""

    is_sufficient: bool = Field(..., description="Whether retrieved documents are sufficient")
    reasoning: str = Field(..., description="Reasoning for the judgment")
    missing_information: list[str] = Field(
        default_factory=list,
        description="List of missing information (empty if sufficient)",
    )


class MultiQueryResponse(BaseModel):
    """LLM multi-query generation response schema."""

    queries: list[str] = Field(..., description="Generated complementary queries (2-3 items)")
    reasoning: str = Field(..., description="Explanation of query generation strategy")


def rrf(*sources: Sequence[Candidate], k: int = 60) -> list[Candidate]:
    """Reciprocal Rank Fusion over N ranked lists; ``score = Σ 1/(k+rank_i)``.

    Returns candidates sorted descending by accumulated RRF score; empty input → empty list.
    """
    doc_rrf_scores: dict[str, float] = {}
    doc_map: dict[str, Candidate] = {}

    for ranked_list in sources:
        for rank, doc in enumerate(ranked_list, start=1):
            if not doc.id:
                continue
            doc_map.setdefault(doc.id, doc)
            doc_rrf_scores[doc.id] = doc_rrf_scores.get(doc.id, 0.0) + 1.0 / (k + rank)

    items: list[tuple[str, float]] = sorted(doc_rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_map[doc_id].model_copy(update={"score": rrf_score}) for doc_id, rrf_score in items]


def lr(
    emb_results: Sequence[Candidate],
    bm25_results: Sequence[Candidate],
    *,
    coefs: LRCoefs | None = None,
) -> list[Candidate]:
    """Logistic Regression fusion of embedding + BM25 results.

    ``logit = emb*emb_coef + bm25*bm25_coef + intercept; prob = sigmoid(logit)``.
    Returns candidates sorted descending by probability; ``coefs=None`` defers to ``weight.default_lr_coefs()``.
    """
    return _weight.multi_field_weighting(
        {"emb": list(emb_results), "bm25": list(bm25_results)},
        coefs=coefs,
    )


def vector_anchored(
    dense: Sequence[Candidate],
    sparse: Sequence[Candidate],
    *,
    saturation_k: float = 5.0,
    alpha: float = 0.7,
) -> list[Candidate]:
    """Vector-anchored fusion of dense (cosine) + sparse (BM25) candidates."""
    vec_score_map: dict[str, float] = {c.id: c.score for c in dense if c.id}
    kw_sat_map: dict[str, float] = {
        c.id: (c.score / (c.score + saturation_k) if c.score > 0 else 0.0) for c in sparse if c.id
    }

    vec_floor = min(vec_score_map.values()) if vec_score_map else 0.0
    kw_floor = min(kw_sat_map.values()) if kw_sat_map else 0.0

    doc_map: dict[str, Candidate] = {c.id: c for c in dense if c.id}
    for c in sparse:
        if c.id and c.id not in doc_map:
            doc_map[c.id] = c

    out: list[Candidate] = []
    for doc_id, doc in doc_map.items():
        vs = vec_score_map.get(doc_id, vec_floor)
        ks = kw_sat_map.get(doc_id, kw_floor)
        out.append(doc.model_copy(update={"score": alpha * vs + (1.0 - alpha) * ks}))

    out.sort(key=lambda c: c.score, reverse=True)
    return out


def cosine_to_lr_score(
    sim: float,
    parent_bm25: float = 0.0,
    *,
    coefs: LRCoefs | None = None,
) -> float:
    """Calibrate a raw cosine similarity to an LR probability in ``[0, 1]``."""
    out = _weight.multi_field_weighting(
        {
            "emb": [Candidate(id="_scalar", score=sim)],
            "bm25": [Candidate(id="_scalar", score=parent_bm25)],
        },
        coefs=coefs,
    )
    return out[0].score


def score_propagation(
    parents: Sequence[Candidate],
    children: Sequence[Candidate],
    *,
    alpha: float = 1.0,
) -> list[Candidate]:
    """Blend child + parent scores: ``final = alpha*child + (1-alpha)*parent``.

    Children whose parent cannot be resolved still appear with parent contribution treated as ``0``.
    Caller decides whether to sort the result.
    """
    parent_score_lookup = {p.id: p.score for p in parents}

    out: list[Candidate] = []
    for child in children:
        parent_id = child.metadata.get("parent_id", "")
        parent_score = parent_score_lookup.get(parent_id, 0.0) if isinstance(parent_id, str) else 0.0
        final = alpha * child.score + (1.0 - alpha) * parent_score
        out.append(child.model_copy(update={"score": final}))
    return out


_TopNEntry = tuple[Candidate | FactCandidate, float, str, str]


def _expand_one_episode(
    *,
    episode_id: str,
    topn: dict[str, _TopNEntry],
    prefetched_facts: dict[str, list[FactCandidate]],
    episode_scores: dict[str, float],
    bm25_map: dict[str, float],
    alpha: float,
    facts_per_episode: int,
    response_top_k: int,
    use_lr: bool,
    lr_coefs: LRCoefs | None,
) -> None:
    """One Phase 3 iteration: score the episode's facts and let them compete with top-N.

    Mutates ``topn`` in place. No-ops when the episode has no pre-fetched facts.
    """
    facts = prefetched_facts.get(episode_id, [])
    if not facts:
        return

    parent_score = episode_scores.get(episode_id, 0.0)
    parent_bm25 = bm25_map.get(episode_id, 0.0)

    scored_facts: list[tuple[int, float]] = []
    for i, fact in enumerate(facts[: facts_per_episode * 2]):
        child_cosine = fact.score
        child_score = cosine_to_lr_score(child_cosine, parent_bm25, coefs=lr_coefs) if use_lr else child_cosine
        final_score = alpha * child_score + (1.0 - alpha) * parent_score
        scored_facts.append((i, final_score))

    scored_facts.sort(key=lambda kv: kv[1], reverse=True)
    top_fact_scores = scored_facts[:facts_per_episode]

    min_topn_score = min((v[1] for v in topn.values()), default=-1.0)

    any_fact_entered = False
    for idx, final_score in top_fact_scores:
        if final_score <= 0:
            continue
        if len(topn) < response_top_k or final_score > min_topn_score:
            fact = facts[idx]
            fact_key = f"fact_{fact.id}" if fact.id else f"fact_{episode_id}__{idx}"
            fact_with_score = fact.model_copy(update={"score": final_score})
            topn[fact_key] = (fact_with_score, final_score, "fact", episode_id)
            any_fact_entered = True

            while len(topn) > response_top_k:
                worst_key = min(topn, key=lambda k: topn[k][1])
                del topn[worst_key]

            min_topn_score = min((v[1] for v in topn.values()), default=-1.0)

    if any_fact_entered and episode_id in topn:
        del topn[episode_id]


def expand(
    sparse: Sequence[Candidate],
    dense: Sequence[Candidate],
    episode_to_facts: dict[str, list[FactCandidate]],
    *,
    response_top_k: int,
    config: RankConfig | None = None,
    lr_coefs: LRCoefs | None = None,
) -> tuple[list[Candidate], list[FactCandidate], dict[str, Any]]:
    """Full MRAG pipeline — Phase 1 RRF+LR fusion and Phase 2-4 hierarchical expansion.

    Facts compete with their parent episode; when a fact climbs into top-N its parent is evicted. Stops when
    top-N is stable for ``max_convergence_rounds`` iterations.

    Returns ``(episodes, facts, metadata)`` where ``episodes`` / ``facts`` are the surviving top-N items.
    """
    if config is None:
        from everalgo.rank import DEFAULT_RANK_CONFIG

        config = DEFAULT_RANK_CONFIG

    sparse_list = list(sparse)
    dense_list = list(dense)

    if not sparse_list and not dense_list:
        return [], [], {"stop_reason": "no_candidates"}

    bm25_scores = {c.id: c.score for c in sparse_list}

    lr_results = (
        lr(dense_list, sparse_list, coefs=lr_coefs) if dense_list and sparse_list else list(dense_list or sparse_list)
    )
    episode_scores = {c.id: c.score for c in lr_results}

    fused_results = (
        rrf(dense_list, sparse_list, k=config.rrf_k) if dense_list and sparse_list else list(dense_list or sparse_list)
    )

    if not fused_results:
        return [], [], {"stop_reason": "no_candidates"}

    return _expand_heap(
        fused_results,
        episode_scores,
        episode_to_facts,
        response_top_k=response_top_k,
        config=config,
        use_lr=True,
        bm25_scores=bm25_scores,
        lr_coefs=lr_coefs,
    )


def _expand_heap(  # noqa: C901  (heap convergence loop — splitting hurts readability)
    fused_results: list[Candidate],
    episode_scores: dict[str, float],
    prefetched_facts: dict[str, list[FactCandidate]],
    *,
    response_top_k: int,
    config: RankConfig | None = None,
    use_lr: bool = False,
    bm25_scores: dict[str, float] | None = None,
    lr_coefs: LRCoefs | None = None,
) -> tuple[list[Candidate], list[FactCandidate], dict[str, Any]]:
    """Phase 2-4 heap convergence loop — inner core of ``expand``.

    Exposed separately so unit tests can pass bespoke ``fused_results`` / ``episode_scores`` without going
    through Phase 1. Returns ``(episodes, facts, metadata)``.
    """
    if config is None:
        from everalgo.rank import DEFAULT_RANK_CONFIG

        config = DEFAULT_RANK_CONFIG

    alpha = config.alpha
    facts_per_episode = config.expand_limit
    max_conv_rounds = config.max_convergence_rounds
    bm25_map = bm25_scores or {}

    heap: list[tuple[float, str]] = []
    doc_map: dict[str, Candidate] = {}

    for doc in fused_results:
        if not doc.id:
            continue
        heapq.heappush(heap, (-doc.score, doc.id))
        doc_map[doc.id] = doc

    topn: dict[str, _TopNEntry] = {}

    for doc in fused_results[:response_top_k]:
        if doc.id:
            score = episode_scores.get(doc.id, 0.0)
            topn[doc.id] = (doc, score, "episode", doc.id)

    prev_topn_keys = frozenset(topn.keys())
    convergence_count = 0
    expansions = 0
    total_iterations = 0

    logger.debug("Phase 2: heap=%d, initial top-N=%d", len(heap), len(topn))

    while heap and convergence_count < max_conv_rounds:
        total_iterations += 1
        _neg_fused, episode_id = heapq.heappop(heap)

        if episode_id not in doc_map:
            continue

        _expand_one_episode(
            episode_id=episode_id,
            topn=topn,
            prefetched_facts=prefetched_facts,
            episode_scores=episode_scores,
            bm25_map=bm25_map,
            alpha=alpha,
            facts_per_episode=facts_per_episode,
            response_top_k=response_top_k,
            use_lr=use_lr,
            lr_coefs=lr_coefs,
        )
        expansions += 1

        current_keys = frozenset(topn.keys())
        if current_keys == prev_topn_keys:
            convergence_count += 1
        else:
            convergence_count = 0
            prev_topn_keys = current_keys

    stop_reason = "convergence" if convergence_count >= max_conv_rounds else "heap_exhausted"
    logger.debug(
        "Phase 3: iterations=%d expansions=%d convergence=%d/%d stop=%s",
        total_iterations,
        expansions,
        convergence_count,
        max_conv_rounds,
        stop_reason,
    )

    sorted_entries = sorted(topn.values(), key=lambda v: v[1], reverse=True)

    episodes_out: list[Candidate] = []
    facts_out: list[FactCandidate] = []

    for entry_doc, score, item_type, _src in sorted_entries:
        if item_type == "episode":
            assert isinstance(entry_doc, Candidate)
            episodes_out.append(entry_doc.model_copy(update={"score": score}))
        else:
            assert isinstance(entry_doc, FactCandidate)
            facts_out.append(entry_doc)

    metadata: dict[str, Any] = {
        "total_iterations": total_iterations,
        "expansions": expansions,
        "convergence_rounds": convergence_count,
        "stop_reason": stop_reason,
        "episodes_in_topn": len(episodes_out),
        "facts_in_topn": len(facts_out),
    }

    return episodes_out, facts_out, metadata


class AgenticConfig(BaseModel):
    """Tunable knobs for ``aagentic_rank``."""

    model_config = ConfigDict(frozen=True)

    # Round 1
    round1_top_n: int = 20
    round1_rerank_top_n: int = 10

    # LLM judgment
    llm_temperature_sufficiency: float = 0.0
    llm_temperature_multi_query: float = 0.4
    llm_max_tokens_sufficiency: int = 500
    llm_max_tokens_multi_query: int = 300

    # Round 2
    enable_multi_query: bool = True
    num_queries: int = 3
    round2_per_query_top_n: int = 50

    # Final merge / rerank
    combined_total: int = 40

    # Sufficiency-check formatting
    sufficiency_max_docs: int = 10


DEFAULT_AGENTIC_CONFIG = AgenticConfig()


type RetrieveFn = Callable[[str, int], Awaitable[Sequence[Candidate]]]
"""``retrieve(query, top_n) -> Sequence[Candidate]`` — Round-2 recall callback."""


type RerankFn = Callable[[str, Sequence[Candidate], int], Awaitable[Sequence[Candidate]]]
"""``rerank(query, candidates, top_n) -> Sequence[Candidate]`` — cross-encoder rerank callback."""


async def aagentic_rank(
    query: str,
    sparse: Sequence[Candidate],
    dense: Sequence[Candidate],
    *,
    rerank: RerankFn,
    retrieve: RetrieveFn | None = None,
    top_k: int,
    llm: LLMClient,
    sufficiency_prompt: str = AGENTIC_SUFFICIENCY_CHECK_PROMPT_EN,
    multi_query_prompt: str = AGENTIC_MULTI_QUERY_PROMPT_EN,
    config: AgenticConfig = DEFAULT_AGENTIC_CONFIG,
) -> list[Candidate]:
    """LLM-guided multi-round agentic retrieval.

    Round 1 fuses sparse + dense; the LLM checks sufficiency; if insufficient and ``retrieve`` is provided,
    Round 2 runs multi-query recall and a final rerank. ``top_k=-1`` returns up to ``combined_total``.
    """
    is_unlimited = top_k == -1
    cfg = config

    # ========== Round 1: concat + dedup ==========
    seen_round1: set[str] = set()
    round1: list[Candidate] = []
    for c in list(sparse) + list(dense):
        if c.id and c.id not in seen_round1:
            seen_round1.add(c.id)
            round1.append(c)
    logger.info("agentic round 1: %d candidates (sparse+dense merged)", len(round1))
    if not round1:
        return []

    # ========== Round 1 rerank ==========
    rerank_n = cfg.round1_rerank_top_n if is_unlimited else max(cfg.round1_rerank_top_n, top_k)
    reranked = list(await rerank(query, round1, rerank_n))
    topn_for_llm = reranked[: cfg.round1_rerank_top_n]

    # ========== LLM sufficiency check ==========
    is_sufficient, _, missing_info = await _acheck_sufficiency(
        query=query,
        candidates=topn_for_llm,
        llm=llm,
        prompt=sufficiency_prompt,
        max_docs=cfg.round1_rerank_top_n,
        max_tokens=cfg.llm_max_tokens_sufficiency,
        temperature=cfg.llm_temperature_sufficiency,
    )
    logger.info("agentic sufficiency: %s", is_sufficient)

    if is_sufficient or retrieve is None or not cfg.enable_multi_query:
        return reranked if is_unlimited else reranked[:top_k]

    # ========== Round 2: multi-query generation ==========
    refined_queries, _ = await _agen_multi_queries(
        original_query=query,
        candidates=topn_for_llm,
        missing_info=missing_info,
        llm=llm,
        prompt=multi_query_prompt,
        max_docs=cfg.round1_rerank_top_n,
        num_queries=cfg.num_queries,
        max_tokens=cfg.llm_max_tokens_multi_query,
        temperature=cfg.llm_temperature_multi_query,
    )
    logger.info("agentic generated %d follow-up queries", len(refined_queries))

    # ========== Round 2: parallel hybrid search per query ==========
    round2_results = await asyncio.gather(
        *[retrieve(q, cfg.round2_per_query_top_n) for q in refined_queries],
    )
    all_round2: list[Candidate] = [c for r in round2_results for c in r]

    # ========== Dedup + merge ==========
    round2_unique = [c for c in all_round2 if c.id and c.id not in seen_round1]
    budget = max(cfg.combined_total - len(round1), 0)
    combined = round1 + round2_unique[:budget]
    logger.info("agentic combined: %d candidates", len(combined))

    # ========== Final rerank ==========
    final_rerank_n = cfg.combined_total if is_unlimited else max(cfg.combined_total, top_k)
    final = list(await rerank(query, combined, final_rerank_n))

    return final if is_unlimited else final[:top_k]


def _format_candidates_for_llm(candidates: Sequence[Candidate], max_docs: int) -> str:
    """Render candidates as a numbered text block for LLM context."""
    if not candidates:
        return "No retrieval results"

    lines: list[str] = []
    for i, cand in enumerate(candidates[:max_docs], start=1):
        meta = cand.metadata
        timestamp = meta.get("timestamp", "N/A")
        content = meta.get("episode") or meta.get("summary") or meta.get("subject") or "N/A"
        lines.append(f"[Memory {i}]\nTime: {timestamp}\nContent: {content}\nRelevance score: {cand.score:.4f}\n")
    return "\n".join(lines)


def _format_sufficiency_response(response: SufficiencyCheckResponse) -> tuple[bool, str, list[str]]:
    """Convert parsed sufficiency-check response to tuple format."""
    return response.is_sufficient, response.reasoning, response.missing_information


def _format_multi_query_response(response: MultiQueryResponse, original_query: str) -> tuple[list[str], str]:
    """Filter and format multi-query response, ensuring queries are valid and distinct from original."""
    original_norm = original_query.lower().strip()
    valid = [
        q.strip()
        for q in response.queries
        if isinstance(q, str) and 5 <= len(q) <= 300 and q.lower().strip() != original_norm
    ]
    if not valid:
        raise ValueError("multi-query: all generated queries were filtered (too short/long or identical to original)")
    return valid[:3], response.reasoning


async def _acheck_sufficiency(
    *,
    query: str,
    candidates: Sequence[Candidate],
    llm: LLMClient,
    prompt: str,
    max_docs: int,
    max_tokens: int,
    temperature: float,
) -> tuple[bool, str, list[str]]:
    """LLM call: are these candidates enough to answer the query?

    Raises:
        LLMError: Propagated from the underlying LLM client call.
        ValueError: If the response cannot be parsed or is missing required fields.
        TypeError: If the response fields have unexpected types.
    """
    client = llm
    rendered = prompt.format(query=query, retrieved_docs=_format_candidates_for_llm(candidates, max_docs))
    response = await client.chat(
        messages=[ChatMessage(role="user", content=rendered)],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=SufficiencyCheckResponse,
    )
    parsed = cast("SufficiencyCheckResponse | None", response.parsed)
    if parsed is None:
        raise ValueError("LLM returned no parsed structured output")
    return _format_sufficiency_response(parsed)


async def _agen_multi_queries(
    *,
    original_query: str,
    candidates: Sequence[Candidate],
    missing_info: list[str],
    llm: LLMClient,
    prompt: str,
    max_docs: int,
    num_queries: int,
    max_tokens: int,
    temperature: float,
) -> tuple[list[str], str]:
    """LLM call: produce ``num_queries`` complementary queries focused on ``missing_info``.

    Raises:
        LLMError: Propagated from the underlying LLM client call.
        ValueError: If the response cannot be parsed or all generated queries are filtered.
        TypeError: If the response fields have unexpected types.
    """
    _ = num_queries
    client = llm
    rendered = prompt.format(
        original_query=original_query,
        retrieved_docs=_format_candidates_for_llm(candidates, max_docs),
        missing_info=", ".join(missing_info) if missing_info else "N/A",
    )
    response = await client.chat(
        messages=[ChatMessage(role="user", content=rendered)],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=MultiQueryResponse,
    )
    parsed = cast("MultiQueryResponse | None", response.parsed)
    if parsed is None:
        raise ValueError("LLM returned no parsed structured output")
    return _format_multi_query_response(parsed, original_query)
