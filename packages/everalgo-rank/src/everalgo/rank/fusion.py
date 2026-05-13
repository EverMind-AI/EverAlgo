"""Fusion algorithms — RRF / LR / cosine→LR / propagation / hierarchical expand / agentic LLM-guided rank."""

from __future__ import annotations

import asyncio
import heapq
import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

import everalgo.llm
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
]

logger = logging.getLogger(__name__)


def rrf(*sources: Sequence[Candidate], k: int = 60) -> list[Candidate]:
    """Reciprocal Rank Fusion — merge N ranked recall lists.

    Score formula::

        rrf_score(doc) = sum(1 / (k + rank_i)) for each list i containing doc

    Args:
        *sources: ``N`` ranked Candidate sequences, each already sorted by its
            own relevance score (descending).
        k: RRF smoothing constant. Default ``60`` matches the original Cormack
            et al. 2009 paper and ``enterprise`` production.

    Returns
    -------
        A new list of Candidates with ``.score`` set to the accumulated RRF
        score, sorted descending. Empty input → empty list.
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

        logit = emb_score * emb_coef + bm25_score * bm25_coef + intercept
        prob = 1 / (1 + exp(-logit))

    Args:
        emb_results: Embedding (cosine) recall results.
        bm25_results: BM25 recall results.
        coefs: Trained LR coefficients. ``None`` (default) falls through to
            ``weight.default_lr_coefs()`` inside ``weight.multi_field_weighting``.
            Monkey-patching that function shifts the default for every caller
            without re-passing ``coefs``.

    Returns
    -------
        Candidates sorted descending by LR probability, with ``.score``
        replaced by the probability.
    """
    return _weight.multi_field_weighting(
        {"emb": list(emb_results), "bm25": list(bm25_results)},
        coefs=coefs,
    )


def cosine_to_lr_score(
    sim: float,
    parent_bm25: float = 0.0,
    *,
    coefs: LRCoefs | None = None,
) -> float:
    """Calibrate a raw cosine similarity to an LR probability.

    Args:
        sim: Cosine similarity in ``[-1, 1]`` (typically ``[0, 1]``).
        parent_bm25: Parent's BM25 score. Default ``0.0`` if unknown.
        coefs: Trained LR coefficients; ``None`` → ``weight.default_lr_coefs()``
            (resolved inside ``weight.multi_field_weighting``).

    Returns
    -------
        Calibrated probability in ``[0, 1]``.
    """
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
    """Hierarchical score propagation: blend child + parent into a final score.

    ``final_score = alpha * child_score + (1 - alpha) * parent_score``

    Args:
        parents: Parent candidates (e.g. Episodes).
        children: Child candidates (e.g. AtomicFacts), each with
            ``metadata['parent_id']`` pointing at a parent ``id``.
        alpha: Child weight in ``[0, 1]``. Default ``1.0`` reproduces enterprise
            behaviour (use child only).
        parent_score_lookup: Optional ``{parent_id: score}`` override. If a
            child's parent is not in the lookup (and ``parent_score_lookup`` is
            not ``None``), the parent contribution is ``0``.

    Returns
    -------
        New list of children with ``.score`` replaced by ``final_score``;
        children whose parent cannot be resolved still appear (parent
        contribution treated as ``0``). Caller decides whether to sort.
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
    """Full enterprise MRAG pipeline — Phase 1 fusion + Phase 2-4 expansion.

    ``expand`` is the high-level entry the episodic facade calls when
    ``fusion_mode == "mrag"``. It owns:

    - **Phase 1**: RRF over (dense, sparse) for heap ordering, AND LR fusion
      over (dense, sparse) for propagation scores. Equivalent to enterprise's
      ``rrf_cosine`` mode (RRF heap + LR-calibrated propagation).
    - **Phase 2-4**: hierarchical heap collapse via the private
      ``_expand_heap`` core — facts compete with their parent episode; when a
      fact climbs into top-N, the parent is evicted. Stop on
      ``max_convergence_rounds`` consecutive iterations with no top-N change.

    For unit tests that need fine-grained control over the heap inputs (e.g.
    bespoke ``fused_results`` / ``episode_scores``), call the private
    ``_expand_heap`` directly.

    Args:
        sparse: BM25 / keyword recall results (descending).
        dense: Vector recall results (descending).
        episode_to_facts: ``{episode_id: [FactCandidate, ...]}`` pre-fetched
            by EverOS Recall, sorted descending by cosine score within each
            episode.
        response_top_k: Maximum items in the final top-N set.
        config: ``alpha`` / ``rrf_k`` / ``expand_limit`` /
            ``max_convergence_rounds`` knobs; ``None`` uses
            ``DEFAULT_RANK_CONFIG``.
        lr_coefs: Override for the trained LR coefficients; ``None`` defers to
            ``weight.default_lr_coefs()``.

    Returns
    -------
        ``(episodes, facts, metadata)``:

        - ``episodes`` — episodes still in top-N (not replaced by their facts).
        - ``facts`` — facts that climbed into top-N.
        - ``metadata`` — diagnostic counters.
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


def _expand_heap(  # noqa: C901  (heap convergence loop ported 1:1 from enterprise mrag_expander)
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
    """Phase 2-4 heap convergence loop — core of ``expand``.

    Private because the high-level ``expand`` is the public surface; this
    helper is exposed only for fine-grained heap-mechanics unit tests that
    need to construct ``fused_results`` / ``episode_scores`` directly.

    Args:
        fused_results: Heap-ordering list (descending). Drives the priority queue.
        episode_scores: ``{episode_id: score}`` used for score propagation.
        prefetched_facts: ``{episode_id: [FactCandidate, ...]}`` grouped by
            parent and sorted descending by cosine score.
        response_top_k: Maximum items in the final top-N set.
        config: ``alpha`` / ``expand_limit`` / ``max_convergence_rounds`` knobs;
            ``None`` uses ``DEFAULT_RANK_CONFIG``.
        use_lr: If ``True``, convert each child fact's cosine to an LR
            probability before propagation so child and parent live on the
            same scale.
        bm25_scores: ``{episode_id: bm25}`` from Phase 1 BM25. Only consumed
            when ``use_lr=True``.
        lr_coefs: Override for ``cosine_to_lr_score``; ``None`` defers to
            ``weight.default_lr_coefs()``.

    Returns
    -------
        Same shape as ``expand``: ``(episodes, facts, metadata)``.
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
    """Tunable knobs for ``aagentic_rank`` — ported 1:1 from opensource ``AgenticConfig``."""

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
    llm: LLMClient | None = None,
    sufficiency_prompt: str = AGENTIC_SUFFICIENCY_CHECK_PROMPT_EN,
    multi_query_prompt: str = AGENTIC_MULTI_QUERY_PROMPT_EN,
    config: AgenticConfig = DEFAULT_AGENTIC_CONFIG,
) -> list[Candidate]:
    """LLM-guided multi-round agentic retrieval — opensource ``retrieve_mem_agentic`` ported.

    Parameters
    ----------
    query
        Original user query.
    sparse
        BM25 / keyword recall results for ``query`` (Round 1 input).
    dense
        Vector recall results for ``query`` (Round 1 input).
    rerank
        Cross-encoder rerank callback (required for Round 1 + final).
    retrieve
        Round-2 recall callback for newly generated queries; ``None`` skips Round 2.
    top_k
        Final result count; ``-1`` for unlimited (returns up to ``combined_total``).
    llm
        Per-call LLM override; ``None`` falls through ``everalgo.llm.resolve``.
    sufficiency_prompt
        Override for sufficiency-check prompt. Must contain ``{query}`` / ``{retrieved_docs}``.
    multi_query_prompt
        Override for multi-query-generation prompt. Must contain ``{original_query}`` /
        ``{retrieved_docs}`` / ``{missing_info}``.
    config
        ``AgenticConfig`` knobs (round sizes, num_queries, …).
    """
    is_unlimited = top_k == -1
    cfg = config

    try:
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
            return_exceptions=True,
        )
        all_round2: list[Candidate] = [c for r in round2_results if not isinstance(r, BaseException) for c in r]

        # ========== Dedup + merge ==========
        round2_unique = [c for c in all_round2 if c.id and c.id not in seen_round1]
        budget = max(cfg.combined_total - len(round1), 0)
        combined = round1 + round2_unique[:budget]
        logger.info("agentic combined: %d candidates", len(combined))

        # ========== Final rerank ==========
        final_rerank_n = cfg.combined_total if is_unlimited else max(cfg.combined_total, top_k)
        final = list(await rerank(query, combined, final_rerank_n))

        return final if is_unlimited else final[:top_k]

    except Exception:
        logger.exception("aagentic_rank failed")
        return []


def _format_candidates_for_llm(candidates: Sequence[Candidate], max_docs: int) -> str:
    """Render candidates as a numbered text block — ported from opensource ``format_documents_for_llm``."""
    if not candidates:
        return "No retrieval results"

    lines: list[str] = []
    for i, cand in enumerate(candidates[:max_docs], start=1):
        meta = cand.metadata
        timestamp = meta.get("timestamp", "N/A")
        content = meta.get("episode") or meta.get("summary") or meta.get("subject") or "N/A"
        lines.append(f"[Memory {i}]\nTime: {timestamp}\nContent: {content}\nRelevance score: {cand.score:.4f}\n")
    return "\n".join(lines)


def _parse_json_response(response: str) -> dict[str, Any]:
    """Extract a JSON object from LLM output (may have leading/trailing prose)."""
    start = response.find("{")
    end = response.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in response")
    payload = json.loads(response[start:end])
    if not isinstance(payload, dict):
        raise TypeError("Top-level JSON is not an object")
    return payload


def _parse_sufficiency_response(response: str) -> tuple[bool, str, list[str]]:
    """Parse sufficiency-check JSON; on failure return ``(True, "<error>", [])``."""
    try:
        payload = _parse_json_response(response)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("sufficiency parse failed: %s", exc)
        return True, f"Parse error: {exc}", []
    if "is_sufficient" not in payload:
        logger.warning("sufficiency payload missing 'is_sufficient' field")
        return True, "Parse error: missing 'is_sufficient'", []
    is_sufficient = bool(payload["is_sufficient"])
    reasoning = payload.get("reasoning", "No reasoning provided")
    missing = payload.get("missing_information", [])
    if not isinstance(missing, list):
        missing = []
    return is_sufficient, reasoning, missing


def _parse_multi_query_response(response: str, original_query: str) -> tuple[list[str], str]:
    """Return ``(queries, reasoning)``; falls back to ``[original_query]`` on any failure."""
    try:
        payload = _parse_json_response(response)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("multi-query parse failed: %s", exc)
        return [original_query], f"Parse error: {exc}"
    queries = payload.get("queries")
    if not isinstance(queries, list):
        logger.warning("multi-query payload missing or invalid 'queries' field")
        return [original_query], "Parse error: missing 'queries'"
    reasoning = payload.get("reasoning", "No reasoning provided")
    original_norm = original_query.lower().strip()
    valid = [
        q.strip() for q in queries if isinstance(q, str) and 5 <= len(q) <= 300 and q.lower().strip() != original_norm
    ]
    if not valid:
        return [original_query], "Fallback: used original query"
    return valid[:3], reasoning


async def _acheck_sufficiency(
    *,
    query: str,
    candidates: Sequence[Candidate],
    llm: LLMClient | None,
    prompt: str,
    max_docs: int,
    max_tokens: int,
    temperature: float,
) -> tuple[bool, str, list[str]]:
    """LLM call: are these candidates enough to answer the query? Returns opensource-compatible tuple."""
    client = everalgo.llm.resolve(llm)
    rendered = prompt.format(query=query, retrieved_docs=_format_candidates_for_llm(candidates, max_docs))
    try:
        response = await client.chat(
            messages=[ChatMessage(role="user", content=rendered)],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.exception("sufficiency LLM call failed")
        return True, f"LLM error: {exc}", []
    return _parse_sufficiency_response(response.content)


async def _agen_multi_queries(
    *,
    original_query: str,
    candidates: Sequence[Candidate],
    missing_info: list[str],
    llm: LLMClient | None,
    prompt: str,
    max_docs: int,
    num_queries: int,
    max_tokens: int,
    temperature: float,
) -> tuple[list[str], str]:
    """LLM call: produce ``num_queries`` complementary queries focused on ``missing_info``."""
    _ = num_queries
    client = everalgo.llm.resolve(llm)
    rendered = prompt.format(
        original_query=original_query,
        retrieved_docs=_format_candidates_for_llm(candidates, max_docs),
        missing_info=", ".join(missing_info) if missing_info else "N/A",
    )
    try:
        response = await client.chat(
            messages=[ChatMessage(role="user", content=rendered)],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.exception("multi-query LLM call failed")
        return [original_query], f"LLM error: {exc}"
    return _parse_multi_query_response(response.content, original_query)
