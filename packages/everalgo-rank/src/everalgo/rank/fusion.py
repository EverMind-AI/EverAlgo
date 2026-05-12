"""Fusion algorithms — RRF / LR / cosine→LR / propagation / hierarchical expand."""

from __future__ import annotations

import heapq
import logging
from typing import TYPE_CHECKING, Any

from everalgo.rank import weight as _weight
from everalgo.types import Candidate, FactCandidate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.rank import RankConfig
    from everalgo.rank.weight import LRCoefs

__all__ = [
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
