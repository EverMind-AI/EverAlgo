"""Cluster-scoped retriever composer for ``aagentic_retrieve``.

``acluster_retrieve`` scans the base retrieval results in score order and accumulates
distinct clusters until ``cluster_top_k`` are reached (first-hit strategy), then
expands those clusters' memberships from a caller-provided ``all_docs`` list and
returns the expansion unreranked. Rerank is intentionally left to the caller
(typically ``aagentic_retrieve`` rerunning the same ``rerank_fn`` over Round 1 and
again over the merged Round 1 + Round 2 set), which avoids the double-rerank trap and
keeps the Round 2 final rerank effective.

The expansion grows with cluster size, not with any ``k``: one large cluster can put
thousands of members in front of the caller's rerank. ``max_members`` bounds it with a
ranked cut (seed hits first, then the caller's in-scope ranking), so a corpus of any
size reaches rerank as at most ``max_members`` candidates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync

from everalgo.utils import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.clustering import Cluster
    from everalgo.rank.protocols import MemberRetrieveFn, RetrieveFn
    from everalgo.types.rank import Candidate

__all__ = ["acluster_retrieve", "cluster_retrieve"]


async def acluster_retrieve(
    query: str,
    *,
    base_retrieve: RetrieveFn,
    base_candidates: int | None = None,
    clusters: Sequence[Cluster],
    all_docs: Sequence[Candidate],
    cluster_top_k: int = 5,
    max_members: int | None = None,
    member_retrieve: MemberRetrieveFn | None = None,
) -> list[Candidate]:
    """Narrow full-corpus retrieve to top-K cluster scope and return the expansion.

    Algorithm:
        1. ``base_retrieve(query, base_candidates)`` — wide candidate pool for cluster
           selection (Level 1). Caller typically passes a hybrid (dense+sparse RRF)
           closure.
        2. Scan results in score order, accumulating distinct clusters (first-hit): the
           first member of a cluster seen determines that cluster's inclusion. Stop as
           soon as ``cluster_top_k`` distinct clusters are reached.
        3. Expand the selected clusters' memberships from ``all_docs`` (id lookup).
           Without ``max_members`` the whole expansion is returned in ``all_docs`` order.
           With it, the expansion is ordered seed hits first (in base order), then the
           remaining members; once it exceeds ``max_members``, the remaining members are
           ranked by ``member_retrieve`` and only the top ones fill the budget.

    Args:
        query: Search query.
        base_retrieve: Async ``(query, k) -> list[Candidate]`` used to seed cluster
            selection. Caller typically passes a hybrid (dense+sparse RRF) closure.
        base_candidates: Top-k passed to ``base_retrieve``. ``None`` means no
            truncation (passes ``len(all_docs)`` — the corpus upper bound). When set,
            typically 50-100; must be wide enough to cover ≥ ``cluster_top_k`` distinct
            clusters.
        clusters: All clusters in the corpus. Each cluster contributes its members at
            most once; the algorithm reads ``c.id`` and ``c.members``.
        all_docs: Corpus snapshot as ``Candidate`` list. Used to expand selected
            clusters' members — items not in any selected cluster are dropped, and a
            member absent from ``all_docs`` is never returned.
        cluster_top_k: How many top-scoring clusters to keep.
        max_members: Upper bound on the returned expansion. ``None`` returns the whole
            expansion, unbounded. Requires ``member_retrieve``.
        member_retrieve: Async ``(query, member_ids, k) -> list[Candidate]`` ranking the
            given members against the query (e.g. a vector search filtered to those ids).
            Called only when the expansion exceeds ``max_members``; ids it returns outside
            ``member_ids`` are ignored, and members it leaves out are not returned.

    Returns:
        Members of the top-``cluster_top_k`` clusters, unscored (each candidate is the
        ``all_docs`` entry and carries its ``score`` — caller should not rely on it).
        Empty list when no base hit maps to any cluster.

    Raises:
        InvalidInputError: ``max_members`` is below 1, or is set without
            ``member_retrieve``.

    Resource contract:
        - ``base_retrieve`` is called exactly once.
        - ``member_retrieve`` is called at most once, only when the expansion exceeds
          ``max_members``.
        - No rerank calls.
        - No LLM calls.
    """
    _check_budget(max_members, member_retrieve)
    base_results = await base_retrieve(query, base_candidates if base_candidates is not None else len(all_docs))
    selected = _select_clusters(base_results, clusters, cluster_top_k)
    if not selected:
        return []
    expansion = _expand(selected, clusters, all_docs)
    if max_members is None or member_retrieve is None:
        return expansion
    return await _ranked_cut(query, base_results, expansion, max_members, member_retrieve)


cluster_retrieve = async_to_sync(acluster_retrieve)
"""Sync bridge for non-event-loop contexts (CLI / pytest). Per ADR 010."""


def _check_budget(max_members: int | None, member_retrieve: MemberRetrieveFn | None) -> None:
    if max_members is None:
        return
    if max_members < 1:
        raise InvalidInputError(f"max_members must be >= 1, got {max_members}")
    if member_retrieve is None:
        raise InvalidInputError("max_members needs member_retrieve to rank the members it cuts")


def _select_clusters(base_results: Sequence[Candidate], clusters: Sequence[Cluster], cluster_top_k: int) -> set[str]:
    """First-hit: the first ``cluster_top_k`` distinct clusters in base-result order."""
    member_to_cluster: dict[str, str] = {
        member: cluster.id for cluster in clusters if cluster.id is not None for member in cluster.members
    }
    selected: set[str] = set()
    for cand in base_results:
        cluster_id = member_to_cluster.get(cand.id)
        if cluster_id is None:
            continue
        selected.add(cluster_id)
        if len(selected) >= cluster_top_k:
            break
    return selected


def _expand(selected: set[str], clusters: Sequence[Cluster], all_docs: Sequence[Candidate]) -> list[Candidate]:
    """Selected clusters' members present in ``all_docs``, in ``all_docs`` order."""
    member_ids = {member for cluster in clusters if cluster.id in selected for member in cluster.members}
    return [d for d in all_docs if d.id in member_ids]


async def _ranked_cut(
    query: str,
    base_results: Sequence[Candidate],
    expansion: list[Candidate],
    max_members: int,
    member_retrieve: MemberRetrieveFn,
) -> list[Candidate]:
    """Seed hits first, then the rest ranked by ``member_retrieve``, at most ``max_members``.

    Seed hits lead because the base pool already ranked them against the whole corpus; the rest
    only reach rerank through the in-scope ranking, never in ``all_docs`` order, whose order
    carries no relevance.
    """
    by_id = {d.id: d for d in expansion}
    seed = list(dict.fromkeys(c.id for c in base_results if c.id in by_id))
    seed_ids = set(seed)
    rest = [d.id for d in expansion if d.id not in seed_ids]
    if len(seed) + len(rest) <= max_members:
        return [by_id[i] for i in seed + rest]
    room = max_members - len(seed)
    if room <= 0:
        return [by_id[i] for i in seed[:max_members]]
    rest_ids = set(rest)
    ranked = await member_retrieve(query, rest, room)
    picked = list(dict.fromkeys(c.id for c in ranked if c.id in rest_ids))[:room]
    return [by_id[i] for i in seed + picked]
