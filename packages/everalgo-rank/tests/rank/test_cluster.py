"""Tests for ``acluster_retrieve``."""

from __future__ import annotations

import numpy as np
import pytest

from everalgo.clustering import Cluster
from everalgo.rank.cluster import acluster_retrieve
from everalgo.types.rank import Candidate
from everalgo.utils import InvalidInputError


def _doc(cid: str, score: float = 1.0) -> Candidate:
    return Candidate(id=cid, score=score, metadata={"episode": {"subject": cid, "content": cid}})


def _cluster(cid: str, members: list[str]) -> Cluster:
    return Cluster(
        id=cid,
        centroid=np.array([0.0], dtype=np.float32),
        count=len(members),
        last_ts=0,
        members=members,
    )


async def test_acluster_picks_first_k_clusters_by_scan_order() -> None:
    """First-hit: clusters are selected in the order their first member appears in base results."""
    clusters = [
        _cluster("c0", ["a", "b"]),
        _cluster("c1", ["c", "d"]),
        _cluster("c2", ["e", "f"]),
    ]
    all_docs = [_doc(x) for x in ["a", "b", "c", "d", "e", "f"]]

    async def base(q: str, k: int) -> list[Candidate]:
        # c1.max = max(0.95, 0.4) = 0.95 -> highest
        # c0.max = max(0.85, 0.2) = 0.85
        # c2.max = max(0.3, 0.1) = 0.3 -> lowest
        return [
            Candidate(id="c", score=0.95, metadata=all_docs[2].metadata),
            Candidate(id="a", score=0.85, metadata=all_docs[0].metadata),
            Candidate(id="d", score=0.4, metadata=all_docs[3].metadata),
            Candidate(id="b", score=0.2, metadata=all_docs[1].metadata),
            Candidate(id="e", score=0.3, metadata=all_docs[4].metadata),
            Candidate(id="f", score=0.1, metadata=all_docs[5].metadata),
        ]

    result = await acluster_retrieve(
        "q",
        base_retrieve=base,
        base_candidates=6,
        clusters=clusters,
        all_docs=all_docs,
        cluster_top_k=2,
    )

    # cluster_top_k=2 picks c1 + c0 -> 4 docs expanded; returned in all_docs order.
    assert {c.id for c in result} == {"a", "b", "c", "d"}
    assert [c.id for c in result] == ["a", "b", "c", "d"]


async def test_acluster_returns_full_expansion_unranked() -> None:
    """No rerank, no truncation — caller (aagentic) handles both."""
    clusters = [_cluster("c0", ["a", "b", "c"])]
    all_docs = [_doc(x) for x in ["a", "b", "c"]]

    async def base(q: str, k: int) -> list[Candidate]:
        return [_doc("a", 1.0)]

    result = await acluster_retrieve(
        "q",
        base_retrieve=base,
        base_candidates=5,
        clusters=clusters,
        all_docs=all_docs,
        cluster_top_k=1,
    )
    # All 3 members of c0 returned, in ``all_docs`` order.
    assert [c.id for c in result] == ["a", "b", "c"]


async def test_acluster_empty_base_returns_empty() -> None:
    clusters = [_cluster("c0", ["a"])]
    all_docs = [_doc("a")]

    async def base(q: str, k: int) -> list[Candidate]:
        return []

    result = await acluster_retrieve(
        "q",
        base_retrieve=base,
        base_candidates=10,
        clusters=clusters,
        all_docs=all_docs,
        cluster_top_k=2,
    )
    assert result == []


async def test_acluster_first_hit_stops_early() -> None:
    """First-hit skips a cluster whose highest-scoring member appears late in the result list."""
    clusters = [
        _cluster("c0", ["a", "b"]),
        _cluster("c1", ["c", "d"]),
        _cluster("c2", ["e"]),
    ]
    all_docs = [_doc(x) for x in ["a", "b", "c", "d", "e"]]

    async def base(q: str, k: int) -> list[Candidate]:
        # Scan order: c0 hit first (a@0.9), c1 hit second (c@0.8) -> top_k=2 reached, stop.
        # c2's member e has the highest score (0.99) but appears after the scan stops.
        return [
            Candidate(id="a", score=0.9, metadata=all_docs[0].metadata),
            Candidate(id="c", score=0.8, metadata=all_docs[2].metadata),
            Candidate(id="e", score=0.99, metadata=all_docs[4].metadata),
            Candidate(id="b", score=0.5, metadata=all_docs[1].metadata),
        ]

    result = await acluster_retrieve(
        "q",
        base_retrieve=base,
        base_candidates=10,
        clusters=clusters,
        all_docs=all_docs,
        cluster_top_k=2,
    )
    # c2 (member e, score 0.99) would win under MaxSim top-K but is excluded by first-hit.
    assert {c.id for c in result} == {"a", "b", "c", "d"}


async def test_acluster_base_candidates_none_passes_corpus_size() -> None:
    """``base_candidates=None`` passes ``len(all_docs)`` to base_retrieve (no truncation)."""
    clusters = [_cluster("c0", ["a", "b"])]
    all_docs = [_doc("a"), _doc("b")]
    received_k: list[int] = []

    async def base(q: str, k: int) -> list[Candidate]:
        received_k.append(k)
        return [_doc("a", 1.0)]

    await acluster_retrieve(
        "q",
        base_retrieve=base,
        clusters=clusters,
        all_docs=all_docs,
        cluster_top_k=1,
    )
    assert received_k == [2]  # len(all_docs)


async def test_acluster_ignores_docs_outside_selected_clusters() -> None:
    """Even if base returns a doc, if it belongs to no selected cluster it must be dropped."""
    clusters = [_cluster("c0", ["a"]), _cluster("c1", ["b"])]
    all_docs = [_doc("a"), _doc("b"), _doc("orphan")]

    async def base(q: str, k: int) -> list[Candidate]:
        # "orphan" hits with high score but maps to no cluster -> ignored
        return [_doc("orphan", 99.0), _doc("a", 1.0)]

    result = await acluster_retrieve(
        "q",
        base_retrieve=base,
        base_candidates=10,
        clusters=clusters,
        all_docs=all_docs,
        cluster_top_k=1,
    )
    assert {c.id for c in result} == {"a"}  # only c0 selected; orphan dropped


# ── max_members: ranked cut of the expansion ─────────────────────────────────


class _MemberRanker:
    """Ranks members by a fixed preference order and records every call.

    Args:
        preference: Member ids, best first; members not listed are never returned.
    """

    def __init__(self, preference: list[str]) -> None:
        self.preference = preference
        self.calls: list[tuple[list[str], int]] = []

    async def __call__(self, q: str, member_ids: list[str], k: int) -> list[Candidate]:
        self.calls.append((list(member_ids), k))
        return [_doc(i) for i in self.preference if i in member_ids][:k]


def _big_cluster_setup() -> tuple[list[Cluster], list[Candidate]]:
    """One cluster of six members; ``all_docs`` order (a..f) carries no relevance."""
    clusters = [_cluster("c0", ["a", "b", "c", "d", "e", "f"])]
    return clusters, [_doc(x) for x in ["a", "b", "c", "d", "e", "f"]]


async def _seed_e_c(q: str, k: int) -> list[Candidate]:
    return [_doc("e", 0.9), _doc("c", 0.8)]


async def test_acluster_budget_ranks_rest_and_keeps_seed_hits_first() -> None:
    """Over budget: seed hits lead in base order, the rest come from ``member_retrieve`` best first."""
    clusters, all_docs = _big_cluster_setup()
    ranker = _MemberRanker(["f", "a", "b", "d"])

    result = await acluster_retrieve(
        "q",
        base_retrieve=_seed_e_c,
        clusters=clusters,
        all_docs=all_docs,
        cluster_top_k=1,
        max_members=4,
        member_retrieve=ranker,
    )

    assert [c.id for c in result] == ["e", "c", "f", "a"]
    assert ranker.calls == [(["a", "b", "d", "f"], 2)]


async def test_acluster_budget_within_limit_skips_member_retrieve() -> None:
    """At or under budget nothing is cut: every member returns, seed hits first, no ranking call."""
    clusters, all_docs = _big_cluster_setup()
    ranker = _MemberRanker([])

    result = await acluster_retrieve(
        "q",
        base_retrieve=_seed_e_c,
        clusters=clusters,
        all_docs=all_docs,
        cluster_top_k=1,
        max_members=6,
        member_retrieve=ranker,
    )

    assert [c.id for c in result] == ["e", "c", "a", "b", "d", "f"]
    assert ranker.calls == []


async def test_acluster_budget_filled_by_seed_hits_alone() -> None:
    """When seed hits already fill the budget, they are cut in base order and nothing else is ranked."""
    clusters, all_docs = _big_cluster_setup()
    ranker = _MemberRanker(["a"])

    result = await acluster_retrieve(
        "q",
        base_retrieve=_seed_e_c,
        clusters=clusters,
        all_docs=all_docs,
        cluster_top_k=1,
        max_members=1,
        member_retrieve=ranker,
    )

    assert [c.id for c in result] == ["e"]
    assert ranker.calls == []


async def test_acluster_budget_ignores_ids_outside_the_cut_scope() -> None:
    """``member_retrieve`` returning a seed hit, a non-member or a duplicate cannot widen or repeat the cut."""
    clusters, all_docs = _big_cluster_setup()

    async def sloppy(q: str, member_ids: list[str], k: int) -> list[Candidate]:
        return [_doc("e"), _doc("zzz"), _doc("b"), _doc("b"), _doc("d")]

    result = await acluster_retrieve(
        "q",
        base_retrieve=_seed_e_c,
        clusters=clusters,
        all_docs=all_docs,
        cluster_top_k=1,
        max_members=4,
        member_retrieve=sloppy,
    )

    assert [c.id for c in result] == ["e", "c", "b", "d"]


async def test_acluster_budget_returns_all_docs_entries() -> None:
    """Returned candidates are the ``all_docs`` entries, whatever ``member_retrieve`` carried."""
    clusters, all_docs = _big_cluster_setup()

    async def bare(q: str, member_ids: list[str], k: int) -> list[Candidate]:
        return [Candidate(id=i, score=0.5) for i in member_ids][:k]

    result = await acluster_retrieve(
        "q",
        base_retrieve=_seed_e_c,
        clusters=clusters,
        all_docs=all_docs,
        cluster_top_k=1,
        max_members=3,
        member_retrieve=bare,
    )

    by_id = {d.id: d for d in all_docs}
    assert all(c is by_id[c.id] for c in result)


async def test_acluster_without_budget_keeps_all_docs_order() -> None:
    """``max_members=None`` keeps the unbounded expansion in ``all_docs`` order, seed hits included."""
    clusters, all_docs = _big_cluster_setup()

    result = await acluster_retrieve(
        "q",
        base_retrieve=_seed_e_c,
        clusters=clusters,
        all_docs=all_docs,
        cluster_top_k=1,
    )

    assert [c.id for c in result] == ["a", "b", "c", "d", "e", "f"]


@pytest.mark.parametrize(
    ("max_members", "member_retrieve", "message"),
    [
        (0, _MemberRanker([]), "max_members must be >= 1"),
        (3, None, "needs member_retrieve"),
    ],
)
async def test_acluster_budget_rejects_bad_input_before_retrieving(
    max_members: int, member_retrieve: _MemberRanker | None, message: str
) -> None:
    """A bad budget fails before ``base_retrieve`` runs."""
    clusters, all_docs = _big_cluster_setup()
    calls: list[int] = []

    async def base(q: str, k: int) -> list[Candidate]:
        calls.append(k)
        return []

    with pytest.raises(InvalidInputError, match=message):
        await acluster_retrieve(
            "q",
            base_retrieve=base,
            clusters=clusters,
            all_docs=all_docs,
            max_members=max_members,
            member_retrieve=member_retrieve,
        )
    assert calls == []
