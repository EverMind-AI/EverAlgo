"""Tests for ``acluster_retrieve``."""

from __future__ import annotations

import numpy as np

from everalgo.clustering import Cluster
from everalgo.retrieval.cluster import acluster_retrieve
from everalgo.types.rank import Candidate


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


async def test_acluster_picks_top_k_clusters_by_max_member_score():
    """Cluster score = max of its members' scores in the base_retrieve result."""
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

    # cluster_top_k=2 picks c1 (members c,d) + c0 (members a,b) -> 4 docs expanded;
    # returned in ``all_docs`` order, unranked.
    assert {c.id for c in result} == {"a", "b", "c", "d"}
    assert [c.id for c in result] == ["a", "b", "c", "d"]


async def test_acluster_returns_full_expansion_unranked():
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


async def test_acluster_empty_base_returns_empty():
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


async def test_acluster_ignores_docs_outside_selected_clusters():
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
