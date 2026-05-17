"""End-to-end pipeline test: caller-embedded MemCells → clustering → profile.

Demonstrates the contract: ``cluster_by_geometry`` returns ``Cluster | None``;
the caller maintains an id-keyed dict ``cluster_id -> Cluster`` and a
``cluster_id -> [MemCell, ...]`` map; the prior ``MemCell``s of the assigned
cluster are prepended to the current cell and passed to ``ProfileExtractor`` as a
single chronological ``memcells`` list.

No real embedding model is used — vectors are deterministic by hand so the
geometric decisions are predictable.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from everalgo.clustering import Cluster, cluster_by_geometry
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory.profile import ProfileExtractor


def _memcell(content: str, ts_ms: int) -> MemCell:
    msg = ChatMessage(
        id="m0",
        role="user",
        content=content,
        timestamp=ts_ms,
        sender_id="u_alice",
        sender_name="Alice",
    )
    return MemCell(items=[msg], timestamp=ts_ms)


async def test_two_similar_memcells_cluster_then_profile_sees_prior() -> None:
    """Three MemCells stream through clustering; topic-similar pair shares a cluster.

    Scenario:
    - mc_python_1 (vec=[1,0,0]) — first ever, appended as clusters[0].
    - mc_python_2 (vec=[0.95,0.05,0]) — cosine ≈ 0.998 vs centroid → merged into clusters[0].
    - mc_cooking  (vec=[0,0,1])     — orthogonal → appended as clusters[1].

    Then ``mc_python_2`` is fed to ``ProfileExtractor`` with the first cell prepended as
    prior context in a single ``memcells`` list.  Asserts:
        1. Cluster index assignments match expectations.
        2. ``ProfileExtractor`` receives the prior MemCell's content in its prompt.
        3. Profile owner_id propagates from ``mc_python_2``.
    """
    base_ts = 1_700_000_000_000
    mc_python_1 = _memcell("How do async retry loops work in Python?", base_ts)
    mc_python_2 = _memcell("Followup on async timeouts in asyncio.", base_ts + 60_000)
    mc_cooking = _memcell("Any tips on sourdough hydration?", base_ts + 120_000)

    vec_python_1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_python_2 = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    vec_cooking = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    cluster_store: dict[str, Cluster] = {}  # id -> Cluster
    id_to_memcells: dict[str, list[MemCell]] = {}
    next_id = 0

    for memcell, vec, entity_id in [
        (mc_python_1, vec_python_1, "mc_001"),
        (mc_python_2, vec_python_2, "mc_002"),
        (mc_cooking, vec_cooking, "mc_003"),
    ]:
        new_c = Cluster(centroid=vec, last_ts=memcell.timestamp, members=[entity_id])
        result = await cluster_by_geometry(new_c, list(cluster_store.values()))
        if result is None:
            cid = f"cid_{next_id}"
            next_id += 1
            cluster_store[cid] = Cluster(
                id=cid,
                centroid=new_c.centroid,
                count=1,
                last_ts=new_c.last_ts,
                preview=new_c.preview,
                members=new_c.members,
            )
            id_to_memcells.setdefault(cid, []).append(memcell)
        else:
            assert result.id is not None
            cluster_store[result.id] = result
            id_to_memcells.setdefault(result.id, []).append(memcell)

    # Geometric expectations.
    assert len(cluster_store) == 2
    assert len(id_to_memcells) == 2
    py_cid = next(cid for cid, cells in id_to_memcells.items() if mc_python_1 in cells)
    cook_cid = next(cid for cid, cells in id_to_memcells.items() if mc_cooking in cells)
    assert py_cid != cook_cid
    assert id_to_memcells[py_cid] == [mc_python_1, mc_python_2]
    assert id_to_memcells[cook_cid] == [mc_cooking]
    # members tracks entity ids through merge — python cluster holds mc_001 and mc_002.
    assert cluster_store[py_cid].members == ["mc_001", "mc_002"]
    assert cluster_store[cook_cid].members == ["mc_003"]

    # Downstream ProfileExtractor: feed [prior_cells..., mc_python_2] as one chronological list.
    captured_prompt: dict[str, str] = {}

    def handler(messages: list[LLMChatMessage], **_kwargs: Any) -> ChatResponse:
        captured_prompt["text"] = messages[0].content
        return ChatResponse(
            content=json.dumps(
                {
                    "explicit_info": [
                        {
                            "category": "Technical Skills",
                            "description": "Alice is a Python developer interested in async.",
                            "evidence": "Asked about async retries and timeouts.",
                        }
                    ],
                    "implicit_traits": [],
                }
            ),
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)
    prior_cells = id_to_memcells[py_cid][:-1]  # everything except mc_python_2 itself

    profile = await ProfileExtractor(llm=fake).aextract(
        [*prior_cells, mc_python_2],
        sender_id="u_alice",
    )

    assert profile.owner_id == "u_alice"
    rendered = captured_prompt["text"]
    assert "async retry loops" in rendered
    assert "async timeouts" in rendered


async def test_state_persistence_roundtrip_keeps_pipeline_consistent() -> None:
    """Persist after one event, restore via pydantic serialisation, continue clustering.

    Verifies Cluster.model_dump() / model_validate() round-trip is lossless enough for the
    geometry decision to be identical on the restored clusters list.
    """
    base_ts = 1_700_000_000_000
    mc_1 = _memcell("How does asyncio handle cancellation?", base_ts)
    mc_2 = _memcell("Followup on asyncio cancel semantics.", base_ts + 60_000)

    vec_1 = np.array([1.0, 0.0], dtype=np.float32)
    vec_2 = np.array([0.99, 0.01], dtype=np.float32)

    clusters: list[Cluster] = []
    new_c1 = Cluster(id="cid_0", centroid=vec_1, last_ts=mc_1.timestamp)
    result_1 = await cluster_by_geometry(new_c1, clusters)
    assert result_1 is None
    clusters.append(new_c1)

    # Caller persists via pydantic serialisation.
    serialised = [c.model_dump() for c in clusters]
    restored: list[Cluster] = [Cluster.model_validate(d) for d in serialised]

    # Continue from restored clusters list.
    new_c2 = Cluster(centroid=vec_2, last_ts=mc_2.timestamp)
    result_2 = await cluster_by_geometry(new_c2, restored)

    assert result_1 is None  # first was a new cluster (appended)
    assert result_2 is not None  # second merges into restored[0]
    assert result_2.id == "cid_0"  # id passes through from existing
