"""End-to-end pipeline test: caller-embedded MemCells → clustering → profile.

Demonstrates the contract: ``cluster_by_geometry`` returns ``(cluster_id, state)``;
the caller maintains a ``cluster_id -> [MemCell, ...]`` index; the prior
``MemCell``s of the assigned cluster get handed to ``ProfileExtractor`` as
``cluster_episodes``.

No real embedding model is used — vectors are deterministic by hand so the
geometric decisions are predictable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import everalgo.llm
from everalgo.clustering import ClusterConfig, ClusterState, cluster_by_geometry
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import MemCell, Message, MessageRole
from everalgo.user_memory.profile import ProfileExtractor

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def reset_everalgo_llm_state() -> Iterator[None]:
    """Reset the everalgo.llm default + active client between tests."""
    saved_default = everalgo.llm._default
    token = everalgo.llm._active.set(None)
    try:
        everalgo.llm._default = None
        yield
    finally:
        everalgo.llm._default = saved_default
        everalgo.llm._active.reset(token)


def _memcell(event_id: str, content: str, ts_ms: int) -> MemCell:
    msg = Message(
        role=MessageRole.USER,
        content=content,
        timestamp=ts_ms,
        sender_id="u_alice",
        sender_name="Alice",
    )
    return MemCell(
        event_id=event_id,
        original_data=[{"message": msg.model_dump(exclude_none=True)}],
        timestamp=ts_ms,
        participants=["u_alice"],
        sender_ids=["u_alice"],
    )


async def test_two_similar_memcells_cluster_then_profile_sees_prior() -> None:
    """Three MemCells stream through clustering; topic-similar pair shares a cluster.

    Scenario:
    - mc_python_1 (vec=[1,0,0]) — first ever, mints ``cluster_000``.
    - mc_python_2 (vec=[0.95,0.05,0]) — cosine ≈ 0.998 vs centroid → joins cluster_000.
    - mc_cooking  (vec=[0,0,1])     — orthogonal → mints ``cluster_001``.

    Then ``mc_python_2`` is fed to ``ProfileExtractor`` with the first cell as
    ``cluster_episodes``. Asserts:
        1. Cluster ids match expectations.
        2. ``ProfileExtractor`` receives the prior MemCell's content in its prompt.
        3. Profile owner_id propagates from ``mc_python_2``.
    """
    base_ts = 1_700_000_000_000
    mc_python_1 = _memcell("mc_py_1", "How do async retry loops work in Python?", base_ts)
    mc_python_2 = _memcell("mc_py_2", "Followup on async timeouts in asyncio.", base_ts + 60_000)
    mc_cooking = _memcell("mc_cook_1", "Any tips on sourdough hydration?", base_ts + 120_000)

    vec_python_1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_python_2 = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    vec_cooking = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    state = ClusterState.empty()
    cid_to_memcells: dict[str, list[MemCell]] = {}
    config = ClusterConfig()

    for memcell, vec in [
        (mc_python_1, vec_python_1),
        (mc_python_2, vec_python_2),
        (mc_cooking, vec_cooking),
    ]:
        cid, state = await cluster_by_geometry(vec, memcell.timestamp, state, config=config)
        cid_to_memcells.setdefault(cid, []).append(memcell)

    # Geometric expectations.
    assert state.next_idx == 2
    assert len(cid_to_memcells) == 2
    py_cid = next(cid for cid, cells in cid_to_memcells.items() if mc_python_1 in cells)
    cook_cid = next(cid for cid, cells in cid_to_memcells.items() if mc_cooking in cells)
    assert py_cid != cook_cid
    assert cid_to_memcells[py_cid] == [mc_python_1, mc_python_2]
    assert cid_to_memcells[cook_cid] == [mc_cooking]

    # Downstream ProfileExtractor: feed mc_python_2 + its prior cluster MemCells.
    captured_prompt: dict[str, str] = {}

    def handler(messages: list[LLMChatMessage], **_kwargs: Any) -> ChatResponse:
        captured_prompt["text"] = messages[0].content
        return ChatResponse(
            content=(
                '{"explicit_info": ['
                '{"category": "Technical Skills",'
                ' "description": "Alice is a Python developer interested in async.",'
                ' "evidence": "Asked about async retries and timeouts.",'
                ' "sources": ["mc_py_1", "mc_py_2"]}'
                "],"
                '"implicit_traits": []}'
            ),
            model="fake",
        )

    fake = FakeLLMClient(handler=handler)
    prior_cells = cid_to_memcells[py_cid][:-1]  # everything except mc_python_2 itself

    profile = await ProfileExtractor().aextract(
        mc_python_2,
        cluster_episodes=prior_cells,
        llm=fake,
    )

    assert profile.owner_id == "u_alice"
    # The prior MemCell's content reached the prompt — proving the pipeline composed.
    rendered = captured_prompt["text"]
    assert "async retry loops" in rendered
    assert "async timeouts" in rendered


async def test_state_persistence_roundtrip_keeps_pipeline_consistent() -> None:
    """Persist after one event, restore, continue clustering — second event lands in the same cluster.

    Verifies ``to_dict``/``from_dict`` round-trip is lossless enough for the
    geometry decision to be identical on the restored state.
    """
    base_ts = 1_700_000_000_000
    mc_1 = _memcell("mc_1", "How does asyncio handle cancellation?", base_ts)
    mc_2 = _memcell("mc_2", "Followup on asyncio cancel semantics.", base_ts + 60_000)

    vec_1 = np.array([1.0, 0.0], dtype=np.float32)
    vec_2 = np.array([0.99, 0.01], dtype=np.float32)

    state = ClusterState.empty()
    cid_1, state = await cluster_by_geometry(vec_1, mc_1.timestamp, state, config=ClusterConfig())

    # Caller persists.
    serialised = state.to_dict()
    restored = ClusterState.from_dict(serialised)

    # Continue from restored state.
    cid_2, _ = await cluster_by_geometry(vec_2, mc_2.timestamp, restored, config=ClusterConfig())

    assert cid_1 == cid_2 == "cluster_000"
