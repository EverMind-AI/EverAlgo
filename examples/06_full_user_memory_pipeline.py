"""Full user-memory pipeline — chat → boundary → 4 extractors.

Chains all five pipeline stages end-to-end:

1. ``BoundaryDetector.adetect``          → ``MemCell``
2. ``EpisodeExtractor.aextract``         → single ``Episode``
3. ``ForesightExtractor.aextract``       → list of ``Foresight``
4. ``AtomicFactExtractor.aextract``      → list of ``AtomicFact``
5. ``cluster_by_geometry``               → cluster ID (no LLM)
6. ``ProfileExtractor.aextract``         → single ``Profile``  (cluster fan-in: 2 MemCells)

One ``FakeLLMClient`` handles all 5 LLM calls.  The handler routes by call-count
because the pipeline's call order is deterministic.

Run:
    uv run python examples/06_full_user_memory_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np

from everalgo.clustering import Cluster, cluster_by_geometry
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import AtomicFact, ChatMessage, Episode, Foresight, MemCell, Profile
from everalgo.user_memory import BoundaryDetector
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.episode import EpisodeExtractor
from everalgo.user_memory.foresight import ForesightExtractor
from everalgo.user_memory.profile import ProfileExtractor

# ---------------------------------------------------------------------------
# Scripted LLM responses — one per pipeline stage, in call order.
# ---------------------------------------------------------------------------

_BOUNDARY_JSON = json.dumps({"reasoning": "single coherent topic", "boundaries": [], "should_wait": False})

_EPISODE_JSON = json.dumps(
    {
        "title": "Alice asks about Python async retry semantics",
        "content": "Alice initiated a discussion on Python async retry patterns; assistant offered a follow-up.",
    }
)

_FORESIGHT_JSON = json.dumps(
    [
        {
            "content": "Alice will read the assistant's follow-up doc on async retries next week",
            "evidence": "Assistant committed to a follow-up doc next week",
            "start_time": "2023-11-14",
            "end_time": "2023-11-21",
            "duration_days": 7,
        }
    ]
)

_ATOMIC_FACT_JSON = json.dumps(
    {
        "atomic_facts": {
            "time": "November 14, 2023 at 22:13 UTC",
            "atomic_fact": [
                "Alice is learning Python async retry semantics.",
                "Assistant promised a follow-up document next week.",
            ],
        }
    }
)

_PROFILE_JSON = json.dumps(
    {
        "explicit_info": [
            {
                "category": "Technical Skills",
                "description": "Alice is a Python developer interested in async patterns.",
                "evidence": "Alice asked about async retry semantics.",
                "sources": ["2023-11-14 22:13|mc_001"],
            }
        ],
        "implicit_traits": [
            {
                "trait": "[Curious Learner]",
                "description": "Actively pursues new technical depth without prompting.",
                "basis": "Self-initiated a deep-dive question into async patterns.",
                "evidence": "Volunteered her current learning focus.",
                "sources": ["2023-11-14 22:13|mc_001"],
            }
        ],
    }
)

_STAGE_ORDER = ("boundary", "episode", "foresight", "atomic_fact", "profile")
_STAGE_RESPONSE: dict[str, str] = {
    "boundary": _BOUNDARY_JSON,
    "episode": _EPISODE_JSON,
    "foresight": _FORESIGHT_JSON,
    "atomic_fact": _ATOMIC_FACT_JSON,
    "profile": _PROFILE_JSON,
}


def _make_fake() -> FakeLLMClient:
    """Route each call to its scripted response by pipeline call-count position."""
    call_index = 0

    def handler(messages: list[LLMChatMessage], **_kwargs: Any) -> ChatResponse:
        nonlocal call_index
        stage = _STAGE_ORDER[call_index]
        call_index += 1
        return ChatResponse(content=_STAGE_RESPONSE[stage], model="fake")

    return FakeLLMClient(handler=handler)


def _conversation() -> list[ChatMessage]:
    return [
        ChatMessage(
            id="m1",
            role="user",
            content="I've been getting into Python async lately. Can you walk me through retry semantics?",
            timestamp=1_700_000_000_000,
            sender_id="u_alice",
            sender_name="Alice",
        ),
        ChatMessage(
            id="m2",
            role="assistant",
            content="Sure — what failure mode are you seeing? I'll send a follow-up doc next week.",
            timestamp=1_700_000_001_000,
            sender_id="assistant",
        ),
    ]


def _prior_memcell() -> MemCell:
    """A prior MemCell from the same sender — feeds the cluster fan-in for ProfileExtractor."""
    return MemCell(
        items=[
            ChatMessage(
                id="m0",
                role="user",
                content="Earlier asked about Python async context managers",
                timestamp=1_699_900_000_000,
                sender_id="u_alice",
                sender_name="Alice",
            )
        ],
        timestamp=1_699_900_000_000,
    )


async def main() -> None:
    """Run all 5 pipeline stages and print each extractor's output."""
    fake = _make_fake()

    # --- 1. Boundary detection -----------------------------------------------
    boundary_output = await BoundaryDetector(llm=fake).adetect(_conversation(), is_final=True)
    mc = boundary_output.cells[0]
    print(f"[boundary]  cells={len(boundary_output.cells)}  tail={len(boundary_output.tail)}")

    # --- 2. Episode ----------------------------------------------------------
    episode: Episode = await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    print(f"[episode]   owner={episode.owner_id!r}  subject={episode.subject!r}")

    # --- 3. Foresight --------------------------------------------------------
    foresights: list[Foresight] = await ForesightExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    print(f"[foresight] count={len(foresights)}  duration_days={foresights[0].duration_days if foresights else 'n/a'}")

    # --- 4. AtomicFact -------------------------------------------------------
    facts: list[AtomicFact] = await AtomicFactExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    print(f"[atomic_fact] count={len(facts)}")
    for i, f in enumerate(facts):
        print(f"  fact[{i}]: {f.content!r}")

    # --- 5. cluster_by_geometry (no LLM call) --------------------------------
    mc_prior = _prior_memcell()
    clusters: list[Cluster] = []
    vec_prior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_current = np.array([0.98, 0.02, 0.0], dtype=np.float32)  # cosine ≈ 0.9998 → same cluster

    # Caller stamps entity ids into members before passing to the clustering operator.
    mc_prior_id = "mc_prior"
    mc_current_id = "mc_current"

    new_c_prior = Cluster(id="cid_0", centroid=vec_prior, last_ts=mc_prior.timestamp, members=[mc_prior_id])
    result_prior = cluster_by_geometry(new_c_prior, clusters)
    assert result_prior is None
    clusters.append(new_c_prior)

    new_c_current = Cluster(centroid=vec_current, last_ts=mc.timestamp, members=[mc_current_id])
    result_current = cluster_by_geometry(new_c_current, clusters)
    assert result_current is not None
    assert result_current.id == "cid_0"
    clusters[0] = result_current
    print(
        f"[clustering] prior=new(cid_0)  current=merged(id={result_current.id!r})"
        f"  total_clusters={len(clusters)}  members={clusters[0].members!r}"
    )

    # --- 6. Profile (cluster fan-in: prior + current MemCell) ----------------
    # Caller derives MemCell list directly from cluster.members — no external map needed.
    memcell_dict = {mc_prior_id: mc_prior, mc_current_id: mc}
    cluster_memcells = [memcell_dict[mid] for mid in clusters[0].members]
    profile: Profile = await ProfileExtractor(llm=fake).aextract(cluster_memcells, sender_id="u_alice")
    print(f"[profile]   owner={profile.owner_id!r}  timestamp={profile.timestamp}")
    # explicit_info / implicit_traits are LLM-emitted extra fields (extra="allow" on Profile).
    extra = profile.model_extra or {}
    explicit: list[object] = extra.get("explicit_info") or []
    traits: list[object] = extra.get("implicit_traits") or []
    print(f"  explicit_info count : {len(explicit)}")
    print(f"  implicit_traits count: {len(traits)}")
    if explicit:
        print(f"  first explicit_info : {explicit[0]}")

    print(f"\ntotal LLM calls: {fake.call_count}  (expected 5)")


if __name__ == "__main__":
    asyncio.run(main())
