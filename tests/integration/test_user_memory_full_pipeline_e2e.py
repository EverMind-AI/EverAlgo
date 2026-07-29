"""End-to-end pipeline test: raw messages → boundary → clustering → all four user-memory extractors.

This test is the algorithm-correctness gate for the user-memory pipeline.  Beyond verifying that
data flows through all 5 stages (boundary + Episode + Foresight + AtomicFact + Profile), it
**captures every rendered LLM prompt and asserts the placeholder substitutions are correct** —
this is the most important coverage, because a silently miswired prompt (missing placeholder,
wrong sender name, garbled conversation rendering) cannot be caught by output-shape assertions
alone.

Pipeline stages:

1. ``BoundaryDetector.adetect(messages)``                    → ``MemCell``
2. ``EpisodeExtractor.aextract(mc, sender_id=...)``          → single ``Episode``
3. ``ForesightExtractor.aextract(mc, sender_id=...)``        → list[Foresight]
4. ``AtomicFactExtractor.aextract(mc, sender_id=...)``       → list[AtomicFact]
   [cluster_by_geometry step — no LLM call; assigns both mc_prior and mc to "cluster_000"]
5. ``ProfileExtractor.aextract([mc_prior, mc], sender_id=...)`` → single ``Profile``

The 5 LLM calls share one ``FakeLLMClient``; its handler routes by call-count (the pipeline's
call order is deterministic) and captures every prompt for later inspection.
``cluster_by_geometry`` makes no LLM call (and is a sync function), so the total LLM call count stays at 5.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from everalgo.clustering import Cluster, cluster_by_geometry
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.assertions import (
    assert_atomic_fact_shape,
    assert_episode_shape,
    assert_foresight_shape,
    assert_profile_shape,
)
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import BoundaryDetector
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.episode import EpisodeExtractor
from everalgo.user_memory.foresight import ForesightExtractor
from everalgo.user_memory.profile import ProfileExtractor


def _prior_memcell() -> MemCell:
    """A prior MemCell constructed directly (no boundary call) for the cluster fan-in test.

    Same sender_id/sender_name as the main conversation so the Profile extractor renders it as
    Alice's earlier exchange.  Timestamp 1_699_900_000_000 precedes the main conversation
    (1_700_000_000_000) to maintain chronological order inside the cluster's MemCell list.
    """
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


def _conversation() -> list[ChatMessage]:
    """Two-turn dialogue with deterministic timestamps for prompt-content assertions."""
    return [
        ChatMessage(
            id="m1",
            role="user",
            content="I've been getting into Python async lately. Can you walk me through retry semantics?",
            timestamp=1_700_000_000_000,  # 2023-11-14T22:13:20Z
            sender_id="u_alice",
            sender_name="Alice",
        ),
        ChatMessage(
            id="m2",
            role="assistant",
            content="Sure — what failure mode are you seeing? I'll send a follow-up doc next week.",
            timestamp=1_700_000_001_000,  # 2023-11-14T22:13:21Z
            sender_id="assistant",
        ),
    ]


# ---------------------------------------------------------------------------
# LLM handler — captures every prompt for later inspection
# ---------------------------------------------------------------------------


_BOUNDARY_JSON = '{"reasoning": "single coherent topic", "boundaries": [], "should_wait": false}'

_EPISODE_JSON = (
    '{"title": "Alice asks about Python async retry semantics",'
    ' "content": "Alice initiated a discussion on Python async retry patterns; assistant offered a follow-up."}'
)

_FORESIGHT_JSON = (
    '{"foresights": ['
    '{"content": "Alice will read the assistant\'s follow-up doc on async retries next week",'
    ' "evidence": "Assistant committed to a follow-up doc next week",'
    ' "start_time": "2023-11-14", "end_time": "2023-11-21", "duration_days": 7}'
    "]}"
)

_ATOMIC_FACT_JSON = (
    '{"atomic_facts": {'
    '"time": "November 14, 2023 at 22:13 UTC",'
    '"atomic_fact": ['
    '"Alice is learning Python async retry semantics.",'
    '"Assistant promised a follow-up document next week."'
    "]}}"
)

_PROFILE_JSON = (
    '{"explicit_info": ['
    '{"category": "Technical Skills",'
    ' "description": "Alice is a Python developer interested in async patterns.",'
    ' "evidence": "Alice asked about async retry semantics."}'
    "],"
    '"implicit_traits": ['
    '{"trait": "[Curious Learner]",'
    ' "description": "Actively pursues new technical depth without prompting.",'
    ' "basis": "Self-initiated a deep-dive question into async patterns.",'
    ' "evidence": "Volunteered her current learning focus."}'
    "]}"
)

# Pipeline call order is deterministic: boundary first, then the 4 extractors in test order below.
_STAGE_ORDER = ("boundary", "episode", "foresight", "atomic_fact", "profile")
_STAGE_RESPONSE = {
    "boundary": _BOUNDARY_JSON,
    "episode": _EPISODE_JSON,
    "foresight": _FORESIGHT_JSON,
    "atomic_fact": _ATOMIC_FACT_JSON,
    "profile": _PROFILE_JSON,
}


def _make_capturing_client() -> tuple[FakeLLMClient, dict[str, str]]:
    """Return a fake client whose handler captures each call's prompt by pipeline-stage name."""
    captured: dict[str, str] = {}
    call_index = 0

    def handler(messages: list[LLMChatMessage], **_kwargs: Any) -> ChatResponse:
        nonlocal call_index
        stage = _STAGE_ORDER[call_index]
        assert isinstance(messages[0].content, str)  # narrow for test
        captured[stage] = messages[0].content
        call_index += 1
        return ChatResponse(content=_STAGE_RESPONSE[stage], model="fake")

    return FakeLLMClient(handler=handler), captured


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


async def test_user_memory_full_pipeline_e2e() -> None:
    """Run the full user-memory pipeline and assert every rendered prompt is wired correctly."""
    fake, captured = _make_capturing_client()

    # --- 1. Boundary detection -------------------------------------------------
    boundary_output = await BoundaryDetector(llm=fake).adetect(_conversation(), is_final=True)
    assert boundary_output.tail == []
    assert len(boundary_output.cells) == 1
    mc = boundary_output.cells[0]

    boundary_prompt = captured["boundary"]
    assert "Alice" in boundary_prompt, "boundary prompt missing sender_name"
    assert "Python async" in boundary_prompt, "boundary prompt missing user message content"
    assert "follow-up doc" in boundary_prompt, "boundary prompt missing assistant message content"

    # --- 2. Episode ------------------------------------------------------------
    episode = await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    ep = assert_episode_shape(episode)
    assert ep.owner_id == "u_alice"

    episode_prompt = captured["episode"]
    # Placeholder substitution: literal labels rendered, no raw `{conversation_start_time}` / `{conversation}` /
    # `{custom_instructions}` left behind.
    assert "Conversation start time:" in episode_prompt
    # Episode renders the start time in YYYY-MM-DD HH:MM UTC (Weekday) format — pin the date portion only.
    assert "2023-11-14" in episode_prompt, "episode prompt missing rendered conversation_start_time"
    assert "{conversation_start_time}" not in episode_prompt
    assert "{conversation}" not in episode_prompt
    assert "{custom_instructions}" not in episode_prompt
    # Conversation is rendered as pseudo-JSON objects: field names quoted, values unquoted
    # (not strictly valid JSON but LLM-readable). Timestamps emitted in YYYY-MM-DD HH:MM UTC (Weekday) format
    # via ``_format_prompt_time`` so LLM can resolve relative time references per message.
    assert '"timestamp": 2023-11-14 22:13 UTC (Tuesday)' in episode_prompt, (
        "episode conversation missing Alice timestamp in prompt format"
    )
    assert '"speaker": Alice' in episode_prompt, "episode conversation missing Alice speaker"
    assert '"timestamp": 2023-11-14 22:13 UTC (Tuesday)' in episode_prompt, (
        "episode conversation missing assistant timestamp in prompt format"
    )
    assert '"speaker": assistant' in episode_prompt, "episode conversation missing assistant speaker"
    assert "Python async" in episode_prompt
    # sender_id is a required keyword argument (no default); passing a non-None value selects
    # USER_EPISODE_GENERATION_PROMPT, while sender_id=None selects the generic variant.
    # The `{user_name}` placeholder is substituted with the resolved sender_name "Alice"; the raw
    # placeholder string must no longer appear in the rendered prompt.
    assert "{user_name}" not in episode_prompt
    assert "User name: Alice" in episode_prompt, "episode prompt missing rendered user_name label"
    assert "focus on Alice" in episode_prompt, "episode prompt should be user-centred on Alice"

    # --- 3. Foresight ----------------------------------------------------------
    foresights = await ForesightExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    assert len(foresights) == 1
    fs = assert_foresight_shape(foresights[0])
    assert fs.owner_id == "u_alice"
    assert fs.duration_days == 7

    foresight_prompt = captured["foresight"]
    # Foresight template uses uppercase placeholders {USER_ID} / {USER_NAME} / {CONVERSATION_TEXT}.
    assert "- user_id: u_alice" in foresight_prompt, "foresight prompt missing sender_id substitution"
    assert "- user_name: Alice" in foresight_prompt, "foresight prompt missing sender_name substitution"
    assert "{USER_ID}" not in foresight_prompt
    assert "{USER_NAME}" not in foresight_prompt
    assert "{CONVERSATION_TEXT}" not in foresight_prompt
    assert "Python async" in foresight_prompt

    # --- 4. AtomicFact ---------------------------------------------------------
    facts = await AtomicFactExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    assert len(facts) == 2
    for f in facts:
        af = assert_atomic_fact_shape(f)
        assert af.owner_id == "u_alice"

    atomic_prompt = captured["atomic_fact"]
    # AtomicFact template uses uppercase placeholders {TIME} / {INPUT_TEXT}.
    assert "{TIME}" not in atomic_prompt
    assert "{INPUT_TEXT}" not in atomic_prompt
    # TIME label is rendered as natural-language UTC.
    assert "November 14, 2023" in atomic_prompt, "atomic_fact prompt missing rendered TIME label"
    # AtomicFact's _render_input_text emits `[<ISO>] <speaker>: <content>` lines (mirroring
    # the upstream reference `atomic_fact_extractor.py:255-262`) — the timestamp prefix anchors message-level
    # time signals into the LLM context so atomic_fact extraction can preserve when each event
    # happened (critical for LoCoMo temporal questions).
    assert "[2023-11-14 22:13:20] Alice: I've been getting into Python async" in atomic_prompt
    assert "[2023-11-14 22:13:21] assistant: Sure — what failure mode" in atomic_prompt

    # --- cluster_by_geometry (no LLM call — count stays at 5) -----------------
    mc_prior = _prior_memcell()
    clusters: list[Cluster] = []
    vec_prior = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_current = np.array([0.98, 0.02, 0.0], dtype=np.float32)  # cosine ~0.9998 > 0.65 threshold

    # Caller stamps entity ids into members before passing to the clustering operator.
    mc_prior_id = "mc_prior"
    mc_current_id = "mc_current"

    new_c_prior = Cluster(id="cid_0", centroid=vec_prior, last_ts=mc_prior.timestamp, members=[mc_prior_id])
    result_prior = cluster_by_geometry(new_c_prior, clusters)
    assert result_prior is None
    clusters.append(new_c_prior)

    new_c_current = Cluster(centroid=vec_current, last_ts=mc.timestamp, members=[mc_current_id])
    result_current = cluster_by_geometry(new_c_current, clusters)
    assert result_current is not None, "expected merged Cluster, got None"
    assert result_current.id == "cid_0", f"expected id passthrough, got {result_current.id!r}"
    clusters[0] = result_current

    assert clusters[0].count == 2, f"expected count=2 (merged), got {clusters[0].count}"
    assert clusters[0].members == [mc_prior_id, mc_current_id], (
        f"expected members=[mc_prior_id, mc_current_id], got {clusters[0].members!r}"
    )

    # Caller derives cluster_memcells from cluster.members — no external id→entity map needed.
    memcell_index = {mc_prior_id: mc_prior, mc_current_id: mc}
    cluster_memcells = [memcell_index[mid] for mid in clusters[0].members]

    # --- 5. Profile (cluster fan-in: both memcells passed, new signature) -----
    profile = await ProfileExtractor(llm=fake).aextract(cluster_memcells, sender_id="u_alice")
    pf = assert_profile_shape(profile)
    assert pf.owner_id == "u_alice"
    assert pf.timestamp == mc.timestamp  # contract: memcells[-1].timestamp drives Profile.timestamp

    profile_prompt = captured["profile"]
    # Profile template uses lowercase placeholder {conversation_text}.
    assert "{conversation_text}" not in profile_prompt
    # Profile's _render_conversation embeds sender_id alongside the speaker — verify both the format
    # and the actual user identity made it through.
    assert "(user_id:u_alice)" in profile_prompt, "profile prompt missing user_id substitution in conversation line"
    assert "Alice(user_id:u_alice)" in profile_prompt, "profile prompt missing speaker-with-user_id marker"
    assert "Python async" in profile_prompt
    # The prior cluster memcell must also appear in the Profile prompt — proves cluster fan-in works.
    assert "Python async context managers" in profile_prompt, (
        "profile prompt must contain prior memcell content to confirm cluster fan-in"
    )

    # --- Pipeline-wide invariants ---------------------------------------------
    assert set(captured.keys()) == set(_STAGE_ORDER), "every pipeline stage must have captured exactly one prompt"
