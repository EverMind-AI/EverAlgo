"""Decision memory pipeline — Extract → Reflect → Principle.

Chains the three Decision-line operators on two MemCells built directly from chat
(no BoundaryDetector, no markdown, no clustering):

1. ``DecisionExtractor.aextract``   (Jan)  → one Decision, ``owner_id=None``
2. ``DecisionExtractor.aextract``   (Aug)  → one Decision, ``owner_id=None``
3. ``DecisionReflector.areflect``   INIT   → one merged Decision (still a Decision)
4. ``PrincipleExtractor.aextract``         → list[Principle] from the two *instance*
   decisions, not the merge. Entry ids are caller-supplied stand-ins for EverOS
   markdown ids; this operator never invents them.

One ``FakeLLMClient`` scripts all four LLM calls in order. No API key.

Run:
    uv run python examples/08_decision_memory_pipeline.py
"""

from __future__ import annotations

import asyncio
import json

from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, Decision, MemCell, Principle
from everalgo.user_memory import DecisionExtractor, DecisionReflector, PrincipleExtractor

# 2026-01-15T12:00:00Z / 2026-08-20T12:00:00Z — chronological INIT input.
_TS_JAN = 1_768_478_400_000
_TS_AUG = 1_787_227_200_000

# Caller-supplied storage ids. EverOS later passes markdown entry ids (``dc_`` prefix);
# the algorithm never generates this format.
_ENTRY_JAN = "dc_001"
_ENTRY_AUG = "dc_002"

# ---------------------------------------------------------------------------
# Scripted LLM responses — one per pipeline stage, in call order.
# ---------------------------------------------------------------------------

_JAN_DECISION_JSON = json.dumps(
    {
        "decisions": [
            {
                "title": "Agent core language",
                "decision": "Use Python and LangChain for the core Agent Runtime.",
                "reason": "Need to ship agent capability quickly while the surface is still moving.",
                "impact": "Device capabilities stay behind a Python API.",
                "tags": ["architecture", "runtime"],
            }
        ]
    }
)

_AUG_DECISION_JSON = json.dumps(
    {
        "decisions": [
            {
                "title": "In-house Agent Runtime",
                "decision": "Replace LangChain with a self-developed Agent Runtime.",
                "reason": "Need control over the main loop; a third-party framework hides too much.",
                "impact": "Device talks through our own APIs; framework upgrades no longer dictate the loop.",
                "tags": ["architecture", "runtime"],
            }
        ]
    }
)

# Structured Output schema order: decision, reason, then title / impact / tags.
_REFLECT_JSON = json.dumps(
    {
        "decision": "Use a self-developed Agent Runtime, keeping Python for the core.",
        "reason": "Iteration speed still matters, but the main loop has to be ours.",
        "title": "Agent Runtime ownership",
        "impact": "Device capabilities connect through our APIs.",
        "tags": ["architecture", "runtime"],
    }
)

_PRINCIPLE_JSON = json.dumps(
    {
        "principles": [
            {
                "title": "Own the loop, iterate the rest",
                "statement": (
                    "Keep ownership of the Agent main loop; prefer iteration speed over a "
                    "premature rewrite of everything else."
                ),
                "source_entry_ids": [_ENTRY_JAN, _ENTRY_AUG],
            }
        ]
    }
)


def _memcell_jan() -> MemCell:
    """January slice: ship with LangChain to move fast."""
    return MemCell(
        items=[
            ChatMessage(
                id="m_jan_1",
                role="user",
                content=(
                    "We need an Agent Runtime this quarter. LangChain in Python gets us a "
                    "working core fast. Device-side can stay Rust."
                ),
                timestamp=_TS_JAN,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ChatMessage(
                id="m_jan_2",
                role="assistant",
                content="Agreed — Python plus LangChain for the core, Rust on device.",
                timestamp=_TS_JAN + 1_000,
                sender_id="assistant",
            ),
        ],
        timestamp=_TS_JAN,
    )


def _memcell_aug() -> MemCell:
    """August slice: take back the main loop."""
    return MemCell(
        items=[
            ChatMessage(
                id="m_aug_1",
                role="user",
                content=(
                    "LangChain is hiding the main loop. We should replace it with our own "
                    "Agent Runtime and keep Python for the core."
                ),
                timestamp=_TS_AUG,
                sender_id="u_alice",
                sender_name="Alice",
            ),
            ChatMessage(
                id="m_aug_2",
                role="assistant",
                content="Then the device talks through our APIs, not the framework's.",
                timestamp=_TS_AUG + 1_000,
                sender_id="assistant",
            ),
        ],
        timestamp=_TS_AUG,
    )


async def main() -> None:
    """Run Extract → Reflect → Principle and print each stage."""
    fake = FakeLLMClient(
        responses=[
            _JAN_DECISION_JSON,
            _AUG_DECISION_JSON,
            _REFLECT_JSON,
            _PRINCIPLE_JSON,
        ]
    )
    extractor = DecisionExtractor(llm=fake)
    reflector = DecisionReflector(llm=fake)
    principles = PrincipleExtractor(llm=fake)

    # --- 1-2. Extract (no sender_id; owner_id stays None) --------------------
    extracted_jan = await extractor.aextract(_memcell_jan())
    extracted_aug = await extractor.aextract(_memcell_aug())
    assert len(extracted_jan) == 1 and extracted_jan[0].owner_id is None
    assert len(extracted_aug) == 1 and extracted_aug[0].owner_id is None
    d_jan, d_aug = extracted_jan[0], extracted_aug[0]
    print(f"[extract]   n=1  owner_id={d_jan.owner_id!r}  ts={d_jan.timestamp}  title={d_jan.title!r}")
    print(f"[extract]   n=1  owner_id={d_aug.owner_id!r}  ts={d_aug.timestamp}  title={d_aug.title!r}")

    # --- 3. Reflect INIT -- still a Decision, not a Principle ----------------
    merged: Decision = await reflector.areflect([d_jan, d_aug])
    assert isinstance(merged, Decision) and not isinstance(merged, Principle)
    assert merged.owner_id is None
    assert merged.timestamp == d_aug.timestamp
    print(f"[reflect]   owner_id={merged.owner_id!r}  timestamp={merged.timestamp}  decision={merged.decision!r}")

    # --- 4. Principle from the instance cluster (not the merge) --------------
    cluster: list[tuple[str, Decision]] = [(_ENTRY_JAN, d_jan), (_ENTRY_AUG, d_aug)]
    extracted_principles = await principles.aextract(cluster, owner_id="u_alice")
    assert len(extracted_principles) == 1
    principle = extracted_principles[0]
    assert isinstance(principle, Principle)
    assert principle.owner_id == "u_alice"
    assert set(principle.source_entry_ids) <= {_ENTRY_JAN, _ENTRY_AUG}
    print(
        f"[principle] n=1  owner={principle.owner_id!r}  "
        f"sources={principle.source_entry_ids!r}  statement={principle.statement!r}"
    )

    print(f"\ntotal LLM calls: {fake.call_count}  (expected 4)")
    assert fake.call_count == 4


if __name__ == "__main__":
    asyncio.run(main())
