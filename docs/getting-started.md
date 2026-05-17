# Getting Started

This guide walks through a complete user-memory pipeline end-to-end — boundary detection, four extractors, clustering, and profile building — in about 5 minutes.
No API key is needed: all LLM calls are handled by `FakeLLMClient`.

---

## 1. Install

```bash
pip install everalgo-user-memory
```

This pulls `everalgo-core`, `everalgo-boundary`, and `everalgo-clustering` automatically.

For the monorepo development install, see [Installation](installation.md).

---

## 2. A runnable end-to-end example

The following script mirrors [`examples/06_full_user_memory_pipeline.py`](../examples/06_full_user_memory_pipeline.py) in the repo.
Copy it into a file and run it — it prints output for every pipeline stage.

```python
"""Full user-memory pipeline using FakeLLMClient — no API key required."""

from __future__ import annotations

import asyncio
import json

import numpy as np

from everalgo.clustering import Cluster, cluster_by_geometry
from everalgo.llm.types import ChatMessage as LLMChatMessage, ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import BoundaryDetector
from everalgo.user_memory import AtomicFactExtractor, EpisodeExtractor, ForesightExtractor, ProfileExtractor

# ---------------------------------------------------------------------------
# Scripted LLM responses — one per pipeline stage, in call order.
# ---------------------------------------------------------------------------

_BOUNDARY_JSON = json.dumps({"reasoning": "single coherent topic", "boundaries": [], "should_wait": False})
_EPISODE_JSON = json.dumps({"title": "Alice asks about Python async", "content": "Discussion on async retry."})
_FORESIGHT_JSON = json.dumps([{"content": "Alice will read the follow-up doc", "evidence": "...", "start_time": "2023-11-14", "end_time": "2023-11-21", "duration_days": 7}])
_ATOMIC_FACT_JSON = json.dumps({"atomic_facts": {"time": "2023-11-14 22:13 UTC", "atomic_fact": ["Alice is learning async retry semantics."]}})
_PROFILE_JSON = json.dumps({"explicit_info": [{"category": "Technical Skills", "description": "Python developer."}], "implicit_traits": []})

_STAGE_ORDER = ("boundary", "episode", "foresight", "atomic_fact", "profile")
_STAGE_RESPONSE = dict(zip(_STAGE_ORDER, [_BOUNDARY_JSON, _EPISODE_JSON, _FORESIGHT_JSON, _ATOMIC_FACT_JSON, _PROFILE_JSON]))


def _make_fake() -> FakeLLMClient:
    """Route each LLM call to its scripted response by pipeline call order."""
    call_index = 0

    def handler(messages: list[LLMChatMessage], **_kwargs: object) -> ChatResponse:
        nonlocal call_index
        stage = _STAGE_ORDER[call_index]
        call_index += 1
        return ChatResponse(content=_STAGE_RESPONSE[stage], model="fake")

    return FakeLLMClient(handler=handler)


async def main() -> None:
    fake = _make_fake()

    # Build a two-turn conversation.
    messages = [
        ChatMessage(id="m1", role="user", content="Can you walk me through Python async retry semantics?", timestamp=1_700_000_000_000, sender_id="u_alice", sender_name="Alice"),
        ChatMessage(id="m2", role="assistant", content="Sure — I'll send a follow-up doc next week.", timestamp=1_700_000_001_000, sender_id="assistant"),
    ]

    # 1. Boundary detection — splits messages into MemCell segments.
    result = await BoundaryDetector(llm=fake).adetect(messages, is_final=True)
    mc = result.cells[0]
    print(f"[boundary]  cells={len(result.cells)}  tail={len(result.tail)}")

    # 2. Episode — a narrative summary of what happened.
    episode = await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    print(f"[episode]   subject={episode.subject!r}")

    # 3. Foresight — predicted future events.
    foresights = await ForesightExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    print(f"[foresight] count={len(foresights)}")

    # 4. AtomicFact — granular factual statements.
    facts = await AtomicFactExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    print(f"[atomic_fact] count={len(facts)}")

    # 5. Clustering — caller wraps each item as a size-1 Cluster; no LLM call.
    prior_mc = MemCell(
        items=[ChatMessage(id="m0", role="user", content="Earlier Python async question", timestamp=1_699_900_000_000, sender_id="u_alice", sender_name="Alice")],
        timestamp=1_699_900_000_000,
    )
    existing: list[Cluster] = []
    c_prior = Cluster(centroid=np.array([1.0, 0.0, 0.0], dtype=np.float32), last_ts=prior_mc.timestamp)
    merged = await cluster_by_geometry(c_prior, existing)
    if merged is None:
        c_prior = c_prior.model_copy(update={"id": "cid_001"})
        existing.append(c_prior)

    c_current = Cluster(centroid=np.array([0.98, 0.02, 0.0], dtype=np.float32), last_ts=mc.timestamp)
    merged = await cluster_by_geometry(c_current, existing)
    if merged is not None:
        idx = next(i for i, c in enumerate(existing) if c.id == merged.id)
        existing[idx] = merged
        print(f"[clustering] merged into cid={merged.id!r}")
    else:
        c_current = c_current.model_copy(update={"id": "cid_002"})
        existing.append(c_current)
        print(f"[clustering] new cluster cid={c_current.id!r}")

    # 6. Profile — a structured user profile from a cluster of MemCells.
    profile = await ProfileExtractor(llm=fake).aextract([prior_mc, mc], sender_id="u_alice")
    print(f"[profile]   owner={profile.owner_id!r}")
    print(f"\ntotal LLM calls: {fake.call_count}  (expected 5)")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
uv run python my_pipeline.py
```

Expected output:

```
[boundary]  cells=1  tail=0
[episode]   subject='Alice asks about Python async'
[foresight] count=1
[atomic_fact] count=1
[clustering] merged into cid='cid_001'
[profile]   owner='u_alice'

total LLM calls: 5  (expected 5)
```

---

## 3. Key concepts to know before going further

### The `a`-prefix convention

Methods named `adetect`, `aextract`, `arank`, `aparse` are **native async** — they make real I/O calls (LLM, network) and must be called with `await`.
Methods without the `a` prefix (`rank`, `rrf`, `count_tokens`) are **synchronous pure-compute** — call them directly.

```python
# async: always await
episode = await EpisodeExtractor(llm=client).aextract(memcell, sender_id="u_alice")

# sync: no await
merged = rank.fusion.rrf(vec_hits, keyword_hits)
```

See [Async–sync bridge](concepts/async-sync-bridge.md) for the full explanation.

### Injecting a real LLM

`FakeLLMClient` is for tests and local exploration.
For production or experiments with a real model, build an `OpenAICompatClient` and pass it to each extractor:

```python
from everalgo.llm.providers.openai_compat import OpenAICompatClient
from everalgo.user_memory import EpisodeExtractor

client = OpenAICompatClient(
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
    model="gpt-4o-mini",
)
episode = await EpisodeExtractor(llm=client).aextract(memcell, sender_id="u_alice")
```

The `llm=` argument is bound at construction time. There is no global default — each operator instance holds its own client.

### Two access paths, same class

Every operator can be imported via the **product path** (what evermem uses) or the **physical path** (what algorithm engineers use to iterate on specific modules):

```python
# Product path — follow evermem contracts
from everalgo.user_memory import BoundaryDetector

# Physical path — edit boundary logic, prompt, tokenize
from everalgo.boundary.chat import BoundaryDetector
```

Both import the same class.

---

## 4. Next steps

- **Architecture deep-dive** → [concepts/architecture.md](concepts/architecture.md)
- **Stateless design rationale** → [concepts/stateless-design.md](concepts/stateless-design.md)
- **Full example scripts** → [`examples/`](../examples/) (01 through 06)
- **API index per package** → [api/README.md](api/README.md)
