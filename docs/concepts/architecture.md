# Architecture

This document describes EverAlgo's structure, the two algorithm axes, the subpackage layout, naming conventions, LLM injection, and the boundary between EverAlgo and its caller (EverOS).

---

## 1. EverAlgo and EverOS

**EverOS** is the AI memory management system: it owns the API gateway, database persistence, orchestration, concurrency control, scene routing (which algorithm step uses which model), and the memory lifecycle.

**EverAlgo** is the algorithm library that EverOS depends on.
It is stateless and has no knowledge of storage, deployment topology, or scene routing.

```
┌──────────────────────────────────────────────────┐
│  EverOS — AI memory management system            │
│  API gateway / persistence / orchestration       │
│  concurrency / scene routing / lifecycle         │
└────────────────┬─────────────────────────────────┘
                 │  in-memory data structures
                 ▼
┌──────────────────────────────────────────────────┐
│  EverAlgo — algorithm library                    │
│  stateless · Extract / Rank operators            │
└──────────────────────────────────────────────────┘
```

**EverAlgo's boundary in one sentence per axis:**

- EverAlgo does not know about the database. The caller loads data and passes it in.
- EverAlgo does not know which LLM to use for which business scenario. The caller selects a client and injects it.
- EverAlgo does not know about concurrency or distributed locks. The caller serialises read-modify-write cycles.

---

## 2. Two algorithm axes: Extract and Rank

Every operator in EverAlgo belongs to one of two axes.
The contract is symmetric: both axes are **stateless**, **in-memory I/O**, and **storage-free**.

| Axis | When | Input | Output |
|---|---|---|---|
| **Extract** | write path | `list[ChatMessage]` → `MemCell` → extractors | structured memories: `Episode` / `Foresight` / `AtomicFact` / `Profile` / `AgentCase` / `AgentSkill` / `KnowledgeMemory` |
| **Rank** | read path | `RankInput` (multi-route recall candidates + pre-fetched cross-memory linkage) | `RankOutput` (ranked list) |

**Extract** is a two-stage pipeline.
The **boundary stage** (`BoundaryDetector.adetect`) consumes raw conversation messages and produces `MemCell` segments — coherent conversation units.
The **extractor stage** consumes each `MemCell` and produces derived memories.

**Rank** performs no storage I/O at all.
Cross-memory linkage (e.g. an episode's atomic facts) must be pre-fetched by the caller and passed into the `RankInput`; the ranker operates entirely in memory.

---

## 3. Subpackage layout

EverAlgo is a monorepo of 8 independently versioned PyPI distributions sharing the `everalgo.*` namespace via [PEP 420](https://peps.python.org/pep-0420/).

```
everalgo/                              # PEP 420 namespace package — no __init__.py
├── everalgo-core/                     # types / llm / prompts / testing
├── everalgo-boundary/                 # MemCell extractors + tokenize / split
├── everalgo-clustering/               # cluster_by_geometry / cluster_by_llm over list[Cluster]
├── everalgo-rank/                     # 4 rankers + fusion / weight / rerank tools
├── everalgo-parser/                   # multimodal raw-file → ParsedContent
├── everalgo-user-memory/              # Episode / Foresight / AtomicFact / Profile / Decision
├── everalgo-agent-memory/             # AgentCase / AgentSkill
└── everalgo-knowledge/                # KnowledgeMemory
```

Dependency topology (arrows point to the dependency):

```
                          everalgo-core
                              ▲
       ┌───────────┬──────────┴───────┬──────────┐
       │           │                  │          │
   boundary    clustering           rank       parser
       ▲           ▲                               ▲
       │           │                               │
   user-memory  agent-memory                 knowledge

Edges (arrow → dependency; every package also depends on core):
  user-memory  → boundary
  agent-memory → boundary, clustering
  knowledge    → parser
```

Each distribution has its own `pyproject.toml` with an independent version number.
Sibling distributions at the same layer do not depend on each other (e.g. `user-memory` does not depend on `agent-memory`).

### Subpackage roles

**Product subpackages** — each produces a specific structured memory type:

| Subpackage | Memory types produced |
|---|---|
| `user_memory` | `Episode`, `Foresight`, `AtomicFact`, `Profile` |
| `agent_memory` | `AgentCase`, `AgentSkill` |
| `knowledge` | `KnowledgeMemory` |

**Tool subpackages** — cross-cutting operators consumed by multiple product subpackages:

| Subpackage | Role |
|---|---|
| `boundary` | MemCell boundary detection (chat / workspace / agent) |
| `clustering` | `cluster_by_geometry` / `cluster_by_llm` operating on caller-owned `list[Cluster]` |
| `rank` | 4 retrieval strategies (`hybrid` / `agentic` / `cluster` / `maxsim`) + 4 business facades (`episodic` / `profile` / `case` / `skill`) + algorithm tools (`fusion` / `weight` / `rerank`) |
| `parser` | Multimodal raw-file → `ParsedContent` (OCR, ASR, document layout, URL fetch) |

**Infrastructure subpackages** (all in `everalgo-core`):

| Subpackage | Role |
|---|---|
| `types` | Shared data contracts: `ChatMessage`, `MemCell`, `Episode`, `RankInput`, `RankOutput`, etc. |
| `llm` | `LLMClient` Protocol, `LLMConfig`, provider routing, `LLMError` hierarchy |
| `prompts` | Prompt validator; prompt strings live as module-level constants in each subpackage's `prompts/en/` (and `prompts/zh/` where a package still ships translations — `user_memory` dropped its tree in favour of an `output_language` argument) |
| `testing` | `FakeLLMClient`, `CallRecord`, structural assertion helpers |

---

## 4. Two import paths — same class

Physical layout follows algorithm responsibility (so algorithm engineers can iterate on a specific module without crossing package boundaries).
The public product API is expressed through `__init__.py` re-exports.
Both paths resolve to the same class:

```python
# Product path — used by EverOS and external callers
from everalgo.user_memory import BoundaryDetector, EpisodeExtractor

# Physical path — used by algorithm engineers iterating on boundary logic
from everalgo.boundary.chat import BoundaryDetector
```

The `everalgo.user_memory` package re-exports `BoundaryDetector` from `everalgo.boundary.chat`.
If you change the boundary algorithm, import whichever path is natural for your work — they are identical objects.

---

## 5. Naming convention

| Dimension | Convention | Example |
|---|---|---|
| Distribution (PyPI) | dash-separated | `everalgo-user-memory` |
| Import path | `everalgo.*` namespace + underscore subpackage | `everalgo.user_memory` |
| Physical directory | underscore | `everalgo/user_memory/` |
| Class names | PascalCase | `EpisodeExtractor`, `BoundaryDetector` |
| Async methods | `a` prefix | `aextract`, `adetect`, `arank`, `aparse` |
| Sync methods (pure compute) | no prefix | `rrf`, `count_tokens` |

The `a` prefix is a strict convention for EverAlgo operator methods: a method named `aextract` always performs real I/O and must be `await`-ed; a method without the prefix is always synchronous pure compute. (One deliberate exception: `LLMClient.chat` — a caller-injected client Protocol, not an EverAlgo operator — is async without the `a` prefix, mirroring the OpenAI SDK client interface.)
See [Async–sync bridge](async-sync-bridge.md) for the full contract.

---

## 6. LLM injection — instance binding

Every I/O operator class binds an `LLMClient` at construction time via `llm=` and holds it as `self._llm`. There is no global default, no scoped context manager, and no per-call `llm=` override. The caller constructs the right client and passes it in.

```python
from everalgo.llm.providers.openai_compat import OpenAICompatClient
from everalgo.user_memory import EpisodeExtractor

client = OpenAICompatClient(api_key="sk-...", base_url="https://api.openai.com/v1", model="gpt-4o-mini")
extractor = EpisodeExtractor(llm=client)
episode = await extractor.aextract(memcell, sender_id="u_alice")
```

EverAlgo has **no scene concept** — it does not know which algorithm step should use which model. That mapping lives in EverOS's scene router; EverOS constructs and injects the appropriate client per operator.

---

## 7. Clustering — caller-owned list[Cluster]

The clustering operators use a **caller-wrap pattern**: the caller converts each incoming item into a size-1 `Cluster` and threads the entire `list[Cluster]` through the call. EverAlgo never stores, never mints IDs, and never owns the list.

```python
from everalgo.clustering import Cluster, cluster_by_geometry
import numpy as np

# The caller loads the cluster list from their own storage.
existing: list[Cluster] = await my_store.load(user_id) or []

vector = np.array([0.9, 0.1, 0.0], dtype=np.float32)  # caller computes embedding
new_cluster = Cluster(centroid=vector, last_ts=event_ts_ms)
merged = cluster_by_geometry(new_cluster, existing)

if merged is not None:
    # Update the matching entry in place (caller finds it by merged.id).
    idx = next(i for i, c in enumerate(existing) if c.id == merged.id)
    existing[idx] = merged
else:
    # No match — caller creates a new cluster and stamps its own id.
    existing.append(new_cluster.model_copy(update={"id": my_new_id()}))

await my_store.save(user_id, existing)
```

EverAlgo owns the **merge transition** (weighted centroid, preview concat, members append). The caller owns the list, the IDs, and the persistence.

Two clustering functions cover the two business paths:

- `cluster_by_geometry` — cosine similarity + time-window + threshold, no LLM. Used for user-memory episode clusters.
- `cluster_by_llm` — top-K geometric recall + LLM semantic ranking when the fast path misses. Raises on LLM failure — no internal fallback. Used for agent-memory case clusters.

---

## 8. Package README cross-references

Each distribution has its own README with a quick-start, API surface, and prompt customisation guide:

| Distribution | README |
|---|---|
| `everalgo-core` | [`packages/everalgo-core/README.md`](../../packages/everalgo-core/README.md) |
| `everalgo-boundary` | [`packages/everalgo-boundary/README.md`](../../packages/everalgo-boundary/README.md) |
| `everalgo-clustering` | [`packages/everalgo-clustering/README.md`](../../packages/everalgo-clustering/README.md) |
| `everalgo-rank` | [`packages/everalgo-rank/README.md`](../../packages/everalgo-rank/README.md) |
| `everalgo-parser` | [`packages/everalgo-parser/README.md`](../../packages/everalgo-parser/README.md) |
| `everalgo-user-memory` | [`packages/everalgo-user-memory/README.md`](../../packages/everalgo-user-memory/README.md) |
| `everalgo-agent-memory` | [`packages/everalgo-agent-memory/README.md`](../../packages/everalgo-agent-memory/README.md) |
| `everalgo-knowledge` | [`packages/everalgo-knowledge/README.md`](../../packages/everalgo-knowledge/README.md) |
