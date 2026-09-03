# Architecture

This document describes EverAlgo's structure, the two algorithm axes, the subpackage layout, naming conventions, LLM injection, and the boundary between EverAlgo and its caller (EverOS).

---

## 1. EverAlgo and EverOS

**EverOS** is the AI memory management system: it owns the API gateway, database persistence, orchestration, concurrency control, scene routing (which algorithm step uses which model), and the memory lifecycle.

**EverAlgo** is the algorithm library that EverOS depends on.
It owns no durable business state and has no knowledge of caller storage, deployment topology, or scene routing. Most operators only transform in-memory values. Parser operators additionally support caller-supplied HTTP(S) URLs and use an internal temporary directory for Office conversion; neither path gives EverAlgo ownership of persistent storage.

```
┌──────────────────────────────────────────────────┐
│  EverOS — AI memory management system            │
│  API gateway / persistence / orchestration       │
│  concurrency / scene routing / lifecycle         │
└────────────────┬─────────────────────────────────┘
                 │  in-memory values / injected callables
                 ▼
┌──────────────────────────────────────────────────┐
│  EverAlgo — algorithm library                    │
│  business-stateless · Extract / Retrieve / Rank  │
└──────────────────────────────────────────────────┘
```

**EverAlgo's boundary in one sentence per axis:**

- EverAlgo does not know about the caller's database or index. The caller loads data or injects retrieval functions.
- EverAlgo does not know which LLM to use for which business scenario. The caller selects a client and injects it.
- EverAlgo does not know about concurrency or distributed locks. The caller serialises read-modify-write cycles.
- Parser URL fetches and temporary Office conversion are implementation I/O, not persistence or caller-filesystem ownership.

---

## 2. Two algorithm axes: Extract and Retrieve / Rank

Every product operator in EverAlgo contributes to one of two axes. Both axes are **business-stateless** and **persistence-free**, but they are not all referentially transparent pure functions: LLM calls, caller-injected retrieval functions, URL fetching, and Office conversion perform I/O.

| Axis | When | Input | Output |
|---|---|---|---|
| **Extract** | write path | conversations, `MemCell`, `AgentCase`, or `ParsedContent` supplied by the caller | structured memories and updates: `Episode` / `Foresight` / `AtomicFact` / `Profile` / `AgentCase` / `AgentSkill` / `AgentProfileUpdate` / `KnowledgeMemory` |
| **Retrieve / Rank** | read path | either query + caller-injected `RetrieveFn` / `RerankFn`, or a pre-built `RankInput` | either `list[Candidate]` (plus `AgenticDecision` for agentic retrieval), or `RankOutput` from a business ranker |

**Extract** has several composable product flows rather than one universal two-stage pipeline:

- User conversations: `BoundaryDetector.adetect` produces `MemCell` segments; user-memory extractors produce `Episode`, `Foresight`, `AtomicFact`, or `Profile`, and `EpisodeReflector` can merge episodes.
- Agent trajectories: `AgentBoundaryDetector.adetect` produces mixed-item `MemCell` segments. `AgentCaseExtractor` distils one segment into an `AgentCase`; `AgentSkillExtractor` combines a case with caller-supplied existing skills and supporting cases; `AgentProfileExtractor` screens a segment for config updates.
- Raw files and URLs: `everalgo.parser.aparse` produces `ParsedContent`; `KnowledgeExtractor` produces `KnowledgeMemory`.

**Retrieve / Rank** has two public layers that coexist:

- The retrieval-composition layer exposes `hybrid`, `agentic`, `cluster`, and `maxsim` strategies plus a category-aware wrapper. They consume caller-injected `RetrieveFn` / `RerankFn` callables. Most return ranked `Candidate` lists, `acluster_retrieve` returns an unranked cluster expansion in the caller's `all_docs` order, and agentic retrieval returns `(candidates, AgenticDecision)`. The callbacks may perform storage or model I/O, but EverAlgo does not own those clients.
- The business-ranker layer exposes episodic, profile, case, and skill facades. The caller pre-fetches candidate sets and cross-memory linkage (for example Episode → AtomicFact) into `RankInput`; the facade returns `RankOutput` without accessing storage.

---

## 3. Subpackage layout

EverAlgo is a monorepo of 8 independently versioned PyPI distributions sharing the `everalgo.*` namespace via [PEP 420](https://peps.python.org/pep-0420/).

```
everalgo/                              # PEP 420 namespace package — no __init__.py
├── everalgo-core/                     # types / llm / prompts / testing
├── everalgo-boundary/                 # MemCell extractors + tokenize / split
├── everalgo-clustering/               # cluster_by_geometry / cluster_by_llm over list[Cluster]
├── everalgo-rank/                     # retrieval strategies + business rankers + ranking tools
├── everalgo-parser/                   # multimodal raw-file → ParsedContent
├── everalgo-user-memory/              # Episode / Foresight / AtomicFact / Profile
├── everalgo-agent-memory/             # AgentCase / AgentSkill / AgentProfile updates
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

**Product subpackages** — each produces structured memory or update types:

| Subpackage | Memory types produced |
|---|---|
| `user_memory` | `Episode`, `Foresight`, `AtomicFact`, `Profile` |
| `agent_memory` | `AgentCase`, `AgentSkill`, `AgentProfileUpdate` |
| `knowledge` | `KnowledgeMemory` |

**Tool subpackages** — cross-cutting operators consumed by multiple product subpackages:

| Subpackage | Role |
|---|---|
| `boundary` | Low-level chat boundary primitives plus an unimplemented workspace placeholder; scenario facades live in `user_memory` and `agent_memory` |
| `clustering` | `cluster_by_geometry` / `cluster_by_llm` operating on caller-owned `list[Cluster]` |
| `rank` | 4 retrieval strategies (`hybrid` / `agentic` / `cluster` / `maxsim`) + category-aware retrieval + 4 business facades (`episodic` / `profile` / `case` / `skill`) + algorithm tools (`fusion` / `weight` / `rerank`) |
| `parser` | Multimodal raw-file → `ParsedContent` (OCR, ASR, document layout, URL fetch) |

**Infrastructure subpackages** (all in `everalgo-core`):

| Subpackage | Role |
|---|---|
| `types` | Shared data contracts: `ChatMessage`, `MemCell`, `Episode`, `Candidate`, `RankInput`, `RankOutput`, etc. |
| `llm` | `LLMClient` Protocol, `LLMConfig`, provider routing, `LLMError` hierarchy |
| `prompts` | Prompt validator; prompt strings live as module-level constants in each subpackage's `prompts/en/` (and `prompts/zh/` where a package still ships translations — `user_memory` dropped its tree in favour of an `output_language` argument) |
| `testing` | `FakeLLMClient`, `CallRecord`, structural assertion helpers |

---

## 4. Product facades and low-level primitives

Physical layout follows algorithm responsibility. Product packages expose scenario-specific facade classes, while tool packages expose lower-level primitives. The user-memory package root and its physical module resolve to the same `BoundaryDetector` class; the boundary package exposes a separate function primitive:

```python
# Product facade — used by EverOS and external callers.
from everalgo.user_memory import BoundaryDetector

# Physical module containing that same facade class.
from everalgo.user_memory.boundary import BoundaryDetector

# Lower-level async primitive used by facade implementations.
from everalgo.boundary import detect_boundaries
```

`everalgo.boundary.chat` does not define `BoundaryDetector`; it defines `detect_boundaries`, `adetect_boundary_step`, and their result types. Agent trajectories use the distinct `everalgo.agent_memory.AgentBoundaryDetector` facade because their `MemCell` items may include tool calls and tool results.

---

## 5. Naming convention

| Dimension | Convention | Example |
|---|---|---|
| Distribution (PyPI) | dash-separated | `everalgo-user-memory` |
| Import path | `everalgo.*` namespace + underscore subpackage | `everalgo.user_memory` |
| Physical directory | underscore | `everalgo/user_memory/` |
| Class names | PascalCase | `EpisodeExtractor`, `BoundaryDetector` |
| Native async methods | `a` prefix | `aextract`, `adetect`, `arank`, `aparse` |
| Sync pure-compute functions | no prefix | `rrf`, `count_tokens`, `cluster_by_geometry` |
| Sync bridges over async I/O | no prefix | `extract`, `detect`, `rank`, `parse` |
| Historical native-async exceptions | no prefix | `LLMClient.chat`, `detect_boundaries`, `cluster_by_llm` |

The `a` prefix is a one-way calling-convention marker: every such name is native async and must be awaited. Most names without the prefix are synchronous, either pure compute or a blocking bridge created with `async_to_sync`; never call a sync bridge from a running event loop. The three tabled historical exceptions are also native async and must be awaited.
See [Async–sync bridge](async-sync-bridge.md) for the full contract.

---

## 6. LLM injection — instance binding

Every I/O operator class binds an `LLMClient` at construction time via `llm=` and holds it as `self._llm`. There is no global default, no scoped context manager, and no per-call `llm=` override. The caller constructs the right client and passes it in.

```python
from everalgo.llm import LLMConfig
from everalgo.llm.providers.openai_compat import OpenAICompatClient
from everalgo.user_memory import EpisodeExtractor

client = OpenAICompatClient(
    LLMConfig(model="gpt-4o-mini", api_key="sk-...", base_url="https://api.openai.com/v1")
)
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
