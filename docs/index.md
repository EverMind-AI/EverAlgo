# EverAlgo

EverAlgo is the **algorithm library** for memory extraction and ranking used by [EverOS](https://github.com/EverMind-AI/everos), EverMind's open-source AI memory framework.

It is **stateless and storage-free**: every operator takes in-memory data structures and returns in-memory data structures.
No database connections, no filesystem reads, no business-state ownership — those responsibilities belong to EverOS, the orchestration layer that calls EverAlgo.

---

## What EverAlgo does

EverAlgo provides two symmetric algorithm axes:

**Extract (write path)** — Turn raw conversation messages or structured inputs into typed memories.

```python
from everalgo.user_memory import BoundaryDetector, EpisodeExtractor
from everalgo.testing.fake_llm import FakeLLMClient

fake = FakeLLMClient(responses=["..."])
result = await BoundaryDetector(llm=fake).adetect(messages, is_final=True)
episode = await EpisodeExtractor(llm=fake).aextract(result.cells[0], sender_id="u_alice")
```

**Rank (read path)** — Re-rank multi-route recall candidates for a retrieval response.

```python
from everalgo import rank

# Synchronous pure-compute fusion (no LLM needed)
merged = rank.fusion.rrf(vec_hits, keyword_hits)
```

---

## What EverAlgo does not do

- Connect to any database or filesystem
- Own retry logic, multi-key rotation, or provider fallback
- Route requests to different LLM models by business scene (that is EverOS's `SceneRouter`)
- Compute embeddings (the caller computes and passes vectors in)

---

## Distributions

EverAlgo is a monorepo of 8 independently versioned [PyPI](https://pypi.org/) distributions sharing the `everalgo.*` namespace via [PEP 420](https://peps.python.org/pep-0420/).
Install only what your use case needs:

| Distribution | Purpose |
|---|---|
| `everalgo-core` | Types, LLM client + providers, prompt validators, testing helpers |
| `everalgo-boundary` | `MemCell` boundary detection (chat / workspace / agent) |
| `everalgo-clustering` | `Cluster` value object + `cluster_by_geometry` / `cluster_by_llm` operators |
| `everalgo-rank` | 4 retrieval strategies (hybrid / agentic / cluster / maxsim) + 4 business rankers (episodic / profile / case / skill) + fusion / weight / rerank tools |
| `everalgo-parser` | Multimodal raw-file → `ParsedContent` |
| `everalgo-user-memory` | `Episode` / `Foresight` / `AtomicFact` / `Profile` extractors |
| `everalgo-agent-memory` | `AgentCase` / `AgentSkill` extractors |
| `everalgo-knowledge` | `KnowledgeMemory` extractor |

---

## Where to start

| Goal | Document |
|---|---|
| Install and run a first example in 5 minutes | [Getting started](getting-started.md) |
| Understand the full architecture | [Architecture](concepts/architecture.md) |
| Install options and prerequisites | [Installation](installation.md) |
| Understand why operators are pure functions | [Stateless design](concepts/stateless-design.md) |
| Understand the `a`-prefix async convention | [Async–sync bridge](concepts/async-sync-bridge.md) |
| Browse per-package API docs | [API index](api/README.md) |
| Versioning and support policy | [Version policy](version-policy.md) |
| Contribute to EverAlgo | [Contributing](contributing.md) |

---

## Quick install

```bash
# Full user-memory pipeline (boundary + 4 extractors)
pip install everalgo-user-memory

# Agent memory
pip install everalgo-agent-memory

# Retrieval re-ranking only
pip install everalgo-rank

# Multimodal knowledge extraction
pip install everalgo-knowledge
```

Development install (all 8 packages, editable):

```bash
git clone git@github.com:EverMind-AI/EverAlgo.git
cd everalgo
uv sync --all-packages
```

---

## License

Apache License 2.0. See [LICENSE](../LICENSE).
