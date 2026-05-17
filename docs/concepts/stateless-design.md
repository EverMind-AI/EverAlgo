# Stateless Design

EverAlgo operators are pure functions: given the same inputs, they produce the same outputs.
They hold no business state, make no storage calls, and do not accumulate results between invocations.

---

## What "stateless" means in practice

**No database connections.**
An extractor never opens a database connection, never queries a search index, and never reads the filesystem.
The caller prepares all necessary data and passes it as function arguments.

**No mutable global state.**
An `EpisodeExtractor` instance carries no per-user context between calls.
You can safely call `EpisodeExtractor().aextract(memcell, sender_id="u_alice")` from multiple concurrent coroutines without any locking.

**Inputs and outputs are in-memory data structures.**
`ChatMessage`, `MemCell`, `Episode`, `RankInput`, `Cluster` — these are plain Pydantic models.
Serialization to/from a storage system is always the caller's responsibility.

---

## Why this matters for the caller

Because EverAlgo is stateless, the caller (evermem or any other orchestrator) retains full control:

- **When to call.** The pipeline can fan out steps in parallel or run them sequentially — EverAlgo imposes no ordering constraints.
- **Which storage backend to use.** MongoDB, Redis, a local file, an in-memory dict for tests — the operator does not care.
- **How to handle concurrency.** If multiple writer coroutines update the same user's `list[Cluster]`, the caller applies its own distributed lock around the read-modify-write cycle. EverAlgo's frozen `Cluster` value objects make this safe: a failed write leaves the original list intact.
- **Whether to retry.** EverAlgo does not add a retry layer on top of the LLM client. The underlying provider SDK already retries transient failures. If the caller needs cross-provider fallback or multi-key rotation, it wraps the `LLMClient` Protocol with its own decorator — EverAlgo operators call the decorated wrapper transparently.

---

## Operators are algorithm IP, not business logic

EverAlgo lives in the same category as NumPy, scikit-learn, and PyTorch — not LangChain or LlamaIndex.
Those frameworks are end-to-end application builders; they own chains, agents, and state.
EverAlgo provides the **algorithm IP** (extraction strategies, ranking math, clustering decision logic) and nothing else.

Operators do not know:

- Which user ID they are running for (some accept `sender_id` as a labelling hint, but never query for data by it)
- Which LLM scene is currently active (the caller injects the right client per call)
- Whether the system is running in a single-tenant or multi-tenant deployment

---

## Clustering: caller-owned list, frozen Cluster

The clustering operators follow a caller-wrap pattern. The caller owns the `list[Cluster]`; EverAlgo's operators take the list as input and return a merged `Cluster | None` without mutating the original:

```python
merged = await cluster_by_geometry(new_cluster, existing_clusters)
# existing_clusters is never mutated; caller decides whether to update its list
```

Each `Cluster` is a frozen Pydantic model. When a merge happens, the operator returns a brand-new `Cluster` with updated centroid, count, and members — the original remains intact. If the call fails, the caller's list is unchanged.

This is the same pattern used by Redux and Python's `frozenset.union`: the merge logic lives in the library; load, persist, and lock decisions live in the caller.

---

## Concrete implications

**Testing is easy.**
Because every operator is a pure function over in-memory data, testing requires no database mocking.
Swap the real `LLMClient` for `FakeLLMClient`, pass a `MemCell`, check the returned `Episode`.

```python
import json
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.user_memory import EpisodeExtractor

fake = FakeLLMClient(responses=[ChatResponse(content=json.dumps({"title": "Test", "content": "Summary."}), model="fake")])
episode = await EpisodeExtractor(llm=fake).aextract(memcell, sender_id="u_alice")
assert episode.subject == "Test"
```

**Parallelism is safe.**
Multiple extractors can run concurrently for the same `MemCell` — they share nothing:

```python
episode, foresights, facts = await asyncio.gather(
    EpisodeExtractor(llm=llm).aextract(mc, sender_id="u_alice"),
    ForesightExtractor(llm=llm).aextract(mc, sender_id="u_alice"),
    AtomicFactExtractor(llm=llm).aextract(mc, sender_id="u_alice"),
)
```

**Prompt customisation does not require a framework.**
Prompts are plain Python string constants.
Override a prompt for a single call with the `prompt=` argument; override globally with a one-time monkey-patch at startup.
No plugin system, no configuration DSL needed.
