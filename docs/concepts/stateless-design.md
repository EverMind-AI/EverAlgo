# Stateless Design

EverAlgo operators are business-stateless transformations, not universally referentially transparent pure functions. They do not own durable business state, connect to caller databases or indexes, or accumulate results between invocations. Operators that invoke an LLM, fetch a URL, call a caller-injected retrieval function, or convert an Office document can perform I/O and need not produce byte-identical output for the same input.

---

## What "stateless" means in practice

**No caller storage ownership.**
Extractors and rankers never open the caller's database or query its search index directly. The caller prepares in-memory data or injects `RetrieveFn` / `RerankFn` callbacks. Parser operators may fetch caller-supplied HTTP(S) URLs, and Office conversion uses a managed temporary directory; local `file://` inputs and arbitrary caller filesystem paths are rejected.

**No mutable global business state.**
An `EpisodeExtractor` instance carries no per-user context between calls.
It holds only its injected `LLMClient`; business inputs such as `MemCell`, `sender_id`, and prior memories are provided per invocation. Concurrency safety of the injected client remains the caller's responsibility.

**Business inputs and outputs are in-memory data structures.**
`ChatMessage`, `MemCell`, `Episode`, `Candidate`, `RankInput`, and `Cluster` are Pydantic models. Serialization to and from durable storage is always the caller's responsibility, even when an operator internally performs transient network or temporary-file I/O.

---

## Why this matters for the caller

Because EverAlgo is stateless, the caller (EverOS or any other orchestrator) retains full control:

- **When to call.** The pipeline can fan out steps in parallel or run them sequentially — EverAlgo imposes no ordering constraints.
- **Which durable storage backend to use.** MongoDB, Redis, a local file, an in-memory dict for tests — the operator does not care.
- **How to handle concurrency.** If multiple writer coroutines update the same user's `list[Cluster]`, the caller applies its own distributed lock around the read-modify-write cycle. EverAlgo's frozen `Cluster` value objects make this safe: a failed write leaves the original list intact.
- **Whether to retry.** EverAlgo does not add a retry layer on top of the LLM client. The underlying provider SDK already retries transient failures. If the caller needs cross-provider fallback or multi-key rotation, it wraps the `LLMClient` Protocol with its own decorator — EverAlgo operators call the decorated wrapper transparently.

---

## Operators are algorithm IP, not business logic

EverAlgo lives in the same category as NumPy, scikit-learn, and PyTorch — not LangChain or LlamaIndex.
Those frameworks are end-to-end application builders; they own chains, agents, and state.
EverAlgo provides the **algorithm IP** (extraction strategies, ranking math, clustering decision logic) and nothing else.

Operators do not know:

- How a supplied user ID maps to tenancy or storage (some accept `sender_id` / `owner_id` as labels but never query data by them)
- Which LLM scene is currently active (the caller injects the right client when constructing a class facade or calling a function operator)
- Whether the system is running in a single-tenant or multi-tenant deployment

---

## Clustering: caller-owned list, frozen Cluster

The clustering operators follow a caller-wrap pattern. The caller owns the `list[Cluster]`; EverAlgo's operators take the list as input and return a merged `Cluster | None` without mutating the original:

```python
merged = cluster_by_geometry(new_cluster, existing_clusters)
# existing_clusters is never mutated; caller decides whether to update its list
```

Each `Cluster` is a frozen Pydantic model. When a merge happens, the operator returns a brand-new `Cluster` with updated centroid, count, and members — the original remains intact. If the call fails, the caller's list is unchanged.

This is the same pattern used by Redux and Python's `frozenset.union`: the merge logic lives in the library; load, persist, and lock decisions live in the caller.

---

## Concrete implications

**Testing is easy.**
Because operators do not access caller storage directly, unit testing requires no database mocking.
Swap the real `LLMClient` for `FakeLLMClient`, pass a `MemCell`, check the returned `Episode`.

```python
import json
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.user_memory import EpisodeExtractor

fake = FakeLLMClient(
    responses=[
        ChatResponse(
            content=json.dumps({"title": "Test", "content": "Summary.", "summary": "Preview."}),
            model="fake",
        )
    ]
)
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
