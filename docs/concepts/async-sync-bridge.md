# Async–Sync Bridge

EverAlgo operators follow a strict naming convention that identifies the calling convention. It distinguishes native async APIs from synchronous APIs; a synchronous API may be either pure compute or a blocking bridge over async I/O.

---

## The `a`-prefix rule

**Methods with an `a` prefix** — `aextract`, `adetect`, `arank`, `aparse` — are **native async**.
They may perform or orchestrate I/O (LLM calls, caller-injected retrieval, OCR, ASR, URL fetch) and must be called with `await`.

**Most methods without an `a` prefix** — `extract`, `rank`, `count_tokens`, `rrf` — are **synchronous**.
Call those without `await`, then distinguish two categories:

- Pure compute such as `count_tokens`, `rrf`, and `cluster_by_geometry` returns directly and performs no I/O.
- Sync bridges such as `extract`, `detect`, `rank`, and `parse` block while running a native async implementation through `asgiref.async_to_sync`; use them only outside a running event loop.

```python
# ✅ Correct — async I/O method, use await
episode = await EpisodeExtractor(llm=client).aextract(memcell, sender_id="u_alice")

# ✅ Correct — sync pure-compute method, no await
merged  = cluster_by_geometry(new_cluster, existing_clusters)
fused   = rank.fusion.rrf(vec_hits, keyword_hits)
n       = everalgo.utils.count_tokens(text)

# ✅ Correct — blocking sync bridge, only outside a running event loop
episode = EpisodeExtractor(llm=client).extract(memcell, sender_id="u_alice")
```

An `a` prefix always tells you to use `await`; operators and the `LLMClient` contract have no prefix-less async names. For synchronous names, the name alone does not tell you whether the implementation is pure compute or a blocking bridge.
This is the same convention used by `dspy.acall` / `dspy.aforward`, `litellm.acompletion`, and `instructor.AsyncInstructor`.

**No historical exceptions.** `LLMClient.achat`, `adetect_boundaries` and `acluster_by_llm` once lacked the prefix (`chat`, `detect_boundaries`, `cluster_by_llm`); all three now carry it. The two functions keep their old names as sync bridges. The protocol method gets no bridge: `LLMClient` is what a host implements, a sync twin would double that work, and operators only ever call it from async code.

---

## Why pure-compute methods are not async

Python's `asyncio` is designed for I/O-bound concurrency, not for CPU-bound work.
Wrapping a millisecond-scale computation in `async def` adds overhead and forces callers to `await` a result that was ready instantly.

NumPy, SciPy, scikit-learn, and PyTorch all follow the same rule: pure-compute functions are synchronous.
EverAlgo's fusion math (`rrf`, `cosine_to_lr_score`, `score_propagation`), token counting, and clustering geometry calculations are in this category.

If a specific computation grows large enough to block the event loop (roughly above ~100 ms), the caller wraps it with `asyncio.run_in_executor` — the operator's API stays synchronous.

---

## Sync bridges where exposed

Most high-level I/O operators provide synchronous wrappers as a convenience for non-event-loop callers (CLI scripts, plain unit tests).
The wrapper is derived from the async implementation using `asgiref.sync.async_to_sync`, which means there is exactly one implementation to maintain.

```python
# In everalgo/user_memory/episode/_extractor.py (simplified)
from asgiref.sync import async_to_sync

class EpisodeExtractor:
    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(self, memcell, *, sender_id):
        ...

    extract = async_to_sync(aextract)
```

**The sync bridge is only safe in non-event-loop contexts.**
Do not call `extract(...)` inside a running event loop (FastAPI handler, `async def` function, Jupyter cell after `asyncio.run` has started).
In those contexts, always use `await aextract(...)`.

Calling the sync bridge from within a running event loop will raise a `RuntimeError` from `asgiref`.

---

## Quick reference

| Operator | Method | Type | Calling convention |
|---|---|---|---|
| `EpisodeExtractor` | `aextract` | async I/O | `await extractor.aextract(mc, sender_id=...)` |
| `EpisodeExtractor` | `extract` | sync bridge | `extractor.extract(mc, sender_id=...)` — non-event-loop only |
| `BoundaryDetector` | `adetect` | async I/O | `await detector.adetect(messages, is_final=True)` |
| `BoundaryDetector` | `detect` | sync bridge | `detector.detect(messages, is_final=True)` — non-event-loop only |
| `BoundaryDetector` | `adetect_step` | async I/O | `await detector.adetect_step(history, new)` |
| `BoundaryDetector` | `detect_step` | sync bridge | `detector.detect_step(history, new)` — non-event-loop only |
| `adetect_boundaries` | `adetect_boundaries` | async I/O | `await adetect_boundaries(messages, llm=..., is_final=True)` |
| `detect_boundaries` | `detect_boundaries` | sync bridge | `detect_boundaries(messages, llm=..., is_final=True)` — non-event-loop only |
| `cluster_by_geometry` | `cluster_by_geometry` | sync pure-compute | `cluster_by_geometry(new_cluster, existing_clusters)` |
| `acluster_by_llm` | `acluster_by_llm` | async I/O | `await acluster_by_llm(new_cluster, existing_clusters, llm=...)` |
| `cluster_by_llm` | `cluster_by_llm` | sync bridge | `cluster_by_llm(new_cluster, existing_clusters, llm=...)` — non-event-loop only |
| `LLMClient` | `achat` | async I/O, no bridge | `await llm.achat(messages, response_format=Model)` |
| `rank.episodic.arank` | `arank` | async I/O | `await rank.episodic.arank(rank_input)` |
| `rank.profile.rank` | `rank` | sync pure-compute | `rank.profile.rank(rank_input)` |
| `rank.fusion.rrf` | `rrf` | sync pure-compute | `rank.fusion.rrf(hits_a, hits_b)` |
| `ahybrid_retrieve` | `ahybrid_retrieve` | async caller-I/O orchestration | `await ahybrid_retrieve(query, dense_retrieve=..., sparse_retrieve=...)` |
| `hybrid_retrieve` | `hybrid_retrieve` | sync bridge | `hybrid_retrieve(query, dense_retrieve=..., sparse_retrieve=...)` — non-event-loop only |

---

## Mixing sync and async in the same pipeline

A typical pipeline has both kinds of calls:

```python
# All I/O operators — use await
result   = await BoundaryDetector(llm=llm).adetect(messages, is_final=True)
mc       = result.cells[0]
episode  = await EpisodeExtractor(llm=llm).aextract(mc, sender_id="u_alice")

# Pure-compute — no await
merged   = cluster_by_geometry(new_cluster, existing_clusters)
fused    = rank.fusion.rrf(vec_hits, keyword_hits)
n_tokens = everalgo.utils.count_tokens(text)
```

The convention covers the current high-level APIs; consult the quick-reference exceptions for the two legacy function names.
