# Async–Sync Bridge

EverAlgo operators follow a strict naming convention to distinguish async I/O methods from synchronous pure-compute methods.

---

## The `a`-prefix rule

**Methods with an `a` prefix** — `aextract`, `adetect`, `arank`, `aparse` — are **native async**.
They perform real I/O (LLM calls, OCR, ASR, URL fetch) and must be called with `await`.

**Methods without an `a` prefix** — `extract`, `rank`, `count_tokens`, `rrf` — are **synchronous**.
They are pure-compute (no I/O) and must be called without `await`.

```python
# ✅ Correct — async I/O method, use await
episode = await EpisodeExtractor(llm=client).aextract(memcell, sender_id="u_alice")

# ✅ Correct — sync pure-compute method, no await
merged  = cluster_by_geometry(new_cluster, existing_clusters)
fused   = rank.fusion.rrf(vec_hits, keyword_hits)
n       = boundary._tokenize.count_tokens(text)
```

Reading the method name tells you its calling convention immediately — no need to look up documentation.
This is the same convention used by `dspy.acall` / `dspy.aforward`, `litellm.acompletion`, and `instructor.AsyncInstructor`.

**Exception — `LLMClient`.** The `a`-prefix rule governs EverAlgo *operator* methods. The `LLMClient` Protocol (`everalgo.llm.protocols`) is a caller-injected client interface, not an operator; its single method is `async def chat(...)` — async but without the `a` prefix — to mirror the OpenAI SDK client surface. EverAlgo operators call it internally via `await self._llm.chat(...)`.

---

## Why pure-compute methods are not async

Python's `asyncio` is designed for I/O-bound concurrency, not for CPU-bound work.
Wrapping a millisecond-scale computation in `async def` adds overhead and forces callers to `await` a result that was ready instantly.

NumPy, SciPy, scikit-learn, and PyTorch all follow the same rule: pure-compute functions are synchronous.
EverAlgo's fusion math (`rrf`, `cosine_to_lr_score`, `score_propagation`), token counting, and clustering geometry calculations are in this category.

If a specific computation grows large enough to block the event loop (roughly above ~100 ms), the caller wraps it with `asyncio.run_in_executor` — the operator's API stays synchronous.

---

## The sync bridge for I/O operators

Every I/O operator provides a synchronous wrapper as a convenience for non-event-loop callers (CLI scripts, plain unit tests).
The wrapper is derived from the async implementation using `asgiref.sync.async_to_sync`, which means there is exactly one implementation to maintain.

```python
# In everalgo/user_memory/episode.py (simplified)
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
| `cluster_by_geometry` | `cluster_by_geometry` | sync pure-compute | `cluster_by_geometry(new_cluster, existing_clusters)` |
| `rank.episodic.arank` | `arank` | async I/O | `await rank.episodic.arank(rank_input)` |
| `rank.profile.rank` | `rank` | sync pure-compute | `rank.profile.rank(rank_input)` |
| `rank.fusion.rrf` | `rrf` | sync pure-compute | `rank.fusion.rrf(hits_a, hits_b)` |

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
n_tokens = boundary._tokenize.count_tokens(text)
```

The naming convention ensures there is no ambiguity even when both styles appear in the same function.
