# everalgo-clustering

Online incremental clustering for EverAlgo — two async functions (`cluster_by_geometry` / `cluster_by_llm`) operating on caller-owned `list[Cluster]` state.

Stateless: the package never embeds, never queries storage, never holds a lock. The caller owns embedding computation, persistence (`list[Cluster]` serialisation), and read-modify-write coordination across concurrent writers.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the architecture document at [`docs/concepts/architecture.md`](../../docs/concepts/architecture.md).

## Install

```bash
pip install everalgo-clustering
```

## What this distribution provides

| Symbol | Role |
|---|---|
| `Cluster` | Frozen Pydantic value object — one cluster snapshot. Caller-supplied `id` / `members`; algorithm supplies merged `centroid`, `count`, `last_ts`, `preview`. |
| `cluster_by_geometry` | Cosine similarity + time-window filter + threshold; no LLM; async |
| `cluster_by_llm` | Top-K geometric recall → fast-path skip → LLM semantic ranking; raises on LLM failure; async |

## Cluster type

```python
class Cluster(BaseModel):
    id: str | None = None           # caller-supplied business id; algo only passes through, never mints
    centroid: np.ndarray
    count: int = 1
    last_ts: int                    # Unix epoch milliseconds
    preview: list[str] = []
    members: list[str] = []         # caller-supplied entity ids; algo appends on merge, never inspects
```

The caller wraps each incoming item as a size-1 `Cluster` (count=1) before passing to either function.
Both functions return `Cluster | None`: a merged snapshot when the item is assigned to an existing cluster, or `None` when no match — the caller then appends the original size-1 `Cluster` as a brand-new entry and mints its own `id`.

## Quick start

```python
import asyncio
import numpy as np
from everalgo.clustering import Cluster, cluster_by_geometry

async def main() -> None:
    existing: list[Cluster] = []  # caller loads from storage; empty on first run
    vector = np.random.rand(2560).astype(np.float32)
    timestamp_ms = 1_700_000_000_000

    new_cluster = Cluster(centroid=vector, last_ts=timestamp_ms)
    merged = await cluster_by_geometry(
        new_cluster,
        existing,
        threshold=0.65,        # cosine similarity floor
        time_window_days=7.0,  # ignore clusters older than this window
    )

    if merged is not None:
        # item assigned to an existing cluster; caller updates the matching entry
        print(f"merged into cluster id={merged.id!r}, new count={merged.count}")
    else:
        # no match — caller appends new_cluster and stamps its own id
        new_cluster_with_id = new_cluster.model_copy(update={"id": "cid_001"})
        existing.append(new_cluster_with_id)
        print("created new cluster")

asyncio.run(main())
```

## LLM-refined clustering

`cluster_by_llm` adds a semantic ranking step over the top-K geometrically-nearest candidates. It **raises** on LLM failure — there is no internal fallback; the caller decides whether to retry or fall back to `cluster_by_geometry`.

Populate `new_cluster.preview` with the item's representative text so the LLM has something to rank against.

```python
import asyncio
import numpy as np
from everalgo.clustering import Cluster, cluster_by_llm
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient

_LLM_JSON = '{"idx": 0}'

async def main() -> None:
    fake = FakeLLMClient(responses=[ChatResponse(content=_LLM_JSON, model="fake")])
    existing: list[Cluster] = []
    vector = np.random.rand(2560).astype(np.float32)

    new_cluster = Cluster(
        centroid=vector,
        last_ts=1_700_000_000_000,
        preview=["Python async retry patterns"],  # shown to the LLM
    )
    merged = await cluster_by_llm(
        new_cluster,
        existing,
        llm=fake,
        k_candidates=30,
        llm_skip_threshold=0.85,
    )
    print(f"merged: {merged}")

asyncio.run(main())
```

## Persistence pattern

`Cluster` is a frozen Pydantic model — serialise with `model_dump()` and reconstruct with `Cluster.model_validate()`. The caller owns the list and the lock:

```python
raw_list = await store.load(user_id) or []
clusters = [Cluster.model_validate(r) for r in raw_list]

async with caller.lock(f"cluster:{user_id}"):
    merged = await cluster_by_geometry(new_cluster, clusters)
    if merged is not None:
        idx = next(i for i, c in enumerate(clusters) if c.id == merged.id)
        clusters[idx] = merged
    else:
        new_cluster_stamped = new_cluster.model_copy(update={"id": new_id})
        clusters.append(new_cluster_stamped)
    await store.save(user_id, [c.model_dump() for c in clusters])
```

## API reference

```python
async def cluster_by_geometry(
    new_cluster: Cluster,
    existing_clusters: list[Cluster],
    *,
    threshold: float = 0.65,
    time_window_days: float = 7.0,
    preview_cap: int = 5,
) -> Cluster | None: ...

async def cluster_by_llm(
    new_cluster: Cluster,
    existing_clusters: list[Cluster],
    *,
    llm: LLMClient,
    k_candidates: int = 30,
    llm_skip_threshold: float = 0.85,
    prompt: str | None = None,
    preview_cap: int = 5,
) -> Cluster | None: ...
```

Both functions return the **merged** `Cluster` (existing cluster's `id` preserved, centroid/count/members updated) or `None` (no match — caller creates a new cluster entry and mints its own `id`).

Tested embedding model: `Qwen3-Embedding-4B` (2560-dim float32). Any consistent-dimension embedding works; EverAlgo does not import or manage embedding SDKs.

## Related distributions

- [`everalgo-user-memory`](../everalgo-user-memory/) — uses `cluster_by_geometry` in the full user-memory pipeline
- [`everalgo-agent-memory`](../everalgo-agent-memory/) — uses `cluster_by_llm` for skill clustering
