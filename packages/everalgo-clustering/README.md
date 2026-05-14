# everalgo-clustering

Online incremental clustering for EverAlgo — two public async functions
(`cluster_by_geometry` / `cluster_by_llm`) operating on a frozen
`ClusterState` value object the caller threads through and persists.

Stateless: the package never embeds, never queries storage, never holds a
lock. Callers own embedding (any model, any dimension as long as it stays
consistent across calls), serialisation (`ClusterState.to_dict` /
`from_dict`), pre-fetching recent text for `cluster_by_llm`, and serialising
read-modify-write across concurrent writers.

See the umbrella project: [EverAlgo monorepo](../../README.md) and the
architecture document at [`docs/design.md`](../../docs/design.md) §2.4.

## Quick start

Two-step pipeline: cluster the incoming `MemCell`, then hand the prior
cluster's `MemCell`s to a downstream extractor (here `ProfileExtractor`).

```python
import numpy as np

from everalgo.clustering import (
    ClusterConfig, ClusterState, cluster_by_geometry,
)
from everalgo.user_memory import ProfileExtractor

# Caller-owned state and per-cluster memcell index.
state = ClusterState.empty()
cid_to_memcells: dict[str, list[MemCell]] = {}
config = ClusterConfig()  # threshold=0.65, time_window_days=7

extractor = ProfileExtractor()

for memcell in benchmark_stream:                          # 1. caller has memcells
    vector = embed(memcell)                               # 2. caller embeds
    cid, state = await cluster_by_geometry(               # 3. assign cluster
        vector, memcell.timestamp, state, config=config,
    )
    prior = cid_to_memcells.setdefault(cid, [])           # 4. index by cluster
    cid_to_memcells[cid] = [*prior, memcell]

    profile = await extractor.aextract(                   # 5. downstream
        memcell, cluster_episodes=prior, llm=llm,
    )
```

### LLM-refined clustering

`cluster_by_llm` adds an embedding top-K recall stage and an LLM ranking
step, with a deterministic geometric fallback if the LLM fails:

```python
from everalgo.clustering import cluster_by_llm

def cluster_text(memcell: MemCell) -> str:
    """Caller-defined: distil the memcell into one string for the LLM prompt.

    Typical sources: the ``task_intent`` of an agent case, the ``episode`` body
    after EpisodeExtractor has run, or just the concatenated user-message
    contents.
    """
    return "\n".join(m.content for m in memcell.messages if m.content)

# Caller pre-fetches recent text per candidate cluster so EverAlgo never
# queries storage.
previews_text = {
    cid: [cluster_text(m) for m in cid_to_memcells[cid][-5:]]
    for cid in state.centroids
}

cid, state = await cluster_by_llm(
    vector, memcell.timestamp, cluster_text(memcell),
    state, config=config, llm=llm, cluster_previews=previews_text,
)
```

LLM exhaustion (3 retries on bad JSON / schema, or any LLM exception)
falls back to "top-1 if cosine ≥ `config.threshold`, otherwise new cluster".
The function **never raises** on LLM failure — the geometric fallback is the
hard contract.

## Persistence

`ClusterState` is a frozen pydantic model. Serialise with `to_dict()`,
restore with `from_dict()`. Numpy centroids serialise as plain
`list[float]`; vector dtype is normalised to `float32` on load.

```python
raw = await store.load(user_id)
state = ClusterState.from_dict(raw) if raw else ClusterState.empty()

async with caller.lock(f"cluster:{user_id}"):
    cid, new_state = await cluster_by_geometry(
        vector, memcell.timestamp, state, config=config,
    )
    await store.save(user_id, new_state.to_dict())
```

The caller picks the backing store (MongoDB / Redis / file / SQLite).
EverAlgo only sees the dict.

## Reference

| Symbol | Role |
|---|---|
| `ClusterState` | Frozen accumulator — `centroids` / `counts` / `last_ts` (ms int) / `next_idx`. |
| `ClusterConfig` | Threshold bundle — `threshold` / `time_window_days` / `k_candidates` / `llm_skip_threshold`. |
| `cluster_by_geometry` | Cosine + time-window + threshold. No LLM. |
| `cluster_by_llm` | Embedding top-K → fast-path → LLM rank → geometric fallback. |

Tested embedding model: `Qwen3-Embedding-4B` (2560-dim float32). Any embedding
model with consistent output dimension works; EverAlgo does not import or
manage embedding SDKs.
