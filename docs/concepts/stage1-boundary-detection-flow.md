# Stage 1 Extract-Base Flow

Stage 1 is the benchmark's extract-base stage. For each conversation it performs incremental boundary detection, extracts one Episode per MemCell, embeds the Episode body and subject, and clusters Episodes geometrically. The canonical stage ordering is `_STAGE_RUNNERS` in `benchmarks/common/runner.py`.

This document describes the benchmark orchestration, not a universal EverAlgo product pipeline. The reusable operators remain stateless; the benchmark owns concurrency, artifact files, IDs, and accumulated cluster state.

## Flow

```text
dataset conversation
        │
        ▼
convert messages to everalgo.types.ChatMessage
        │
        ▼
incremental BoundaryDetector.adetect_step
        │
        ▼
list[MemCell]
        │
        ├── in parallel: EpisodeExtractor.aextract(sender_id=None)
        │                  ├── embed Episode.episode
        │                  └── embed Episode.subject when non-empty
        │
        ▼
MemCell artifacts + Episode artifacts
        │
        ▼
sequential cluster_by_geometry over Episode embeddings
        │
        ▼
cluster artifacts + reverse Episode-to-cluster map
```

## 1. Incremental boundary detection

`_detect_all_boundaries` in `benchmarks/common/stages/extract.py` keeps a per-conversation `history` buffer. The first two messages are buffered without an LLM call. Each later message is passed with the current history to `BoundaryDetector.adetect_step`; closed cells are appended and the returned tail becomes the next history. After input is exhausted, any remaining history is force-closed as the final MemCell with the last message timestamp.

The result is an ordered `list[MemCell]`. At this point a MemCell contains only boundary-segmented conversation items and a timestamp; it does not embed an Episode or AtomicFacts.

## 2. Episode extraction and embeddings

`_extract_one_conversation` runs `_extract_memcell_data` for all MemCells with `asyncio.gather`, bounded by a shared semaphore. Each task:

1. Calls `EpisodeExtractor(llm=llm).aextract(memcell, sender_id=None)`.
2. Fails if the returned Episode body is empty.
3. Embeds the Episode body and, when present, its subject in parallel.
4. Serializes the MemCell and Episode as separate entities linked through `memcell_ids`.

`sender_id=None` is intentional for this benchmark: the Episode represents the whole conversation segment rather than one participant.

AtomicFact extraction is not part of Stage 1. Stage 3 (`enrich.py`) extracts and embeds AtomicFacts from the final Episode set after optional Stage 2 reflection.

## 3. Geometric Episode clustering

Episodes are processed sequentially because each assignment depends on the accumulated caller-owned `list[Cluster]`. `_cluster_one_episode` wraps the Episode vector, timestamp, ID, and body preview in a size-one `Cluster`, then calls `cluster_by_geometry`.

The defaults in `benchmarks/common/config.py` are cosine-similarity threshold `0.70` and maximum time gap `7.0` days. A match replaces the existing frozen Cluster with the merged value; a miss creates a benchmark-owned ID such as `cluster_0`.

The cluster artifact stores Episode IDs and an `episode_to_cluster` reverse map. It does not store MemCells inside Cluster objects.

## 4. Artifacts

For each conversation index `<i>`, Stage 1 writes:

| Artifact | Contents |
|---|---|
| `memcells_conv_<i>.json` | Pure boundary segments: conversation items and timestamps |
| `episodes_conv_<i>.json` | Episode entities, `memcell_ids`, body / subject embeddings |
| `clusters_conv_<i>.json` | Geometric clusters with `episode_ids` and `episode_to_cluster` |
| `stats_conv_<i>.json` | Counts and estimated token usage |

Boundary or Episode extraction failures are isolated per conversation in `memcells_conv_<i>.error.txt`; clustering failures propagate because they indicate an invalid artifact contract. Downstream stages may replace the Episode set through reflection and enrichment, but Stage 1's MemCell artifacts remain the original boundary output.

## Source of truth

- Stage registry: `benchmarks/common/runner.py::_STAGE_RUNNERS`
- Boundary and extraction orchestration: `benchmarks/common/stages/extract.py::_detect_all_boundaries`, `_extract_memcell_data`, `_extract_one_conversation`
- Clustering and artifacts: `benchmarks/common/stages/extract.py::_cluster_one_episode`, `_run_clustering_pass`, `_process_conversation`
- Entity schemas: `benchmarks/common/stages/serialization.py`
- AtomicFact stage: `benchmarks/common/stages/enrich.py`
