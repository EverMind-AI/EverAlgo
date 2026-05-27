# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-27

> First archived changelog. Entries below accumulated since the initial `0.1.0` PyPI release (published manually, without a git tag or per-package changelog), so this `0.2.0` section also consolidates the previously-unarchived `0.1.0` surface.

### Added

- `cluster_by_geometry(new_cluster: Cluster, existing_clusters: list[Cluster], *, threshold=0.65, time_window_days=7.0, preview_cap=5) -> Cluster | None`: cosine similarity + time-window incremental assignment; no LLM; **sync pure-compute**. Returns the merged `Cluster` (weighted centroid + preview concat + members append) when a match is found within the window, else `None`.
- `cluster_by_llm(new_cluster: Cluster, existing_clusters: list[Cluster], *, llm, k_candidates=30, llm_skip_threshold=0.85, prompt=None, preview_cap=5) -> Cluster | None`: top-K geometric recall + cosine fast-path skip (top-1 >= `llm_skip_threshold`) + LLM ranking when the fast path misses; async; raises on LLM failure or malformed JSON (no geometric fallback).
- `Cluster` frozen pydantic model (`arbitrary_types_allowed=True` for the numpy centroid): `id: str | None = None` (caller-stamped), `centroid: np.ndarray`, `count: int = 1`, `last_ts: int` (ms epoch), `preview: list[str] = []`, `members: list[str] = []` (caller-supplied entity ids; the algorithm appends on merge, never inspects semantics).
- English clustering prompt `CLUSTER_LLM_ASSIGN_PROMPT` in `prompts/en/cluster.py`; Chinese variant re-exports the English prompt (the template is language-neutral; responses adapt to the corpus language).

### Changed

- `cluster_by_geometry` is now **synchronous** (`def`, no `await`) — it is pure geometry (cosine + time-window) with no I/O, following the sync-for-pure-compute convention used by `fusion.rrf` / `count_tokens`. `cluster_by_llm` stays async (it calls an LLM). Callers must drop the `await`.

### Removed

- `ClusterState` value object and `ClusterConfig` dataclass: replaced by caller-owned `list[Cluster]` plus per-function keyword arguments. The caller owns the cluster list, stamps cluster IDs, and serialises; the algorithm owns only the merge transition.
- Geometric fallback path inside `cluster_by_llm`: the function raises on LLM failure rather than silently falling back to geometry.
- 3-retry loop in `cluster_by_llm`: the function raises `ValueError` immediately on bad LLM JSON.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-clustering/v0.2.0...HEAD
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-clustering/v0.2.0
