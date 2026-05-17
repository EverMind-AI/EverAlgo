# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `cluster_by_geometry(embedding, state, *, threshold, time_window_days) -> tuple[int, ClusterState]`: cosine similarity + time-window incremental assignment; no LLM; pure-sync function.
- `cluster_by_llm(embedding, state, cluster_previews, *, llm, threshold, time_window_days, k_candidates, llm_skip_threshold) -> tuple[int, ClusterState]`: embedding top-K recall + cosine fast-path skip + LLM ranking; async; raises on LLM failure or malformed JSON.
- `ClusterState` frozen pydantic model with four fields: `centroids: dict[int, list[float]]`, `counts: dict[int, int]`, `last_ts: dict[int, int]` (ms-epoch int), `next_idx: int`. Provides `empty()`, `to_dict()`, `from_dict()`, and a private `_assign()` mutation method.
- English clustering prompt `CLUSTER_LLM_ASSIGN_PROMPT` in `prompts/en/cluster.py`; Chinese variant re-exports the English prompt (the template is language-neutral; responses adapt to the corpus language).

### Changed

- `ClusterState.last_ts` stores millisecond epoch integers instead of float seconds, aligning with `MemCell.timestamp` and `Episode.timestamp` conventions across EverAlgo.
- `ClusterState` tracks `next_idx` as an explicit field instead of deriving cluster numbering from `max(centroids.keys()) + 1`, which was a footgun when cluster IDs were deleted.

### Removed

- `ClusterConfig` dataclass: threshold, time-window, and candidate-count parameters are now keyword arguments on each function, allowing callers to pass only what they use.
- Geometric fallback path inside `cluster_by_llm`: the function raises on LLM failure rather than silently falling back to geometry.
- 3-retry loop in `cluster_by_llm`: the function raises `ValueError` immediately on bad LLM JSON.

[Unreleased]: https://gitlab.com/npc-work/aic/ai/everalgo/-/compare/main...HEAD
