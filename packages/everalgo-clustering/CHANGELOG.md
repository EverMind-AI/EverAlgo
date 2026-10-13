# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0rc1] - 2026-10-13

### Changed

- **BREAKING: `acluster_by_llm` / `cluster_by_llm` follow the new LLM contract** (`docs/concepts/llm-contract.md`). The call passes `response_format=<pydantic model>` (a strict `{"idx": int, "reason": str}` object; both fields required, extra fields rejected) and reads the returned `str`; a custom client passed as `llm` must implement `achat(messages, *, response_format, ...) -> str`. A reply that is not exactly that object, including one wrapped in markdown fences or with a quoted `idx`, now raises `everalgo.llm.LLMOutputError` (it was a `ValueError`, found by a regex over the reply, with `reason` optional and `idx` coerced by `int()`). No retry.
- **BREAKING: the assignment prompt is split into a system message of rules and a user message of data.** `CLUSTER_LLM_ASSIGN_PROMPT` is replaced by `CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT` (no placeholder) and `CLUSTER_LLM_ASSIGN_USER_PROMPT` (`{{clusters_json}}`, `{{memcell_text}}`, each in its own tag); the bracketed headings became XML-style tags and every rule sentence is unchanged. The reply skeleton in `<reply_format>` is now a parseable JSON object, `{"idx": 0, "reason": "short reason"}`, instead of `{"idx": <existing_idx or -1>, "reason": "short reason"}`; what the stand-in said moved into the sentence before it (`idx is an existing idx or -1`). The reply shape is unchanged.
- **BREAKING:** `prompt=` is removed from `acluster_by_llm` and `cluster_by_llm`; passing it raises `TypeError`. They take `custom_instructions=` in its place, which appends to the bundled rules instead of replacing them (the old `prompt=` replaced the whole single-message template): the text is sent verbatim, never rendered as a template (a `{{...}}` in it stays literal), as a `<custom_instructions>` block that is always the system message's last; the user message is unchanged. `None` or a blank string adds nothing, so the system message is byte-for-byte `CLUSTER_LLM_ASSIGN_SYSTEM_PROMPT`. No way is left to replace the bundled rules per call: a replacement also dropped the reply shape the operator parses. `everalgo.clustering.prompts.en.cluster.assemble_system_prompt` builds the message.
- **BREAKING:** operators call `LLMClient.achat`, the new name of the protocol method; a custom client passed as `llm` must implement `achat` instead of `chat`. See `everalgo-core` for the contract.
- **BREAKING: `cluster_by_llm` is renamed `acluster_by_llm`**, the native-async operator; `cluster_by_llm` is now its blocking `async_to_sync` bridge, callable only outside a running event loop, as `detect_boundaries` is for `adetect_boundaries`. `await cluster_by_llm(...)` fails: await `acluster_by_llm(...)` instead. The package now declares `asgiref>=3.0`.
- `pydantic>=2.7` is declared as a direct dependency: the package imports pydantic itself and no longer relies on the transitive pull-in via `everalgo-core`.

## [0.2.1] - 2026-06-15

### Changed

- **License relicensed from MIT to Apache-2.0** as part of the pre-open-source security audit.

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

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-clustering/v0.3.0rc1...HEAD
[0.3.0rc1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-clustering/v0.2.1...everalgo-clustering/v0.3.0rc1
[0.2.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-clustering/v0.2.0...everalgo-clustering/v0.2.1
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-clustering/v0.2.0
