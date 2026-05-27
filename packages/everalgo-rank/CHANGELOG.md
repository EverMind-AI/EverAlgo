# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-05-19

### Fixed

- `arerank` (and the sync `rerank` bridge): `json.dumps` of the candidates payload now passes `default=str`, so candidates whose `metadata` contains non-JSON-native values (e.g. `datetime` from LanceDB rows) no longer raise `TypeError: Object of type datetime is not JSON serializable`. Previously the `enable_llm_rerank=True` path crashed whenever a caller forwarded raw storage rows (such as evermem's `row_to_candidate`) without first ISO-serializing date/time fields. The serialized fallback uses Python's `str()` representation, which is sufficient for prompting an LLM but is not guaranteed to round-trip; callers needing strict round-trip should ISO-serialize before constructing the `Candidate`.

## [Unreleased]

### Added

- `arank(rank_input, **kwargs) -> RankOutput`: async unified entry point; dispatches to the appropriate algorithm via an `_ALGO_REGISTRY` keyed by memory type. Sync bridge `rank(rank_input, **kwargs) -> RankOutput` via `asgiref`.
- `EpisodicRanker`, `CaseRanker`, `SkillRanker`: facade classes that accept `llm=` at construction time and delegate to algorithm-layer functions. Profile ranking is a module-level `profile.rank(rank_input, ...) -> RankOutput` (sync pure-compute — no class, no LLM).
- Retrieval facade (each returns `list[Candidate]` and composes as `base_retrieve`, with a sync bridge): `ahybrid_retrieve` (dual-route RRF/LR), `aagentic_retrieve(query, *, base_retrieve, llm, ...) -> tuple[list[Candidate], AgenticDecision]` (LLM-guided sufficiency + multi/refined-query rounds), `acluster_retrieve` (cluster-scoped recall expansion), `amaxsim_retrieve` (parent MaxSim nearest-neighbour).
- `arerank(items, *, query, prompt, top_k, llm) -> list[Candidate]`: LLM-scored pointwise reranker.
- `RankConfig` pydantic model (+ `DEFAULT_RANK_CONFIG`) for fusion / rerank tuning; agentic retrieval parameters are plain function kwargs (there is no `AgenticConfig`). `AgenticDecision` carries the agentic round metadata returned alongside the candidates.
- `FusionMode = Literal["rrf", "lr", "vector_anchored"]`.
- `fusion.rrf(*sources, k=60) -> list[Candidate]` (Reciprocal Rank Fusion); `fusion.lr` / `fusion.vector_anchored` (logistic-regression and vector-anchored hybrid fusion); `fusion.cosine_to_lr_score` / `fusion.score_propagation` (score normalisation + parent→child propagation helpers).
- `weight` module: LR weight helpers (`LRCoefs`, `weighted_score`, `multi_field_weighting`, `default_lr_coefs`).
- English rerank prompts for episodic, case, and skill memory types in `rank/prompts/en/`.

### Removed

- Silent fallback returns in `_acheck_sufficiency` and `_agen_multi_queries`: both functions now raise on unexpected LLM response shapes instead of returning empty / default values.
- Silent fallback in `_apply_rerank_scores`: the function raises `ValueError` when the LLM response cannot be parsed into a valid score list.
- Outer `try/except` block in `aagentic_retrieve`: errors from sub-steps propagate to the caller unchanged.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/main...HEAD
