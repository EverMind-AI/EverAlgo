# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-05-19

### Fixed

- `arerank` (and the sync `rerank` bridge): `json.dumps` of the candidates payload now passes `default=str`, so candidates whose `metadata` contains non-JSON-native values (e.g. `datetime` from LanceDB rows) no longer raise `TypeError: Object of type datetime is not JSON serializable`. Previously the `enable_llm_rerank=True` path crashed whenever a caller forwarded raw storage rows (such as evermem's `row_to_candidate`) without first ISO-serializing date/time fields. The serialized fallback uses Python's `str()` representation, which is sufficient for prompting an LLM but is not guaranteed to round-trip; callers needing strict round-trip should ISO-serialize before constructing the `Candidate`.

## [Unreleased]

### Added

- `arank(rank_input, **kwargs) -> RankOutput`: async unified entry point; dispatches to the appropriate algorithm via an `_ALGO_REGISTRY` keyed by memory type.
- `rank(rank_input, **kwargs) -> RankOutput`: sync bridge via `asgiref.async_to_sync`.
- `EpisodicRanker`, `CaseRanker`, `SkillRanker`, `ProfileRanker`: facade classes that accept `llm=` at construction time and delegate to algorithm-layer functions.
- `aagentic_rank(query, candidates, *, llm, config) -> RankOutput`: multi-round agentic retrieval — LLM-guided sufficiency check followed by multi-query expansion, then RRF fusion over all rounds.
- `arerank(query, candidates, *, llm, prompt, config) -> list[Candidate]` and sync `rerank` bridge: LLM-scored pointwise reranker.
- `RankConfig` and `AgenticConfig` pydantic models for tuning rerank and agentic retrieval parameters.
- `FusionMode` enum: `rrf`, `lr`, `vector_anchored`.
- `fusion.rrf(*sources, k) -> list[Candidate]`: Reciprocal Rank Fusion over multiple recall lists.
- `fusion.lr(candidates, *, weights) -> list[Candidate]`: linear-rank combination.
- `fusion.cosine_to_lr_score`, `fusion.score_propagation`, `fusion.expand`: score normalisation and episode-to-fact cross-memory expansion helpers.
- `weight` module: per-route weight configuration helpers.
- English rerank prompts for episodic, case, and skill memory types in `rank/prompts/en/`.

### Removed

- Silent fallback returns in `_acheck_sufficiency` and `_agen_multi_queries`: both functions now raise on unexpected LLM response shapes instead of returning empty / default values.
- Silent fallback in `_apply_rerank_scores`: the function raises `ValueError` when the LLM response cannot be parsed into a valid score list.
- Outer `try/except` block in `aagentic_rank`: errors from sub-steps propagate to the caller unchanged.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/main...HEAD
