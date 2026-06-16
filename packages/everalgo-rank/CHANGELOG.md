# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-06-16

### Added

- `acategory_retrieve` / `category_retrieve`: category-aware retrieval facade with soft search-after boost. `rollup_category_mass` estimates score-weighted category confidence; `apply_category_boost` adds a precision-only tiebreaker that auto-mutes on ambiguous queries.

## [0.3.1] - 2026-06-15

### Changed

- Retrieval alignment: cluster selection changed from MaxSim scoring to first-hit scan; `cluster_base_candidates` default changed from 100 to None (no cap); embedding fallback order updated to prefer atomic-facts over episode-level embeddings.

## [0.3.0] - 2026-05-28

### Changed

- **License relicensed from MIT to Apache-2.0** as part of the pre-open-source security audit.
- Skill ranker now reorders **and** quality-grades candidates in a single LLM pass. The 0.0–1.0 relevance bands previously carried by a separate post-rerank verify stage are folded into `SKILL_RERANK_PROMPT_{EN,ZH}`.

### Added

- `SkillRanker.arank(..., min_rerank_score=0.4)` and module-level `arank(..., min_rerank_score=0.4)`: skill-only post-rerank hard threshold. After rerank, drops items whose LLM score is below the threshold. No-op when rerank did not run (raw fusion scores are not on a 0–1 scale, e.g. RRF ≈ 1/k). Set to `0.0` to disable.

### Removed (BREAKING)

- Public API `averify` / `verify` (the post-rerank LLM relevance verify stage) — folded into the rerank prompt. Callers should rely on `enable_rerank=True` together with the new `min_rerank_score` gate.
- `SkillRanker.arank` / module-level `arank` parameters: `enable_verify`, `verify_threshold`, `verify_prompt`. Replaced by `min_rerank_score`.
- `everalgo.rank.prompts.{en,zh}.skill_verify` modules and the `SKILL_VERIFY_PROMPT_EN` / `SKILL_VERIFY_PROMPT_ZH` constants.
- `VerifiedItem` pydantic model.

### Migration from 0.2.0

Before:

```python
result = await SkillRanker(llm=...).arank(
    rank_input, enable_rerank=True, enable_verify=True, verify_threshold=0.4,
)
```

After (equivalent):

```python
result = await SkillRanker(llm=...).arank(
    rank_input, enable_rerank=True, min_rerank_score=0.4,
)
```

## [0.2.0] - 2026-05-27

> First archived changelog. Entries below accumulated since the initial `0.1.0` PyPI release (published manually, without a git tag or per-package changelog), so this `0.2.0` section also consolidates the previously-unarchived `0.1.0` surface. (`0.1.1` below was a hotfix released 2026-05-19.)

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

## [0.1.1] - 2026-05-19

### Fixed

- `arerank` (and the sync `rerank` bridge): `json.dumps` of the candidates payload now passes `default=str`, so candidates whose `metadata` contains non-JSON-native values (e.g. `datetime` from LanceDB rows) no longer raise `TypeError: Object of type datetime is not JSON serializable`. Previously the `enable_llm_rerank=True` path crashed whenever a caller forwarded raw storage rows (such as EverOS's `row_to_candidate`) without first ISO-serializing date/time fields. The serialized fallback uses Python's `str()` representation, which is sufficient for prompting an LLM but is not guaranteed to round-trip; callers needing strict round-trip should ISO-serialize before constructing the `Candidate`.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-rank/v0.3.1...HEAD
[0.3.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-rank/v0.3.0...everalgo-rank/v0.3.1
[0.3.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-rank/v0.3.0
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-rank/v0.2.0
[0.1.1]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-rank/v0.1.1
