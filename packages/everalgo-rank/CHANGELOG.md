# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `acluster_retrieve` / `cluster_retrieve` take `max_members` and `member_retrieve` to bound the cluster expansion. The expansion grows with cluster size rather than with any `k`, so a large cluster sent thousands of members to the caller's rerank. With `max_members` set, seed hits come first in base order; once the expansion exceeds the budget, the remaining members are ranked by `member_retrieve(query, member_ids, k)` and only the top ones fill it. At or under the budget nothing is cut and `member_retrieve` is not called. `max_members` below 1, or set without `member_retrieve`, raises `InvalidInputError`. The default (`None`) returns the unbounded expansion in `all_docs` order, unchanged.
- `MemberRetrieveFn`, the callable type of `member_retrieve`, exported from `everalgo.rank` and `everalgo.rank.protocols`.

## [0.5.0rc3] - 2026-10-13

### Changed

- Every system message (rerank, sufficiency check, multi-query, refined query) closes its `<custom_instructions>` block with core's `everalgo.utils.append_custom_instructions`; the package's private `_render.append_custom_instructions` is removed. The rendered system message is byte-for-byte unchanged.
- Requires `everalgo-core>=0.6.0rc3` (the floor was `0.6.0rc2`): the package imports `everalgo.utils.append_custom_instructions`, which `0.6.0rc2` does not define.

## [0.5.0rc2] - 2026-10-13

### Changed

- Invalid caller input now raises `everalgo.utils.InvalidInputError` (a `ValueError` subclass, so existing handlers keep working) instead of a bare `ValueError`: `enable_rerank=True` without an `llm`, an unsupported `fusion_mode` for the facade, and a candidate shown to the agentic LLM with an empty `episode.content` (named by its id).
- Requires `everalgo-core>=0.6.0rc2` (the floor was `0.6.0rc1`): the package imports `everalgo.utils.InvalidInputError`, which `0.6.0rc1` does not define.

## [0.5.0rc1] - 2026-10-13

### Removed

- **BREAKING: `RankConfig.expand_limit` and `RankConfig.max_convergence_rounds`.** Nothing read them: they belonged to the episode → atomic-fact expansion the episodic ranker no longer performs.

### Changed

- **BREAKING:** the four LLM calls (`aagentic_retrieve`'s sufficiency check, multi-query and refined-query generation, and `arerank`, which the `case` / `skill` / `episodic` facades use when `enable_rerank=True`) request their reply as a strict pydantic model passed as `response_format` and validate it with `everalgo.llm.parse_reply`, instead of asking for `{"type": "text"}` and pulling a `{...}` block out of `.content`. Sufficiency: `{"is_sufficient", "reasoning", "key_information_found", "missing_information"}`; multi-query: `{"queries", "reasoning"}`; rerank: `{"ranked": [{"id", "score"}]}`. Refined-query generation, which used to return free text, now asks for `{"text": "..."}` (the contract has no free-text tier); the prefix stripping, the 5-300 character check and the fall-back to the original query apply to `text` as before. A reply that fails now raises `everalgo.llm.LLMOutputError` instead of `ValueError` / `json.JSONDecodeError`; its message names each failing location, never the reply's content. Replies previously repaired are now rejected: prose around the object, an extra key, a missing key (the sufficiency fields used to default to `False` / `""` / `[]`, a rerank item's `score` to `0.0`), or a mistyped value (a non-string rerank `id` used to be skipped). Rerank ids the model made up are still skipped. Temperatures are unchanged (0.0 / 0.4 / 0.3; rerank none). The `LLMClient` passed as `llm` must implement the current contract: `achat(messages, *, response_format: type[BaseModel]) -> str`. See `everalgo-core` for the contract.
- **BREAKING:** each LLM call sends a system message with the rules and a user message with the data, instead of one user message holding both. The constants keep their names and now hold the rules only: `SUFFICIENCY_CHECK_PROMPT_EN`, `MULTI_QUERY_PROMPT_EN` and `REFINED_QUERY_PROMPT_EN` have no placeholder; the six `*_RERANK_PROMPT_{EN,ZH}` keep only `{{top_k}}`. The data templates are new: `SUFFICIENCY_CHECK_USER_PROMPT`, `MULTI_QUERY_USER_PROMPT`, `REFINED_QUERY_USER_PROMPT` beside them, and `everalgo.rank.prompts.rerank_user.RERANK_USER_PROMPT` shared by every rerank. Every rule line is kept; Markdown headings became XML-style tags, `**` and inline backticks were dropped, each prompt closes with a `<reply_format>` block whose skeleton is now parseable JSON, and the rerank candidates line points at `<candidates>`.
- **BREAKING:** the prompt overrides are removed, and a host's text goes in `custom_instructions=`, which appends to the bundled rules instead of replacing them. `aagentic_retrieve` loses `sufficiency_prompt`, `multi_query_prompt` and `refined_query_prompt` and takes one `custom_instructions`, sent with each of its LLM calls (the sufficiency check and the Round 2 query generation); the `case` / `skill` / `episodic` rankers (module `arank` and the `CaseRanker` / `SkillRanker` / `EpisodicRanker` classes) lose `prompt` and take `custom_instructions`, sent with the rerank call when `enable_rerank=True`; passing a removed name raises `TypeError`. The text is sent verbatim, never rendered as a template (a `{{...}}` in it stays literal), as a `<custom_instructions>` block that is always the system message's last; `None` or a blank string sends exactly the bundled rules. The data always goes in the user message the operator renders. `arerank` keeps `prompt`, the rerank rules it renders with `{{top_k}}` (it has no bundled default; a rule template naming any other slot raises `ValueError` before the call), and takes `custom_instructions` the same way, after the rendered rules. With the ranker `prompt` gone, the `*_RERANK_PROMPT_ZH` rules reach the model only through `arerank(prompt=...)`.
- **BREAKING:** operators call `LLMClient.achat`, the new name of the protocol method; a custom client passed as `llm` must implement `achat` instead of `chat`. See `everalgo-core` for the contract.
- The `Date` row of each retrieved document in the sufficiency, multi-query and refined-query prompts renders as `YYYY-MM-DD HH:MM UTC` (`everalgo.utils.format_utc_time`, seconds truncated) instead of ISO `YYYY-MM-DDTHH:MM:SSZ`, the shape every other prompt instant takes; a missing or invalid timestamp still renders `N/A`.
- `pydantic>=2.7` is declared as a direct dependency: the package imports pydantic itself and no longer relies on the transitive pull-in via `everalgo-core`.

## [0.4.1] - 2026-06-24

### Changed

- `acluster_retrieve`: internal variable `memcell_to_cluster` renamed to `member_to_cluster` to reflect entity-split model where clusters reference episode IDs, not memcell IDs.

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

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-rank/v0.5.0rc3...HEAD
[0.5.0rc3]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-rank/v0.5.0rc2...everalgo-rank/v0.5.0rc3
[0.5.0rc2]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-rank/v0.5.0rc1...everalgo-rank/v0.5.0rc2
[0.5.0rc1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-rank/v0.4.1...everalgo-rank/v0.5.0rc1
[0.4.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-rank/v0.4.0...everalgo-rank/v0.4.1
[0.4.0]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-rank/v0.3.1...everalgo-rank/v0.4.0
[0.3.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-rank/v0.3.0...everalgo-rank/v0.3.1
[0.3.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-rank/v0.3.0
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-rank/v0.2.0
[0.1.1]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-rank/v0.1.1
