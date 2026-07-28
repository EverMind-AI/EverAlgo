# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`aextract_with_reason` on `AgentCaseExtractor` and `AgentSkillExtractor`** (plus `extract_with_reason` sync bridges) — same algorithm as `aextract`, but a rejection comes back as a typed reason instead of only reaching the log. Callers that serve "why is this session's memory empty?" no longer have to scrape log lines to attribute an empty result.
- **`CaseSkipReason` (13 members) and `SkillSkipReason` (12 members)** — one member per rejection gate in the two pipelines, exported from `everalgo.agent_memory`. Reasons are algorithmic: they name the gate, not what the rejection means for any particular product.
- **`CaseExtractionResult`, `SkillExtractionResult`, `OpOutcome`** NamedTuples. `CaseExtractionResult` carries `(cases, reason, detail)`. The skill side carries `(pre_reason, pre_detail, outcomes)` — the `pre_*` pair reports short-circuits that fire before any LLM call, while `outcomes` holds one `OpOutcome` per LLM-proposed operation (so `len(outcomes) == len(operations)`), which keeps the three distinct empty states — quality short-circuit, LLM proposed nothing, every operation dropped — distinguishable. Convenience properties: `SkillExtractionResult.skills` / `.dropped`.
- `detail` payloads carry observed values alongside the thresholds they missed (e.g. `{"rounds": 2, "min_rounds": 3}`) as structured data, so callers can compose user-facing messages without parsing strings.

### Changed

- `aextract` on both extractors is now a thin wrapper over `aextract_with_reason`. Signatures, return types, and behaviour are unchanged.
- Internal (underscore-prefixed) helpers now return their rejection reason: `_should_skip` returns `(reason, detail) | None` rather than a prose string, `_is_worth_extracting` returns `(worth, llm_reason)`, `_compress_experience` returns `(data, reason)`, and `_apply_add` / `_apply_update` return an `OpOutcome` and take a required `op_index` keyword.

## [0.3.1] - 2026-06-15

### Fixed

- Raise `everalgo-core` dependency lower bound from `>=0.2.0` to `>=0.2.1`. `AgentProfileExtractor` imports `AgentProfilePatch` from `everalgo.types`, which was added in core 0.2.1; the previous floor allowed resolving core 0.2.0 where the type does not exist.

## [0.3.0] - 2026-06-15

### Changed

- **License relicensed from MIT to Apache-2.0** as part of the pre-open-source security audit.
- Case filter tightened: exploration turns and user-correction signals now contribute to case boundary detection, reducing noise in extracted cases.

### Added

- `AgentProfileExtractor`: extract and update agent profiles (SOUL.md / AGENTS.md section-level updates). Supports INIT and UPDATE modes mirroring `ProfileExtractor`. Re-exported from `everalgo.agent_memory`.

## [0.2.0] - 2026-05-27

> First archived changelog. Entries below accumulated since the initial `0.1.0` PyPI release (published manually, without a git tag or per-package changelog), so this `0.2.0` section also consolidates the previously-unarchived `0.1.0` surface.

### Added

- `AgentBoundaryDetector`: facade class that filters a `list[ConversationItem]` to only tool-call-carrying turns, calls `everalgo.boundary.detect_boundaries` on the result, then remaps indices back to the original sequence.
- `AgentCaseExtractor`: 11-step pipeline — heuristic skip check, tool-call-round counting, message trimming / pre-compression, `AgentCase` extraction via LLM. Accepts `llm=` at construction time.
- `AgentSkillExtractor`: add / update / retire operations with `cluster_id` binding; accepts `llm=` at construction time. Configuration is handled via `_SkillCfg` (internal dataclass, not public API); maturity and retire thresholds are passed as per-call kwargs.
- English prompts for case and skill extraction under `agent_memory/prompts/en/`.

### Changed

- `AgentCaseExtractor` and `AgentSkillExtractor` rewired to the unified `MemCell` contract (replacing the previous `AgentMemCell` type).

### Removed

- Broad `except Exception → None` handler in `_evaluate_maturity`: the function now propagates exceptions from the LLM call.
- `asyncio.gather(return_exceptions=True)` in `_pre_compress_to_list`: individual compression errors now propagate instead of being swallowed.
- Fail-open path in `_is_worth_extracting`: the function raises on LLM error instead of returning `True` unconditionally.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-agent-memory/v0.3.1...HEAD
[0.3.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-agent-memory/v0.3.0...everalgo-agent-memory/v0.3.1
[0.3.0]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-agent-memory/v0.2.0...everalgo-agent-memory/v0.3.0
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-agent-memory/v0.2.0
