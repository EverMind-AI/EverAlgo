# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-27

> First archived changelog. Entries below accumulated since the initial `0.1.0` PyPI release (published manually, without a git tag or per-package changelog), so this `0.2.0` section also consolidates the previously-unarchived `0.1.0` surface.

### Added

- `AgentBoundaryDetector`: facade class that filters a `list[ConversationItem]` to only tool-call-carrying turns, calls `everalgo.boundary.detect_boundaries` on the result, then remaps indices back to the original sequence.
- `AgentCaseExtractor`: 11-step pipeline — heuristic skip check, tool-call-round counting, message trimming / pre-compression, `AgentCase` extraction via LLM. Accepts `llm=` at construction time.
- `AgentSkillExtractor`: add / update / retire operations with `cluster_id` binding; accepts `llm=` at construction time. `SkillConfig` pydantic model exposes maturity and retire thresholds.
- English prompts for case and skill extraction under `agent_memory/prompts/en/`.

### Changed

- `AgentCaseExtractor` and `AgentSkillExtractor` rewired to the unified `MemCell` contract (replacing the previous `AgentMemCell` type).

### Removed

- Broad `except Exception → None` handler in `_evaluate_maturity`: the function now propagates exceptions from the LLM call.
- `asyncio.gather(return_exceptions=True)` in `_pre_compress_to_list`: individual compression errors now propagate instead of being swallowed.
- Fail-open path in `_is_worth_extracting`: the function raises on LLM error instead of returning `True` unconditionally.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-agent-memory/v0.2.0...HEAD
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-agent-memory/v0.2.0
