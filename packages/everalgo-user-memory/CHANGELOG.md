# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `ProfileExtractor` no longer leaks other participants' information into the target user's `Profile` in multi-speaker conversations. INIT and UPDATE prompts now receive `sender_id` as an explicit `{target_user}` with speaker-attribution rules, and `aextract` fail-loud validates that `sender_id` is a human (`role == "user"`) speaker present in the input. Removed the never-consumed `TEAM_PROFILE_UPDATE_PROMPT` dead constant.

## [0.3.1] - 2026-06-24

### Fixed

- `EpisodeExtractor`, `ForesightExtractor`, and `AtomicFactExtractor` now pass the first message's timestamp (`memcell.items[0].timestamp`) as the conversation start time to LLM prompts. Previously they passed `memcell.timestamp` (closing time of the slice), which skewed absolute date resolution for relative time expressions.
- Episode prompt: relative time references (e.g. "last Friday", "last summer") are now resolved using each message's own timestamp instead of `conversation_start_time`. Fixes off-by-one-week errors when a MemCell spans multiple days. Per-message timestamps switched from ISO 8601 to human-readable format with weekday labels.

## [0.3.0] - 2026-06-15

### Changed

- **License relicensed from MIT to Apache-2.0** as part of the pre-open-source security audit.

### Added

- `EpisodeReflector`: merge N chronologically-ordered episodes into one accurate narrative. Two modes: INIT (full merge, `old_episode=None`) and UPDATE (incremental, `old_episode=Episode`). Uses OpenAI Structured Outputs via `response_format`. Re-exported from `everalgo.user_memory`.

## [0.2.0] - 2026-05-27

> First archived changelog. Entries below accumulated since the initial `0.1.0` PyPI release (published manually, without a git tag or per-package changelog), so this `0.2.0` section also consolidates the previously-unarchived `0.1.0` surface.

### Added

- `BoundaryDetector`: facade class wrapping `everalgo.boundary.detect_boundaries`; accepts `llm=` at construction time and manages the carry-forward `tail` across calls.
- `EpisodeExtractor`: per-sender Episode fan-out — one LLM call per unique `sender_id` found in the `MemCell`; accepts `llm=` and `prompt=` at construction time.
- `ForesightExtractor`: single `MemCell` → `list[Foresight]`; async with sync bridge via `asgiref.async_to_sync`.
- `AtomicFactExtractor`: single `MemCell` → `list[AtomicFact]`; async with sync bridge.
- `ProfileExtractor`: chronological `list[MemCell]` (last element = most recent) → single `Profile`; single-shot LLM snapshot.
- English and Chinese prompts for all four extractors under `user_memory/prompts/{en,zh}/`.
- `DetectionResult` re-export from `everalgo.boundary`.

### Changed

- `WorkspaceMemCellExtractor` is no longer re-exported from `everalgo.user_memory` (`__all__`). It was an unimplemented stub that raised `NotImplementedError`; it now lives only in `everalgo.boundary.workspace`. Re-adding it once implemented is a non-breaking addition.
- `BoundaryDetector` renamed from `UserBoundaryDetector` (which was itself renamed from `ChatBoundaryDetector`) to match the no-prefix naming convention used across the package.
- `EpisodeExtractor.aextract` parameter `owner_id` renamed to `sender_id` to align with `ChatMessage.sender_id`.
- `ProfileExtractor` signature changed from separate `memcell` + `cluster_episodes` parameters to a single `memcells: Sequence[MemCell]` list, matching the other extractor contracts.
- `Episode`, `Foresight`, `AtomicFact`, `Profile` schemas dropped `parent_id` / `parent_type` fields and the `id` field; schemas now carry only the minimal required fields plus `ConfigDict(extra="allow")`.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-user-memory/v0.3.0...HEAD
[0.3.0]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-user-memory/v0.2.0...everalgo-user-memory/v0.3.0
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-user-memory/v0.2.0
