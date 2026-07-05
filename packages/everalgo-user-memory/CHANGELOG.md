# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Target-aware profile extraction (multi-speaker attribution).** `ProfileExtractor` now threads the `sender_id` it is already given into both the INIT and UPDATE prompts as `target_user`. The prompts instruct the model to attribute each fact to the speaker who stated it (every conversation line is tagged `(user_id:...)`) and to treat other participants — and the AI assistant — as context, never as the profile's subject. This prevents cross-speaker contamination in group conversations (e.g. another member's facts, or the assistant's own persona, leaking into a user's profile). Applied symmetrically to `prompts/en/profile.py` and `prompts/zh/profile.py`.

### Changed

- **Anti-bloat "durable abstraction" rule for profile descriptions.** Both INIT and UPDATE prompts (en + zh) now carry a HARD RULE requiring each `description` to be a timeless generalization: no date/weekday/clock-time in `description` (coarse time belongs in `evidence`), and repeated instances of a pattern must be folded into one existing item via `update` rather than appended as new dated clauses. Includes WRONG/RIGHT examples and a pre-output self-check. Reduces the accumulation of dated, event-log-style profile entries over many updates.
- **`_render_conversation` coarsens per-message timestamps to date granularity** (`YYYY-MM-DD`). Profile extraction needs no intraday precision; dropping the clock component removes second-level noise, avoids UTC/local intraday misreads, and reinforces the anti-bloat "no clock time" rule by keeping only a coarse date in the rendered transcript.

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
