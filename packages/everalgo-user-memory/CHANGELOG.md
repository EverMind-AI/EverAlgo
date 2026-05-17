# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `BoundaryDetector`: facade class wrapping `everalgo.boundary.detect_boundaries`; accepts `llm=` at construction time and manages the carry-forward `tail` across calls.
- `EpisodeExtractor`: per-sender Episode fan-out — one LLM call per unique `sender_id` found in the `MemCell`; accepts `llm=` and `prompt=` at construction time.
- `ForesightExtractor`: single `MemCell` → `list[Foresight]`; async with sync bridge via `asgiref.async_to_sync`.
- `AtomicFactExtractor`: single `MemCell` → `list[AtomicFact]`; async with sync bridge.
- `ProfileExtractor`: chronological `list[MemCell]` (last element = most recent) → single `Profile`; single-shot LLM snapshot.
- English and Chinese prompts for all four extractors under `user_memory/prompts/{en,zh}/`.
- `WorkspaceMemCellExtractor` re-export from `everalgo.boundary` for callers that import from one place.
- `DetectionResult` re-export from `everalgo.boundary`.

### Changed

- `BoundaryDetector` renamed from `UserBoundaryDetector` (which was itself renamed from `ChatBoundaryDetector`) to match the no-prefix naming convention used across the package.
- `EpisodeExtractor.aextract` parameter `owner_id` renamed to `sender_id` to align with `ChatMessage.sender_id`.
- `ProfileExtractor` signature changed from separate `memcell` + `cluster_episodes` parameters to a single `memcells: Sequence[MemCell]` list, matching the other extractor contracts.
- `Episode`, `Foresight`, `AtomicFact`, `Profile` schemas dropped `parent_id` / `parent_type` fields and the `id` field; schemas now carry only the minimal required fields plus `ConfigDict(extra="allow")`.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/main...HEAD
