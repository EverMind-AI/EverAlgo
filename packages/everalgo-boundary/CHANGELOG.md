# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-27

> First archived changelog. Entries below accumulated since the initial `0.1.0` PyPI release (published manually, without a git tag or per-package changelog), so this `0.2.0` section also consolidates the previously-unarchived `0.1.0` surface.

### Added

- `detect_boundaries(messages, *, llm, is_final, prompt, hard_token_limit, hard_msg_limit) -> DetectionResult`: async public entry point that splits a `list[ChatMessage]` into `MemCell` slices using a single LLM call; no retry.
- `DetectionResult` `NamedTuple`: `(cells: list[MemCell], tail: list[ChatMessage])` where `tail` is the unconsumed carry-forward window.
- `count_tokens(text: str) -> int`: token count using OpenAI `o200k_base` encoding via `tiktoken`.
- `force_split(text: str, *, max_tokens: int) -> list[str]`: token-bounded chunking for caller-side prompt fitting.
- English and Chinese boundary detection prompts under `boundary/prompts/{en,zh}/`.
- `BoundaryDetector` (in `everalgo-user-memory`) and `AgentBoundaryDetector` (in `everalgo-agent-memory`) are the facade classes; `detect_boundaries` is this package's low-level public primitive.
- 81 unit tests across `test_chat.py` (boundary logic), `test_boundary_public_api.py` (public contract), and `test_tokenize.py` (token helpers).

### Changed

- `WorkspaceMemCellExtractor` is no longer exported from `everalgo.boundary` (`__all__`). It is an unimplemented stub whose methods raise `NotImplementedError`; keeping it off the public surface removes the import-then-crash trap. It stays importable from `everalgo.boundary.workspace` as a forward reference. Re-adding it to the public API once implemented is a non-breaking addition.

### Removed

- 5-retry loop in the internal `_detect_boundaries` helper: the function now raises `ValueError` immediately on malformed LLM JSON instead of silently retrying.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-boundary/v0.2.0...HEAD
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-boundary/v0.2.0
