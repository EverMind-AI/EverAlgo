# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `detect_boundaries(messages, *, llm, tail) -> DetectionResult`: async public entry point that splits a `list[ChatMessage]` into `MemCell` slices using a single LLM call; no retry.
- `DetectionResult` `NamedTuple`: `(cells: list[MemCell], tail: list[ChatMessage])` where `tail` is the unconsumed carry-forward window.
- `WorkspaceMemCellExtractor`: stub extractor for Jira / Email / Confluence inputs (EXPERIMENTAL; implementation pending).
- `count_tokens(text: str) -> int`: token count using OpenAI `o200k_base` encoding via `tiktoken`.
- `force_split(text: str, *, max_tokens: int) -> list[str]`: token-bounded chunking for caller-side prompt fitting.
- English and Chinese boundary detection prompts under `boundary/prompts/{en,zh}/`.
- `BoundaryDetector` (in `everalgo-user-memory`) and `AgentBoundaryDetector` (in `everalgo-agent-memory`) are the facade classes; `detect_boundaries` is this package's low-level public primitive.
- 81 unit tests across `test_chat.py` (boundary logic), `test_boundary_public_api.py` (public contract), and `test_tokenize.py` (token helpers).

### Removed

- 5-retry loop in the internal `_detect_boundaries` helper: the function now raises `ValueError` immediately on malformed LLM JSON instead of silently retrying.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/main...HEAD
