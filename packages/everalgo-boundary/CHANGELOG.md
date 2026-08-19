# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-08-19

### Changed

- **BREAKING: `DetectionResult` gained a third field, `should_wait`, so two-value unpacking no longer works.** `cells, tail = await detect_boundaries(...)` now raises `ValueError`; use `cells, tail, should_wait = ...` or named access. The field was already being produced and thrown away: the batch prompt has always asked for it, `_call_llm_for_batch_boundary` has always rejected a response that omits it, and `detect_boundaries` then read only `batch.boundaries` — so every caller since 0.1 has been unable to see a verdict the model was computing and the library was validating.

  `should_wait` answers a narrower question than "is the tail non-empty". A non-empty tail is the normal case for `is_final=False`; `should_wait` says the trailing segment carries too little to place in an episode at all — only media placeholders, an intent-free "ok", a system notification, or an ambiguous 30-minute-to-4-hour gap. Without it a caller has to treat every tail the same way: extract on all of them and it extracts those, or wait on all of them and it never extracts. The prompt's `### should_wait` section has specified these cases all along.
- `should_wait` is `bool | None`, and `None` is not interchangeable with `False`. `detect_boundaries` returns the LLM's `bool`. An empty message list returns `None` — no LLM was called, so nothing judged the tail, and reporting `False` there would be a fabricated verdict.

### Fixed

- The `DetectionResult` docstring no longer advertises `cells, tail = ...` two-value unpacking as a supported form.

## [0.2.1] - 2026-06-15

### Changed

- **License relicensed from MIT to Apache-2.0** as part of the pre-open-source security audit.

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

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-boundary/v0.3.0...HEAD
[0.3.0]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-boundary/v0.2.1...everalgo-boundary/v0.3.0
[0.2.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-boundary/v0.2.0...everalgo-boundary/v0.2.1
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-boundary/v0.2.0
