# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-15

### Fixed

- `parse_llm_json` retries once with invalid backslash escapes repaired before giving up. A model copying LaTeX out of the source document writes `$\kappa$` as `\k`, which is not a legal JSON escape, so a complete and well-formed response was discarded and the document extracted to nothing. The retry runs only after a strict parse fails, so valid JSON is never rewritten. LaTeX colliding with a legal escape (`\theta` / `\frac` / `\bar` are tab / form-feed / backspace) stays lossy; the upstream fix is to request `response_format={"type": "json_object"}`.

## [0.2.0rc1] - 2026-09-14

### Fixed

- `KnowledgeExtractor` now splits oversized table and list atoms before batching, keeping each atom within a 20,000-token ceiling while preserving dense block references and repeating Markdown table headers across pieces. This prevents table-heavy documents from bypassing the 80,000-token batch budget; callers of the internal `split_content_to_blocks` helper can tune the ceiling with `max_atom_tokens` or pass `0` to retain the previous unbounded behavior.

## [0.1.1] - 2026-06-16

First public release. 0.1.0 was burned by a partial PyPI upload (filename reuse policy).
Previously a reserved namespace with `Private :: Do Not Upload`.

### Added

- `KnowledgeExtractor`: file-based knowledge extraction pipeline — `ParsedContent → list[KnowledgeMemory]` via preprocess + atomize → per-batch topic-tree LLM extraction → cross-batch merge → postprocess (split unsplit leaves + assign uncovered blocks) → tree assembly + DFS flatten. Multi-provider LLM client; tiktoken-bounded batching. Ships English prompts, unit + functional tests, a CLI walkthrough script, and an HTML visualizer.
- `aclassify_category` / `classify_category`: document-level classification against a caller-owned `CategorySpec` taxonomy. Single LLM call, closed-set; parse failure / out-of-set collapses to `""`. Integrated into `KnowledgeExtractor.aextract` via `categories=` / `category_id=` kwargs.

[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-knowledge/v0.2.0
[0.2.0rc1]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-knowledge/v0.2.0rc1
[0.1.1]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-knowledge/v0.1.1
[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-knowledge/v0.2.0...HEAD
