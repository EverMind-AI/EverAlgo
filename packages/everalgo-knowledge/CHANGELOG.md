# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-06-16

First public release. 0.1.0 was burned by a partial PyPI upload (filename reuse policy).
Previously a reserved namespace with `Private :: Do Not Upload`.

### Added

- `KnowledgeExtractor`: file-based knowledge extraction pipeline — `ParsedContent → list[KnowledgeMemory]` via preprocess + atomize → per-batch topic-tree LLM extraction → cross-batch merge → postprocess (split unsplit leaves + assign uncovered blocks) → tree assembly + DFS flatten. Multi-provider LLM client; tiktoken-bounded batching. Ships English prompts, unit + functional tests, a CLI walkthrough script, and an HTML visualizer.
- `aclassify_category` / `classify_category`: document-level classification against a caller-owned `CategorySpec` taxonomy. Single LLM call, closed-set; parse failure / out-of-set collapses to `""`. Integrated into `KnowledgeExtractor.aextract` via `categories=` / `category_id=` kwargs.

[0.1.1]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-knowledge/v0.1.1
[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-knowledge/v0.1.1...HEAD
