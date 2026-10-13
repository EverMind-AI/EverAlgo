# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0rc2] - 2026-10-13

### Changed

- Every system message (topic-tree extraction, content merge, topic split, uncovered-block assignment, classification) closes its `<custom_instructions>` block with core's `everalgo.utils.append_custom_instructions`; the package's private `_render.append_custom_instructions` is removed. The rendered system message is byte-for-byte unchanged.
- Requires `everalgo-core>=0.6.0rc3` (the floor was `0.6.0rc1`): the package imports `everalgo.utils.append_custom_instructions`, which `0.6.0rc1` does not define.

## [0.3.0rc1] - 2026-10-13

### Changed

- **BREAKING:** every LLM call follows the new `LLMClient` contract (`docs/concepts/llm-contract.md`). Each of the six calls (topic-tree extraction, content merge, topic merge, classification, leaf split, uncovered-block assignment) passes a strict pydantic model as `response_format`, receives the reply as `str`, and validates it with `everalgo.llm.parse_reply`; a reply that does not validate raises `everalgo.llm.LLMOutputError`, and the client's own exceptions propagate unchanged. The operators no longer degrade on an unparsable reply: a failed batch used to be skipped (and a document whose every batch failed extracted to `[]`), a failed merge or post-processing pass returned its input, and a failed classification returned `""`; each now raises. Business fallbacks on a valid reply stay: an out-of-taxonomy category still collapses to `""`, a blank merged summary still falls back to the first batch's, and merge groups, split results and assignments that name an unknown topic or an out-of-range index are still skipped.
- **BREAKING:** each prompt is a system message of static rules plus a user message of data in snake_case XML tags (`<document_title>`, `<paragraphs count="N" max_id="M">`, ...); the classification taxonomy closes the system message (`CATEGORY_CLASSIFY_SYSTEM_PROMPT`, then `assemble_system_prompt` appends the taxonomy), so the rules stay a cacheable prefix across one ingest run. The rule sentences are unchanged except where the strict reply model or the move demanded it: leaf topics must write `"children": []` (they could omit it), a `should_split=false` split result writes `block_refs` and `children` as `null` (it could omit them), and the rules that pointed at data by position or slot name now name its tag. The `topic_split` and `uncovered_assign` skeletons, which reached the model with doubled braces, are valid JSON. The prompt constants are renamed `*_SYSTEM_PROMPT` / `*_USER_PROMPT` (the `*_PROMPT_EN` names are gone).
- **BREAKING:** the prompt overrides are removed, and a host's text goes in `custom_instructions=`, which appends to the bundled rules instead of replacing them. `KnowledgeExtractor.aextract` loses `prompt` (it replaced the topic-tree rules only) and takes one `custom_instructions`, sent with every LLM call of the extraction: each topic-tree batch, the content and topic merges, classification, leaf split and uncovered-block assignment. `aclassify_category` loses `prompt` and takes `custom_instructions`; passing a removed name raises `TypeError`. The text is sent verbatim, never rendered as a template (a `{{...}}` in it stays literal), as a `<custom_instructions>` block that is always the system message's last (after the taxonomy in classification); `None` or a blank string sends exactly the bundled rules. `category_classify.assemble_system_prompt` is now `(*, taxonomy, custom_instructions)`.
- **BREAKING:** operators call `LLMClient.achat`, the new name of the protocol method; a custom client passed as `llm` must implement `achat` instead of `chat`. See `everalgo-core` for the contract.

### Removed

- The internal `_llm_json` module (`parse_llm_json`): fenced / prose-wrapped JSON extraction and the invalid-backslash-escape repair. Replies are parsed strictly against their models; a model that copies LaTeX such as `\kappa` into a JSON string unescaped now gets an `LLMOutputError` instead of a repaired parse; the remedy, as the 0.2.0 entry below said, is a client that constrains the reply to JSON (`OpenAICompatClient`'s `json_schema` or `json_object` mode).

## [0.2.1] - 2026-09-16

### Changed

- The minimum `everalgo-core` version is now `0.5.1`, because knowledge splitting relies on that release's special-token-safe `count_tokens` behavior.

### Fixed

- `split_content_to_blocks` and `split_and_batch_content` no longer raise `ValueError` on a document that quotes an `o200k_base` special-token literal such as `<|endoftext|>` — tokenizer documentation and bug reports could not be ingested at all. `_block_split` had grown a second, independent tokenizer; it now counts through `everalgo._tokenize.count_tokens`, which is the shared entry point fixed in `everalgo-core` (see that changelog for the rationale). Block boundaries and batch budgets for documents without such a literal are unchanged.

### Removed

- The direct `tiktoken` dependency. The package no longer imports it; token counting goes through `everalgo-core`, which declares it. No action is needed from callers — `tiktoken` still arrives transitively.

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
[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-knowledge/v0.3.0rc2...HEAD
[0.3.0rc2]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-knowledge/v0.3.0rc1...everalgo-knowledge/v0.3.0rc2
[0.3.0rc1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-knowledge/v0.2.1...everalgo-knowledge/v0.3.0rc1
[0.2.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-knowledge/v0.2.0...everalgo-knowledge/v0.2.1
