# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0rc1] - 2026-10-13

### Changed

- **BREAKING:** `AgentBoundaryDetector` is removed. `everalgo.agent_memory` exports `BoundaryDetector` instead — `everalgo.boundary.BoundaryDetector`, re-exported unchanged, with `everalgo.agent_memory.boundary` holding nothing but the re-export — whose `adetect(items, *, is_final=False, custom_instructions=None)` takes the same mixed trajectories and keeps tool items in the cells as before. It also takes `hard_token_limit` / `hard_msg_limit`, defaulting to 65536 tokens and 500 messages where `AgentBoundaryDetector` used 8192 and 50.
- **BREAKING:** every LLM call follows the new `LLMClient` contract (`docs/concepts/llm-contract.md`). `AgentCaseExtractor` (tool pre-compress, filter, compress), `AgentSkillExtractor` (skill extract, maturity) and `AgentProfileExtractor` pass a pydantic reply model as `response_format`, take the reply as `str`, and validate it with `everalgo.llm.parse_reply`; a reply that does not validate raises `LLMOutputError`, whose message never quotes the reply. The local brace-matching JSON extraction is gone, and so are the old `ValueError` / `json.JSONDecodeError` / `TypeError` paths: a business check that fails on a valid reply (a pre-compressed chunk of the wrong length, a compressed tool message with no `tool_call_id`) raises `LLMOutputError` too. Operators still never retry.
- **BREAKING:** replies are validated whole, so what used to be tolerated or dropped piece by piece now fails the reply: a filter reply missing a signal (formerly read as `False`), a compress reply with a non-numeric `quality_score` (formerly `0.5`), a skill operation that is not an object, names an action outside add / update / none, or carries a non-integer `index` or `confidence`, and a profile candidate missing a field or carrying a gate value outside its listed vocabulary. `SkillSkipReason.OP_NOT_DICT`, `OP_UNKNOWN_ACTION` and `UPDATE_INDEX_INVALID`, which only those cases produced, are removed.
- **BREAKING:** every prompt is split into a static system message of rules and a user message holding only the call's data in snake_case XML-style elements. Rule sentences are unchanged; the reply skeletons now parse (placeholder ranges and `|` alternatives became one sample value, the pre-compress skeleton shows the two message shapes), and the skill prompts' ```` ```json ```` fences gave way to "Reply with one JSON object, nothing else." The prompt constants are renamed `AGENT_*_SYSTEM_PROMPT` / `*_USER_PROMPT`; the operator modules re-export the system constants for startup monkey-patching.
- **BREAKING:** the per-call prompt overrides are removed: `prompt_filter` / `prompt_compress` / `prompt_tool_pre_compress` on `AgentCaseExtractor.aextract[_with_reason]`, `prompt_success` / `prompt_failure` / `prompt_maturity` on `AgentSkillExtractor.aextract[_with_reason]`, and `prompt` on `AgentProfileExtractor.aextract`. A whole-prompt template no longer matches a split prompt; patch the system constant instead.
- **BREAKING:** `AgentProfileExtractor` renders each chat line through `everalgo.utils.render_chat_line`, prefixed with the role: `(user) [2024-04-10 14:00 UTC (Wednesday)] Alice: text` instead of `[2024-04-10 14:00:00] user(Alice): text`. A message with text and no timestamp (0) now raises `ValueError` instead of rendering as 1970-01-01.
- **BREAKING:** operators call `LLMClient.achat`, the new name of the protocol method; a custom client passed as `llm` must implement `achat` instead of `chat`. See `everalgo-core` for the contract.
- **BREAKING:** `DetectionResult.should_wait` is renamed `DetectionResult.tail_complete` and its polarity is reversed: `True` means the trailing segment already forms a complete episode and can be closed now, `False` that it is held for the next batch. A trajectory with no `ChatMessage` still returns `None`. See `everalgo-boundary` for the details.
- `pydantic>=2.7` is declared as a direct dependency: the package imports pydantic itself and no longer relies on the transitive pull-in via `everalgo-core`.

### Fixed

- Tool items ahead of a trajectory's first chat message are no longer dropped once a cell is closed: they join the first cell. `AgentBoundaryDetector` started the first cell at the first chat message, so those items landed in neither a cell nor the tail.

## [0.5.1] - 2026-09-16

### Changed

- The minimum `everalgo-core` version is now `0.5.1`, because `truncate_text` uses the shared special-token-safe encoder introduced in that release. This prevents dependency resolution from selecting an older core that cannot import the helper.

### Fixed

- `truncate_text` no longer raises `ValueError` on a trajectory that quotes an `o200k_base` special-token literal such as `<|endoftext|>`. It encoded through the raw tokenizer, so it shared the defect fixed in `everalgo-core` (see that changelog for why `disallowed_special="all"` is wrong at this layer); it now goes through `everalgo._tokenize.encode`. Truncation positions for text without such a literal are unchanged.

## [0.5.0] - 2026-08-19

### Changed

- **BREAKING: `AgentBoundaryDetector.adetect` returns the three-field `DetectionResult`** (see `everalgo-boundary` 0.3.0). Two-value unpacking of its result no longer works.
- `should_wait` is forwarded from the underlying chat-message detection rather than dropped by the trajectory remapping. A trajectory with no `ChatMessage` at all reaches no LLM and returns `None`, matching the semantics its own short-circuit comment already claimed.

## [0.4.0] - 2026-07-30

### Added

- **`aextract_with_reason` on `AgentCaseExtractor` and `AgentSkillExtractor`** (plus `extract_with_reason` sync bridges) — same algorithm as `aextract`, but a rejection comes back as a typed reason instead of only reaching the log. Callers that serve "why is this session's memory empty?" no longer have to scrape log lines to attribute an empty result.
- **`CaseSkipReason` (13 members) and `SkillSkipReason` (12 members)** — one member per rejection gate in the two pipelines, exported from `everalgo.agent_memory`. Reasons are algorithmic: they name the gate, not what the rejection means for any particular product.
- **`CaseExtractionResult`, `SkillExtractionResult`, `OpOutcome`** NamedTuples. `CaseExtractionResult` carries `(cases, reason, detail)`. The skill side carries `(pre_reason, pre_detail, outcomes)` — the `pre_*` pair reports short-circuits that fire before any LLM call, while `outcomes` holds one `OpOutcome` per LLM-proposed operation (so `len(outcomes) == len(operations)`), which keeps the three distinct empty states — quality short-circuit, LLM proposed nothing, every operation dropped — distinguishable. Convenience properties: `SkillExtractionResult.skills` / `.dropped`.
- `detail` payloads carry observed values alongside the thresholds they missed (e.g. `{"rounds": 2, "min_rounds": 3}`) as structured data, so callers can compose user-facing messages without parsing strings.

### Changed

- `aextract` on both extractors is now a thin wrapper over `aextract_with_reason`. Signatures, return types, and behaviour are unchanged.
- Internal (underscore-prefixed) helpers now return their rejection reason: `_should_skip` returns `(reason, detail) | None` rather than a prose string, `_is_worth_extracting` returns `(worth, llm_reason)`, `_compress_experience` returns `(data, reason)`, and `_apply_add` / `_apply_update` return an `OpOutcome` and take a required `op_index` keyword.

### Fixed

- `asgiref` is now declared in `[project.dependencies]`. All four extractor facades derive their sync bridges with `asgiref.sync.async_to_sync`, but the dependency had only ever been satisfied transitively via `everalgo-boundary` — so a resolver that dropped or replaced that path would break `import everalgo.agent_memory` at runtime.

## [0.3.1] - 2026-06-15

### Fixed

- Raise `everalgo-core` dependency lower bound from `>=0.2.0` to `>=0.2.1`. `AgentProfileExtractor` imports `AgentProfilePatch` from `everalgo.types`, which was added in core 0.2.1; the previous floor allowed resolving core 0.2.0 where the type does not exist.

## [0.3.0] - 2026-06-15

### Changed

- **License relicensed from MIT to Apache-2.0** as part of the pre-open-source security audit.
- Case filter tightened: exploration turns and user-correction signals now contribute to case boundary detection, reducing noise in extracted cases.

### Added

- `AgentProfileExtractor`: extract and update agent profiles (SOUL.md / AGENTS.md section-level updates). Supports INIT and UPDATE modes mirroring `ProfileExtractor`. Re-exported from `everalgo.agent_memory`.

## [0.2.0] - 2026-05-27

> First archived changelog. Entries below accumulated since the initial `0.1.0` PyPI release (published manually, without a git tag or per-package changelog), so this `0.2.0` section also consolidates the previously-unarchived `0.1.0` surface.

### Added

- `AgentBoundaryDetector`: facade class that filters a `list[ConversationItem]` to only tool-call-carrying turns, calls `everalgo.boundary.detect_boundaries` on the result, then remaps indices back to the original sequence.
- `AgentCaseExtractor`: 11-step pipeline — heuristic skip check, tool-call-round counting, message trimming / pre-compression, `AgentCase` extraction via LLM. Accepts `llm=` at construction time.
- `AgentSkillExtractor`: add / update / retire operations with `cluster_id` binding; accepts `llm=` at construction time. Configuration is handled via `_SkillCfg` (internal dataclass, not public API); maturity and retire thresholds are passed as per-call kwargs.
- English prompts for case and skill extraction under `agent_memory/prompts/en/`.

### Changed

- `AgentCaseExtractor` and `AgentSkillExtractor` rewired to the unified `MemCell` contract (replacing the previous `AgentMemCell` type).

### Removed

- Broad `except Exception → None` handler in `_evaluate_maturity`: the function now propagates exceptions from the LLM call.
- `asyncio.gather(return_exceptions=True)` in `_pre_compress_to_list`: individual compression errors now propagate instead of being swallowed.
- Fail-open path in `_is_worth_extracting`: the function raises on LLM error instead of returning `True` unconditionally.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-agent-memory/v0.6.0rc1...HEAD
[0.6.0rc1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-agent-memory/v0.5.1...everalgo-agent-memory/v0.6.0rc1
[0.5.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-agent-memory/v0.5.0...everalgo-agent-memory/v0.5.1
[0.5.0]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-agent-memory/v0.4.0...everalgo-agent-memory/v0.5.0
[0.4.0]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-agent-memory/v0.3.1...everalgo-agent-memory/v0.4.0
[0.3.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-agent-memory/v0.3.0...everalgo-agent-memory/v0.3.1
[0.3.0]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-agent-memory/v0.2.0...everalgo-agent-memory/v0.3.0
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-agent-memory/v0.2.0
