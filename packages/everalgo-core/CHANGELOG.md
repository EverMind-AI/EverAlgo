# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `MemCell` pydantic model: ordered `items: list[ConversationItem]` slice produced by boundary detection, plus `timestamp` (Unix epoch ms of the closing item).
- `ConversationItem` discriminated union (`ChatMessage | ToolCallRequest | ToolCallResult`) keyed on the `kind` field.
- `ChatMessage` with `kind="text"` discriminator, multimodal `content: str | list[ContentBlock]`, required `id`, `sender_id`, `sender_name`, and ms-epoch `timestamp`.
- `ContentBlock` / `TextContent` for multimodal message payloads.
- `Episode`, `Foresight`, `AtomicFact`, `Profile` user-memory output types with `ConfigDict(extra="allow")` so additional LLM-emitted keys are preserved without a schema bump.
- `AgentCase`, `AgentSkill` agent-memory output types; `ToolCall`, `ToolCallFunction`, `ToolCallRequest`, `ToolCallResult` agent-trajectory wire types.
- `Candidate`, `FactCandidate`, `ScoredItem`, `RankInput`, `RankOutput` ranking I/O types.
- `ParsedContent`, `RawData`, `RawFile`, `KnowledgeMemory` types reserved for the parser and knowledge distributions.
- `LLMClient` Protocol (async-first: `achat` / `achat_stream`; sync bridge via `asgiref.async_to_sync`).
- `ChatMessage` (LLM-layer), `ChatResponse`, `Usage` wire types for the LLM facade.
- `LLMConfig` pydantic model carrying model name, base URL, API key, and per-call timeout.
- `LLMError` base exception with provider-neutral subclass hierarchy.
- `build_client` factory that auto-detects provider from `LLMConfig.base_url`.
- `SensitiveHeadersFilter`: redacts authorization-style values from log `record.args` mappings; attached to the `everalgo.llm` logger by default (ADR-013).
- `render_prompt` helper: string-replace template rendering that tolerates literal `{` / `}` in template bodies without raising `KeyError`.
- `everalgo.testing.FakeLLMClient`: deterministic LLM replay via a call queue; raises `AssertionError` on unexpected calls.
- `everalgo.testing.CallRecord`: frozen record of a single fake LLM call (prompt, model, usage).
- `everalgo.testing.assert_episode_shape`, `assert_foresight_shape`, `assert_atomic_fact_shape`, `assert_profile_shape`: structural assertion helpers for memory output types.

### Changed

- LLM binding simplified to instance-only injection: `build_client` / constructor `llm=` parameter is the sole binding path. The prior 4-layer resolution (configure / use / current / resolve) was removed in favour of the pattern used by `openai-python`, `anthropic-sdk-python`, LangChain, and Instructor.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/main...HEAD
