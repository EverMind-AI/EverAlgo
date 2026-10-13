# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0rc2] - 2026-10-13

### Added

- `everalgo.utils.InvalidInputError` (a `ValueError` subclass): the one error for data a caller passed that does not meet a precondition, the counterpart of `LLMOutputError`. The same input fails the same way on every run, so a host reads it as a terminal outcome (fix the data or the call, do not retry) instead of guessing from a bare `ValueError` whether it was its own bug. `render_chat_line` (a message with no timestamp), `format_utc_date` (a naive datetime) and `fetch_uri` (an empty uri or a scheme other than http / https) now raise it; their messages read `<whose field> <what is wrong>: <what to do>`, name the offender and never quote user content. Existing `except ValueError` handlers keep working. A repository test fails on any distribution source that raises a bare `ValueError` outside `llm/schema.py`, `llm/openai_compat.py`, `utils/templates.py` and `testing/`.

## [0.6.0rc1] - 2026-10-13

### Added

- `everalgo.types.ExplicitInfoItem` and `everalgo.types.ImplicitTraitItem`, the item models of `Profile.explicit_info` and `Profile.implicit_traits`: a required `description`, an optional label (`category` / `trait`) and grounding (`evidence` / `basis`), and `observed_at` (epoch ms, written by the profile extractor). Both keep `extra="allow"`, so a host's own keys (ids, provenance) ride through untouched.
- `everalgo.llm.StrictOutputModel`: a configuration-only base (`strict=True, extra="forbid"`) for every `response_format` model, nested sub-models included. Without it a default pydantic model fills in missing fields, coerces `"3"` into `3` and ignores extra keys, so what an operator receives would depend on whether the client constrained generation with a strict schema or relied on JSON mode or the prompt; with it, local validation accepts exactly what the strict schema accepts.
- `everalgo.llm.LLMOutputError`, the one error the contract defines: an operator raises it when the reply fails to parse or fails business validation, so a host can tell "the model returned something unusable" from "the call failed" and from an operator bug by type alone. It derives from `Exception` only.
- `OpenAICompatClient(response_format_mode=...)` (`"json_schema"` default / `"json_object"` / `"prompt_only"`) chooses how the reference client requests the reply format, so the same operator can be compared across the three techniques without code changes.
- `everalgo.llm.to_openai_response_format(model, mode)` builds the value of the OpenAI Chat Completions `response_format` parameter for a `response_format` model, so any OpenAI-compatible implementation can reuse it. It returns the field's value, never fields to merge; other APIs will get their own function. In `json_schema` mode the schema is pinned byte-for-byte to what openai SDK 2.35.1 produces for the same model; outside the strict subset (a `RootModel` or other non-object top level, `additionalProperties: true`, a dynamic-key `dict[str, X]`, a multi-arm `allOf`) the builder raises with the field path and the alternative, instead of producing a schema the API rejects with a 400.
- `everalgo.llm.to_anthropic_response_format`, `to_gemini_response_format` and `to_ollama_response_format`: the same builder for Anthropic Messages (`output_config.format`), Gemini `generateContent` (`generationConfig` entries) and Ollama (`format`). Each lists only the modes its API has. Where an API takes only part of JSON Schema (Anthropic, Gemini), the keywords it does not list move into the description, as the Anthropic SDK's `transform_schema` does, and the reply is still validated against the full model.
- A discriminated union (`Field(discriminator=...)`) is the one place every `to_<api>_response_format` departs from the openai SDK 2.35.1 baseline: its `oneOf` becomes `anyOf` and its `discriminator` is dropped. OpenAI strict mode rejects `oneOf` (`'oneOf' is not permitted`) and the SDK passes it through, so such a model used to build a schema OpenAI refuses; each arm fixes the discriminator to its own `const`, so `anyOf` accepts the same replies, and `discriminator` is OpenAPI vocabulary that none of the four APIs lists. A node carrying both `oneOf` and `anyOf` raises `ValueError`. The rewritten shape is not yet confirmed against the live API.
- `OpenAICompatClient` follows every recommendation of the contract, so it is a complete example to copy: it validates each reply against `response_format`, retries a reply that fails validation at most `invalid_reply_retries` times (default 1) with the identical request, never retries a reported refusal, `length` or `content_filter`, and logs every failed validation with its attempt number.
- `everalgo.utils.format_utc_time`, `format_utc_date` and `parse_utc_date`: every time a prompt shows a model now has one shape, `YYYY-MM-DD HH:MM UTC` — 24-hour, to the minute, seconds truncated rather than rounded, with the zone label — plus ` (Wednesday)` where the model resolves relative times (`weekday=True`). It is the shape the Episode prompt already asks the model to write absolute times in, so the times the model reads and the times stored in narratives compare as equal strings. `format_utc_date` renders `YYYY-MM-DD` from epoch milliseconds or an aware datetime; `parse_utc_date` accepts exactly a zero-padded, real `YYYY-MM-DD` and returns aware UTC midnight, or `None`. Rendering and parsing a time now live only in `everalgo.utils.timestamps`; a repository test fails on any other `strftime`, `strptime`, `fromtimestamp` or `%` f-string format spec in package source, with no exception.
- Three tools every operator had a private copy of. `everalgo.utils.render_chat_line` and `speaker` (module `chat_lines`): one transcript line per `ChatMessage`, `[YYYY-MM-DD HH:MM UTC (Weekday)] speaker: text`, the speaker being `sender_name`, else `sender_id`; a timestamp of 0 raises `ValueError` naming only the message's id, and skipping a message with no text stays the caller's choice. `everalgo.utils.tagged(tag, body, /, **attributes)` and `paragraphs(*blocks)` (module `markup`): an XML-style block, open and close tag each alone on its line, and blocks joined by one blank line. `everalgo.llm.parse_reply(model, text, *, operator)` and `describe_errors(exc)` (module `everalgo.llm.output`): validate a reply against its `response_format` model, raising `LLMOutputError` as `<operator> reply is not a valid <Model>: <location>: <message>; …`, never quoting the reply. Boundary and user-memory now call these; their prompts and error messages are unchanged byte for byte, apart from the weekday their chat lines gain.
- `everalgo.testing.conformance`: the executable half of the `LLMClient` contract. An implementation supplies an `LLMClientTestHarness`; `REQUIRED_CHECKS` cover the clauses an operator cannot detect a breach of (explicit values, message delivery, cancellation, reported refusal, reported truncation), `OPTIONAL_CHECKS` the recommendations. The reference client passes every required check.
- `everalgo.utils.text` (`ascii_width`, `decode_bytes`), `everalgo.utils.json` (`json_default`) and `everalgo.utils.http` (`fetch_uri`): general-purpose tools, public now, that lived privately in user-memory, agent-memory and parser. `decode_bytes` tries UTF-8, then GBK, then Latin-1, which decodes any bytes; the GB2312 step and the unreachable lossy fallback of the two former copies are gone, with no change to any result.

### Changed

- **BREAKING: `Profile` gains typed `explicit_info` / `implicit_traits`** (`list[ExplicitInfoItem]` / `list[ImplicitTraitItem]`, default empty) instead of carrying them as bare dicts through `extra="allow"`; `Profile` keeps `extra="allow"` for any other top-level key. Items are models: a host reading them as dicts calls `model_dump()` (`model_dump(exclude_unset=True)` to leave out fields never set), and `profile.model_extra` no longer contains the two buckets. An item without a `description`, or one that is not an object, now fails `Profile` validation instead of riding along until a merge trips over it; so does an item field pydantic cannot coerce to its type (a list as `category`, `1.5` as `observed_at`).
- **BREAKING: `LLMClient.achat(messages, *, response_format, temperature=None, max_tokens=None, **extra) -> str`.** `response_format` is now the pydantic model the operator will validate the reply against, and the method returns the reply text. How to get the model to produce that structure is the implementation's choice; operators always parse the reply themselves. `model` is still not a parameter — it is the host's routing decision.
- **BREAKING: `LLMClient.chat` is renamed `achat`**, with no alias and no sync bridge, so the LLM contract follows the `a`-prefix rule the operators do. `OpenAICompatClient` and `FakeLLMClient` follow, and `everalgo.testing.conformance` calls `achat`. A host's own client must rename its method: one that still offers only `chat` no longer passes `isinstance(client, LLMClient)`, and every operator fails on it with `AttributeError`. The protocol gets no `chat` bridge because hosts implement it — a sync twin would double that work — and operators only call it from async code.
- **BREAKING: `ChatMessage` is renamed `LLMMessage`**, with no alias, so it is no longer confused with the domain type `everalgo.types.chat.ChatMessage`.
- **BREAKING: `LLMError` moves to `everalgo.llm.openai_compat`**, next to the client that raises it, and is no longer exported from `everalgo.llm`: it belongs to the reference implementation, not the contract, which leaves call-failure types to each implementation.
- **BREAKING: everalgo-core depends on `httpx>=0.27` instead of `openai`.** `OpenAICompatClient` speaks OpenAI Chat Completions over httpx, so a host that brings its own client installs no provider SDK; a host that imported `openai` through everalgo-core must declare it itself. The client owns a connection pool, released with `aclose()` (or `contextlib.aclosing`). A connection error, a timeout, HTTP 408, 409, 429 or 5xx, a 200 with no choices, or an aborted generation is resent at most `transport_retries` times (default 2), with the openai SDK's backoff and its `Retry-After` limit.
- **BREAKING: `OpenAICompatClient` takes keyword arguments** — `model`, `api_key` (a plain `str`), `base_url` (default `https://api.openai.com/v1`), `temperature`, `max_tokens`, `timeout`, `response_format_mode`, `invalid_reply_retries`, `transport_retries` — as the openai and anthropic SDK clients do, instead of an `LLMConfig`. `temperature` now defaults to `None`, which leaves it out of the request, instead of `0.0`; pass `temperature=0.0` to keep the old behaviour. A negative retry count raises `ValueError`.
- **BREAKING: `everalgo.utils` is the home of general-purpose tools, one kind per module**, and a public API: the operator packages are separate distributions, so what they share cannot live in a private module. `everalgo.prompts` becomes `everalgo.utils` (`render_prompt` in `templates`), and each operator package's `prompts/` keeps holding prompt text only.
- **BREAKING: the module-private `everalgo._tokenize` becomes `everalgo.utils.tokens`**, exporting `encode`, `count_tokens` and `truncate_text`. `truncate_text` moves in from `everalgo.agent_memory._text`, whose private access to the tokenizer instance it removes; `force_split`, which nothing called, is removed.
- **BREAKING: the reference client moves to `everalgo.llm.openai_compat`**, a flat module named after the API it speaks; `everalgo.llm.providers` is gone. `providers` promised a family of vendor adapters where there is one reference `LLMClient`.
- `OpenAICompatClient` never returns a generation the upstream reports as unfinished, even if the text is closed, valid JSON — a reply cut off right after its closing brace still parses, so only the client can catch it. `length` and `content_filter` raise `LLMError` at once, since the identical request hits them again; `abort` and `error`, the server giving up on a generation it could have finished, are resent within `transport_retries`. It also raises when the content is absent instead of returning `""`. A missing finish reason is accepted as is.

### Removed

- **BREAKING: `ChatResponse` and `Usage`.** Operators read only the content; the model, finish reason and usage are the host's concern and already visible to the implementation.
- **BREAKING: `TextFormat`, `JsonObjectFormat`, `JsonSchemaSpec`, `JsonSchemaFormat`, `ResponseFormat` and `json_schema_format`.** Wire-shaped formats are an implementation detail now; the `to_<api>_response_format` builders replace `json_schema_format`.
- **BREAKING: `UnsupportedParameterError`.** Nothing raised it; an implementation that cannot deliver a content part raises its own error type.
- **BREAKING: `everalgo.llm.parse` (`parse_llm_json_object`, `extract_final_answer`).** Its lenient extraction — strip a fence, then fall back to the outermost braces — contradicts the contract, under which an operator parses the reply strictly with `model_validate_json`; no operator called it any more. `extract_final_answer` had one caller, the benchmark answer stage, which now inlines it.
- **BREAKING: `FactCandidate` and `RankInput.episode_to_facts` are removed.** They fed an episode → atomic-fact expansion in the episodic ranker that no longer exists; no ranker read them, and hosts that expand do so on their own recall.
- **BREAKING: `ChatMessage.content` is `str`; `ContentBlock`, `TextContent` and the private `render_content` are removed.** The union was a placeholder for native multimodal input that nothing ever filled: hosts parse images, audio and files into text before handing a message over, and every operator read `content` through `render_content` only to get that text back. Native multimodal messages get their own types once they are designed; widening `str` to a union then breaks no caller.
- **BREAKING: `format_message_timestamp` is removed** (`YYYY-MM-DD HH:MM:SS` in UTC, no zone marker), from `everalgo.llm` and everywhere else. Every prompt instant now renders with `everalgo.utils.format_utc_time` and every chat line with `render_chat_line`; agent-memory, its last caller, moved to those. `format_iso_timestamp`, `format_natural_language_time` and `Lang` are removed too: nothing called them.
- **BREAKING: `format_atomic_fact_time` is removed** (`March 10, 2024(Sunday) at 02:00 PM`, a 12-hour clock with no zone label). AtomicFact's `TIME` is now `format_utc_time(..., weekday=True)`; nothing else called it.
- **BREAKING: `SensitiveHeadersFilter` is removed from the `everalgo.llm` logger.** It redacted mapping-shaped log arguments, and no record in the library has any: the library never writes a credential, a request or a response body into a log record, which is the only redaction that holds.
- **BREAKING: `LLMConfig` and `build_client`.** The config object was one more type between the caller and the client, and the factory could only ever return `OpenAICompatClient`; construct the client directly. `LLMConfig.extra` goes with them: per-call `**extra` already reaches the request.
- **BREAKING: `everalgo.prompts.validator` (`check_placeholders`, `check_length`) is removed.** Nothing in the repository called either; a host that validates its own templates checks them itself.
- The empty module `everalgo.config` is removed.

## [0.5.1] - 2026-09-16

### Fixed

- `count_tokens` and `force_split` no longer raise `ValueError` on text that quotes an `o200k_base` special-token literal such as `<|endoftext|>` or `<|endofprompt|>`. `Encoding.encode` defaults to `disallowed_special="all"`, a guard meant for a caller assembling a prompt; this module only measures and slices text, so the default turned ordinary tokenizer documentation and bug reports into a permanent failure, retried forever on the same input. The new module-private `everalgo._tokenize.encode` passes `disallowed_special=()` and is the single sanctioned encode in every published distribution. Such a literal is now counted as the several ordinary tokens it is made of rather than as one control token; nothing else about token counts, split points, or truncation positions changes.

## [0.5.0] - 2026-08-19

### Changed

- **BREAKING: `Episode.summary` is now a declared, required field.** It used to arrive through `extra="allow"`, which let a construction site omit it silently; requiring it makes the type-checkers name every site that does. External callers constructing `Episode` themselves must pass it. `assert_episode_shape` gained a matching non-empty check (whitespace-only counts as empty — a blank preview reads as a bug downstream, not as an absent field). The field's meaning is unchanged in principle and fixed in practice: a display preview of `episode`, faithful to it and readable without it. For 0.1 through 0.4 nothing produced one — the user-memory prompts never asked for it, so every caller received a blind `episode[:200]` slice: cut mid-word in English, and in Chinese a verbatim copy of most of the body, since 200 characters is most of an episode there. See `everalgo-user-memory` for the prompt side.

### Fixed

- `everalgo.llm.format` module docstring no longer claims `format_iso_timestamp` is used by the Episode prompt's conversation block — it has no in-tree consumer, and the Episode prompt builds its own format.

## [0.4.0] - 2026-06-24

### Removed

- **BREAKING:** `retry_on_json_parse_failure` removed from `everalgo.llm.parse`. Callers should use [tenacity](https://github.com/jd/tenacity) or equivalent for retry policies.
- **BREAKING:** `allm_judge` and `JudgeResult` removed from `everalgo.testing`. Evaluation infrastructure now lives in the caller layer.

## [0.3.0] - 2026-06-16

### Added

- `CategorySpec` type in `everalgo.types`: caller-owned classification taxonomy entry `(id, description)` for document-level categorization.
- `KnowledgeMemory.category_id` field: document-level category denormalized onto every topic-tree node.

## [0.2.1] - 2026-06-15

### Changed

- **License relicensed from MIT to Apache-2.0** as part of the pre-open-source security audit.

### Added

- `AgentProfile` type in `everalgo.types` to support agent-memory's `AgentProfileExtractor`.

## [0.2.0] - 2026-05-27

> First archived changelog. Entries below accumulated since the initial `0.1.0` PyPI release (published manually, without a git tag or per-package changelog), so this `0.2.0` section also consolidates the previously-unarchived `0.1.0` surface.

### Added

- `MemCell` pydantic model: ordered `items: list[ConversationItem]` slice produced by boundary detection, plus `timestamp` (Unix epoch ms of the closing item).
- `ConversationItem` discriminated union (`ChatMessage | ToolCallRequest | ToolCallResult`) keyed on the `kind` field.
- `ChatMessage` with `kind="text"` discriminator, multimodal `content: str | list[ContentBlock]`, required `id`, `sender_id`, `sender_name`, and ms-epoch `timestamp`.
- `ContentBlock` / `TextContent` for multimodal message payloads.
- `Episode`, `Foresight`, `AtomicFact`, `Profile` user-memory output types with `ConfigDict(extra="allow")` so additional LLM-emitted keys are preserved without a schema bump.
- `AgentCase`, `AgentSkill` agent-memory output types; `ToolCall`, `ToolCallFunction`, `ToolCallRequest`, `ToolCallResult` agent-trajectory wire types.
- `Candidate`, `FactCandidate`, `ScoredItem`, `RankInput`, `RankOutput` ranking I/O types.
- `ParsedContent`, `RawData`, `RawFile`, `KnowledgeMemory` types reserved for the parser and knowledge distributions.
- `LLMClient` Protocol — a single async method `chat(messages, *, model, temperature, max_tokens, response_format, **extra) -> ChatResponse` (PEP 544 structural contract). No `a` prefix (mirrors the OpenAI SDK client interface; callers implement it and inject it, EverAlgo never wraps it in a sync bridge). No streaming method.
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

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-core/v0.6.0rc2...HEAD
[0.6.0rc2]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-core/v0.6.0rc1...everalgo-core/v0.6.0rc2
[0.6.0rc1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-core/v0.5.1...everalgo-core/v0.6.0rc1
[0.5.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-core/v0.5.0...everalgo-core/v0.5.1
[0.5.0]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-core/v0.4.0...everalgo-core/v0.5.0
[0.4.0]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-core/v0.3.0...everalgo-core/v0.4.0
[0.3.0]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-core/v0.2.1...everalgo-core/v0.3.0
[0.2.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-core/v0.2.0...everalgo-core/v0.2.1
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-core/v0.2.0
