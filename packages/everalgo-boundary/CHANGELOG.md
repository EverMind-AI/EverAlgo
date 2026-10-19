# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0rc4] - 2026-10-19

### Changed

- The batch reply format bounds `reasoning`: it names the topics in order, each as a short label with its message range, then the message numbers after which the topic changes, and nothing more; the worked examples' `reasoning` takes the same shape (`"login bug (1-5); lunch plans (6-7). Changes after 5."`). The previous wording set no length, and the model wrote each topic out as a full sentence. The reply's keys and types are unchanged.
- The batch prompt's first worked example gains the other speaker's courtesy reply ("Anytime.") after the thanks, and its split moves after that reply, so it shows a courtesy reply staying with the topic it closes; all seven of its messages now fall on one day.

### Removed

- The batch prompt's cross-day rule ("adjacent messages have different calendar dates — MUST split at the date boundary", highest priority) is gone, and the first worked example's `reasoning` no longer cites the new day; its messages now all fall on one day, and it splits after the courtesy reply, where the topic changes. A date change alone is no longer a split signal: the host already closes an idle conversation, and the UTC date the timestamps carry is not the user's day, so the rule split overnight conversations at a fixed local hour (08:00 in UTC+8). The other split signals, the long-gap rule included, are unchanged.
- The batch prompt's "do not split for" line on transition phrases ("by the way", "oh also" — "usually continue the current episode") is gone. It stated a prior rather than a test, and the model took a transition phrase as a cue to split. The other two "do not split for" lines are unchanged.

## [0.4.0rc3] - 2026-10-13

### Changed

- `prompts.en.chat.assemble_system_prompt` builds the `<custom_instructions>` block with core's `everalgo.utils.append_custom_instructions` instead of its own copy of the logic. The rendered system message is byte-for-byte unchanged.
- Requires `everalgo-core>=0.6.0rc3` (the floor was `0.6.0rc2`): the package imports `everalgo.utils.append_custom_instructions`, which `0.6.0rc2` does not define.

## [0.4.0rc2] - 2026-10-13

### Changed

- A message with text but no timestamp now raises `everalgo.utils.InvalidInputError` (a `ValueError` subclass, so existing handlers keep working) instead of a bare `ValueError`; the message names the chat message's id and what to do about it.
- Requires `everalgo-core>=0.6.0rc2` (the floor was `0.6.0rc1`): the error above is raised by core's `render_chat_line`, which on `0.6.0rc1` still raises a bare `ValueError`.

## [0.4.0rc1] - 2026-10-13

### Changed

- **BREAKING:** `BoundaryDetector` lives in this package, as the one class for both boundary paths: `adetect` / `detect` (batch, forwarding to `adetect_boundaries`) and `adetect_step` / `detect_step` (incremental, with its smart-mask, force-split, cut-and-bridge and time-gap handling). `everalgo-user-memory` and `everalgo-agent-memory` re-export this class unchanged; `everalgo-agent-memory`'s `AgentBoundaryDetector` is removed. The step path's `should_end=True but topic_summary is empty` warning now logs on the `everalgo.boundary` logger instead of `everalgo.user_memory.boundary`.
- **BREAKING:** `detect_boundaries` is now a sync bridge; await `adetect_boundaries` instead. The native-async function is renamed `adetect_boundaries`, so the name follows the `a`-prefix convention, and `detect_boundaries = async_to_sync(adetect_boundaries)` takes the old name, as `BoundaryDetector.detect` does for `adetect`. `await detect_boundaries(...)` inside an event loop now raises `RuntimeError` from `asgiref`.
- **BREAKING:** `adetect_boundaries` (and `detect_boundaries`, `BoundaryDetector.adetect`) takes `Sequence[ConversationItem]` instead of `list[ChatMessage]`. Only the `ChatMessage` items are rendered for the model; `ToolCallRequest` / `ToolCallResult` items stay in the returned cells: between two chat messages they go with the earlier one, ahead of the first chat message with the first cell, after the last one into the tail (into the last cell when `is_final=True`). Every item lands exactly once. A chat-only input gives the same cells, tail, verdict on the tail and prompt as before. Without any `ChatMessage` no LLM is called, every item goes to the tail and `tail_complete` is `None`.
- **BREAKING:** `DetectionResult.tail` is typed `list[ConversationItem]` instead of `list[ChatMessage]`; for a chat-only input it still holds only `ChatMessage`s. A caller that feeds the step path's tail back in as `history: list[ChatMessage]` may need to narrow it for a type checker.
- **BREAKING:** `DetectionResult.should_wait` is renamed `tail_complete` and its polarity is reversed; the batch reply's `should_wait` key is renamed `tail_complete` with it. `tail_complete` says whether the trailing segment (after the last boundary, or the whole window when there is none) already forms a complete episode: `True` means it can be closed into a MemCell now, `False` means it is not complete yet and is held for the next batch. `None` still means no path judged it: the step path, and inputs with no `ChatMessage`. A caller that read `should_wait` as a `bool` reads `not tail_complete`, and keeps `None` as `None`. The rule behind the field changes too: it used to ask whether the segment had too little information to place in an episode, which missed a segment that is clear but not finished. A segment that only opens a topic the next messages will continue is not complete, however clear the opening; short replies leave a segment incomplete only when nothing before them in the segment is what they answer; and a segment ending with an acknowledgement or farewell that closes its own topic is complete. A third worked example shows a thanks closing one episode and a new request that is not complete yet.
- **BREAKING:** `adetect_boundaries` and `adetect_boundary_step` request their reply as a strict pydantic model — `{"reasoning", "boundaries", "tail_complete"}` and `{"reasoning", "should_end", "confidence", "topic_summary"}` — passed as `response_format`, and validate it with `model_validate_json` instead of pulling the first flat `{...}` out of the text. A reply that fails now raises `everalgo.llm.LLMOutputError` instead of `ValueError` / `TypeError`; its message names each failing location, never the reply's content. Replies previously repaired are now rejected: prose or a ```` ```json ```` fence around the object, an extra key, a missing key (the step reply's `reasoning`, `confidence` and `topic_summary` used to default to `""`, `1.0` and `""`), a `null` boundary (used to be skipped), a numeric string (used to be coerced) or a non-boolean flag (used to go through `bool()`). Out-of-range, duplicate and unsorted boundaries are still dropped, deduplicated and sorted after parsing. The `LLMClient` passed as `llm` must implement the current contract: `achat(messages, *, response_format: type[BaseModel]) -> str`. See `everalgo-core` for the contract.
- **BREAKING:** operators call `LLMClient.achat`, the new name of the protocol method; a custom client passed as `llm` must implement `achat` instead of `chat`. See `everalgo-core` for the contract.
- **BREAKING:** each boundary call sends a system message with the rules and a user message with the data, instead of one user message holding both. `BATCH_BOUNDARY_DETECT_PROMPT_EN` and `STEP_BOUNDARY_DETECT_PROMPT_EN` in `everalgo.boundary.prompts.en.chat` are removed; each splits into `*_SYSTEM_PROMPT` (the rules, no placeholder) and `*_USER_PROMPT` (data slots only: `{{count}}` and `{{messages}}` for batch, `{{conversation_history}}`, `{{time_gap_info}}` and `{{new_messages}}` for step). Every rule sentence is kept except the batch `should_wait` rule, replaced by `tail_complete`, and the batch output-language rule, removed (both below); Markdown headings and emphasis became XML-style tags, the reply skeletons are now parseable JSON, and the batch data block now states its message count (`<messages count="N">`, the chat messages in the window the model sees).
- **BREAKING:** `prompt=` is removed from `adetect_boundaries`, `detect_boundaries`, `adetect_boundary_step`, `BoundaryDetector.adetect` / `detect` and `BoundaryDetector.adetect_step` / `detect_step`; passing it raises `TypeError`. Each takes `custom_instructions=` in its place, which appends to the bundled rules instead of replacing them: the text is sent verbatim, never rendered as a template (a `{{...}}` in it stays literal), as a `<custom_instructions>` block that is always the system message's last; the user message is unchanged. `None` or a blank string adds nothing, so the system message is byte-for-byte the bundled rules. No way is left to replace the bundled rules per call: a replacement also dropped the reply shape the operator parses. `everalgo.boundary.prompts.en.chat.assemble_system_prompt` builds the message. The reserved `WorkspaceMemCellExtractor` stub takes `custom_instructions=` in place of `prompt=` too.
- **BREAKING:** message lines show time as `YYYY-MM-DD HH:MM UTC` and carry the weekday, `[2026-07-15 14:00 UTC (Wednesday)] speaker: text`, instead of `[YYYY-MM-DD HH:MM:SS] speaker: text` with no zone marker, on both paths. The line is rendered by `everalgo.utils.render_chat_line` (seconds truncated), the line every operator now shows a model, and the batch prompt's worked examples use the same shape instead of ISO 8601 `...Z`. A chat message with text and a timestamp of 0 now raises `ValueError` (`chat message '<id>' has no timestamp`) instead of rendering as 1970-01-01; a message with no text is still dropped, its timestamp never read. Replies are parsed by `everalgo.llm.parse_reply`; the `LLMOutputError` messages are unchanged.
- **BREAKING:** the batch prompt's output-language rule is removed. `boundaries`, `tail_complete` and `should_end` carry no language, and `reasoning` is never read.
- **BREAKING:** the default hard limits are 65536 tokens and 500 messages instead of 8192 and 50, on both paths (`hard_token_limit` / `hard_msg_limit` of the batch path, `hard_token_limit` / `hard_message_limit` of `adetect_step`). Batch inputs between the old and new limits now reach the model in one window instead of being force-split first; on the step path they go to the model instead of being force-cut. Pass the old values explicitly to keep the old cuts.
- **BREAKING:** the `everalgo.boundary.chat` module is removed; its contents are split across the private `_detector`, `_segments` and `_render` modules. Import from `everalgo.boundary`, whose public names are unchanged apart from the added `BoundaryDetector`.
- The batch reply format asks `reasoning` to first name the topics the conversation goes through, in order, then state where they change, instead of one sentence explaining all boundary decisions; the worked examples' `reasoning` follows the same shape. The reply's keys and types are unchanged.
- Each LLM call logs one INFO line on the `everalgo.boundary` logger, counts and flags only: `boundary batch reply: messages=… boundaries=… tail_complete=… dropped_out_of_range=…` and `boundary step reply: should_end=… confidence=…`.

### Removed

- **BREAKING:** the Chinese batch prompt `everalgo.boundary.prompts.zh.chat.CHAT_BOUNDARY_DETECT_PROMPT_ZH` and the `prompts.zh` subpackage: nothing selected it, and it had drifted from the English prompt.

### Fixed

- The step prompt's time-gap line uses the singular for a count of one (`Time gap: 1 minute`, not `1 minutes`).

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

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-boundary/v0.4.0rc4...HEAD
[0.4.0rc4]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-boundary/v0.4.0rc3...everalgo-boundary/v0.4.0rc4
[0.4.0rc3]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-boundary/v0.4.0rc2...everalgo-boundary/v0.4.0rc3
[0.4.0rc2]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-boundary/v0.4.0rc1...everalgo-boundary/v0.4.0rc2
[0.4.0rc1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-boundary/v0.3.0...everalgo-boundary/v0.4.0rc1
[0.3.0]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-boundary/v0.2.1...everalgo-boundary/v0.3.0
[0.2.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-boundary/v0.2.0...everalgo-boundary/v0.2.1
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-boundary/v0.2.0
