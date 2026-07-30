# Changelog

All notable changes to this package are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Episode output language is now defined by what the conversation participants themselves write, not by the conversation's overall character mix. Both prompt variants (generic and user-centred, `en` and `zh`) state the rule at head and tail and explicitly exclude pasted documents, code blocks, logs and long verbatim excerpts from the judgement — a Chinese question quoting a long English document now yields a Chinese episode. The generic variant previously carried no language rule at all. The rule statement itself dropped its trailing "if Chinese, ...; if English, ..." enumeration: the requirement sentence already states the rule language-agnostically, and enumerating two languages as if they were the only options only invited ambiguity for any other output language.
- Timestamps injected into the episode prompts switched to a 24-hour format (`2026-05-29 12:25 UTC (Friday)`) from the previous 12-hour, month-name one (`May 29, 2026 (Friday) at 12:25 PM UTC`). The model echoes the injected anchor's shape into its output, and the 12-hour form produced drifting formats and, in the worst case, a mistranslation of `12:21 PM` into a Chinese "before noon" phrasing. The weekday label is retained because relative-time resolution depends on it. Episode bodies are stored exactly as the LLM wrote them — code prepends nothing — so the format of the times a reader sees is enforced by the prompt rule below.
- Episode prompts: all date examples unified on `YYYY-MM-DD`.
- Episode prompts now require every absolute time that states a clock time to carry the `UTC` zone label (`2024-03-14 15:00 UTC`, never `2024-03-14 15:00` or a bare `15:00`), in all four variants (generic and user-centred × `en` and `zh`). A real-LLM run over mixed-language conversations wrote the narrative's per-event times with the label 25 times and without it 15 times, which is the same "sometimes UTC, sometimes not" inconsistency this release set out to remove. Dates without a clock time need no label, and clock times quoted from a speaker keep their original wording (`at 3:30 PM`) because the speaker's timezone is unknown.
- An instruction forbidding `content` from opening with a date or time of any form was tried and removed. It contradicted the same prompt's demand for a chronological account carrying per-event times, and a real-LLM run showed the model ignoring it in all 15 attempts — correctly, since those times belong to the events it narrates.
- `EpisodeReflector`'s rendered timeline no longer brackets each episode with a `format_natural_language_time` timestamp: that value was `Episode.timestamp` (the span's closing time) in the 12-hour format this release removed, and it disagreed with the times inside the episode body itself. The bracket is dropped; each episode's own narrative is the timeline's only time source.
- `EpisodeReflector`'s prompts now constrain time formatting, which they previously did not mention at all: every time carries over from the episodes being merged exactly as written, absolute times stating a clock time must keep the `UTC` label, and no episode may lose its time in the merge. Without this, reflection was the one path that could reintroduce the drift the extractor prompts had just eliminated — it rewrites the narrative wholesale, and `Episode.timestamp` is not rendered into answer context, so a time dropped here is gone for good.
- `EpisodeExtractor` now rejects an empty or whitespace-only `content` field from the LLM with a `ValueError` instead of silently accepting it. A truncated or empty structured-output response is an extraction failure, and the caller learns far more from raising at the call site than from an empty-body guard firing later in `assert_episode_shape` or the benchmark extract stage.
- `ForesightExtractor` prompts now judge output language the same way `EpisodeExtractor` does. `prompts/en/foresight.py` and `prompts/zh/foresight.py` dropped the hard-coded "if Chinese, ...; if English, ..." enumeration and now judge language only from what the conversation participants themselves write, explicitly excluding pasted documents, code blocks, logs, error messages and long verbatim excerpts even when those dominate the conversation by volume. The `zh` prompt previously carried no standalone language rule at all — its only language requirement was a half-sentence clause tucked into the closing instruction line; that clause is now removed in favour of the same head-and-tail rule statement used by `en`.
- `AtomicFactExtractor` prompts no longer contradict the language rule they state. `prompts/en/atomic_fact.py` and `prompts/zh/atomic_fact.py` (`aextract`) dropped a hard-coded "English"/"Chinese" sentence-language mandate that overrode their own language rule; the `zh` file's rule was previously stated once, weakly, at the very end, and is now stated at head and tail like the `en` file. `prompts/en/atomic_fact_from_text.py` (`aextract_from_text`'s `EVENT_LOG_PROMPT` and `ATOMIC_FACT_FROM_TEXT_PROMPT_EN`, both on production paths) previously forced English output unconditionally with no language rule at all; both now state at head and tail that output follows the language of `EPISODE_TEXT` itself, which for the benchmark path is the already-extracted episode body — fact language now inherits episode language instead of being pinned to English. `aextract`'s raw-conversation path additionally gained the same judgement clauses the episode and foresight prompts use — since it reads the raw conversation and so faces the same mixed-input problem; its own requirement sentence is unchanged. `aextract_from_text` deliberately keeps only the inherit-the-input-language sentence, because language judgement belongs to the layer that reads the raw conversation. All rule statements also dropped the same trailing "if Chinese, ...; if English, ..." enumeration as the episode prompts, for the same reason: the rule sentence is already language-agnostic.
- `ProfileExtractor`'s three prompts now have a language rule each, stated at head and tail, with the rule matching what each call actually does. `PROFILE_INITIAL_EXTRACTION_PROMPT` is the call that fixes a profile's language: it follows the input conversation and carries the same judgement clauses as the episode prompts. `PROFILE_UPDATE_PROMPT` and `PROFILE_COMPACT_PROMPT` instead preserve the existing profile's language and must not switch even when a later conversation arrives in a different language, so they carry no judgement clauses — re-judging downstream is what would split a profile's language. All three rules now bind personality tags to the output language as well; previously tags were only ever shown as English examples (`[Risk-Averse]`), which pulled Chinese profiles towards English tags. The compaction rule also no longer cites "the input conversation content" — `_compact` receives a `Profile` only and never sees a conversation. On the `zh` side all three prompts previously had no language rule at all, while `en` had six statements; they are now mirrored one-for-one.
- `EpisodeReflector`'s prompts now state a language rule at head and tail: merged output stays in the language of the episodes being merged, and an incremental update stays in the language of the existing narrative even when the new episodes arrive in a different language. Previously neither prompt said anything about language at all, so a merged episode's language was whatever the model happened to produce — and reflection writes its result back over the same episode records. Like `aextract_from_text` and the profile update/compaction calls, these prompts inherit rather than judge: they read already-extracted episodes, so they carry no mixed-input judgement clauses. `prompts/zh/reflect.py` also stopped being a re-export of the English constants — it had been a placeholder that handed callers English prompts — and is now a real translation with the same placeholders.
- `prompts/zh/atomic_fact_from_text.py` did not exist, so callers wanting a Chinese variant of the from-text extraction path (`aextract_from_text`) had none — `user_memory`'s last en/zh prompt gap is now closed. It mirrors `prompts/en/atomic_fact_from_text.py` one-for-one, including its inherit-the-input-language rule (EPISODE_TEXT is already an extracted, single-language narrative) and its deliberate absence of mixed-input judgement clauses, which belong to episode extraction instead.
- The judgement clauses now hand the model an operational test for what counts as pasted material, instead of only telling it to ignore such material. The instruction to ignore "pasted documents, code blocks, logs, error messages and long verbatim excerpts" presumes the model can tell which block is pasted, and with unmarked prose it frequently cannot: English participants pasting an unmarked Chinese document produced an English episode in only 48 of 74 real-LLM attempts, against 24 of 25 when the same document sat inside a code fence — the failure was recognition, not willingness. The clauses now define pasted material as a run of two or more consecutive sentences reading as finished prose from elsewhere — explanatory or documentary in tone, addressed to no one in the conversation, advancing no request, answer or decision of its own — and state explicitly that this holds whatever language it is in and whether or not it is wrapped in quotation marks or a code fence. Measured on the failing shape of input: 40 of 40 attempts, against a 48/74 baseline. A regression sweep over all six mixed-language scenarios (Chinese participants with English tracebacks, English with Chinese prose, fenced and unfenced, embedded English terms in Chinese sentences, Japanese with English code) scored 120 of 120 at 20 attempts each. Applied to all four extractors that judge language from a raw conversation — `EpisodeExtractor` (both variants), `ForesightExtractor`, `AtomicFactExtractor.aextract`, `ProfileExtractor` INIT — in both `en` and `zh`, 14 statements in total. The inheriting prompts (`aextract_from_text`, profile update/compaction, reflection) are untouched: they carry no judgement clauses by design.

### Known limitations

- Relative-time resolution uses UTC dates while speakers phrase relative time in their own timezone. For a UTC+8 speaker, references made during local 00:00–07:59 resolve one day early. Fixing this requires the caller to supply a timezone, which the stateless algorithm layer does not receive.
- Whether an output language is retrievable depends on the caller's tokenizer. A Chinese-only tokenizer indexes no tokens at all for Japanese kana, Hangul, Cyrillic, Greek, Arabic, Thai or Hebrew, and truncates accented Latin words.

## [0.3.2] - 2026-07-21

### Fixed

- `ProfileExtractor` no longer leaks other participants' information into the target user's `Profile` in multi-speaker conversations. INIT and UPDATE prompts now receive `sender_id` as an explicit `{target_user}` with speaker-attribution rules, and `aextract` fail-loud validates that `sender_id` is a human (`role == "user"`) speaker present in the input. Removed the never-consumed `TEAM_PROFILE_UPDATE_PROMPT` dead constant.

## [0.3.1] - 2026-06-24

### Fixed

- `EpisodeExtractor`, `ForesightExtractor`, and `AtomicFactExtractor` now pass the first message's timestamp (`memcell.items[0].timestamp`) as the conversation start time to LLM prompts. Previously they passed `memcell.timestamp` (closing time of the slice), which skewed absolute date resolution for relative time expressions.
- Episode prompt: relative time references (e.g. "last Friday", "last summer") are now resolved using each message's own timestamp instead of `conversation_start_time`. Fixes off-by-one-week errors when a MemCell spans multiple days. Per-message timestamps switched from ISO 8601 to human-readable format with weekday labels.

## [0.3.0] - 2026-06-15

### Changed

- **License relicensed from MIT to Apache-2.0** as part of the pre-open-source security audit.

### Added

- `EpisodeReflector`: merge N chronologically-ordered episodes into one accurate narrative. Two modes: INIT (full merge, `old_episode=None`) and UPDATE (incremental, `old_episode=Episode`). Uses OpenAI Structured Outputs via `response_format`. Re-exported from `everalgo.user_memory`.

## [0.2.0] - 2026-05-27

> First archived changelog. Entries below accumulated since the initial `0.1.0` PyPI release (published manually, without a git tag or per-package changelog), so this `0.2.0` section also consolidates the previously-unarchived `0.1.0` surface.

### Added

- `BoundaryDetector`: facade class wrapping `everalgo.boundary.detect_boundaries`; accepts `llm=` at construction time and manages the carry-forward `tail` across calls.
- `EpisodeExtractor`: per-sender Episode fan-out — one LLM call per unique `sender_id` found in the `MemCell`; accepts `llm=` and `prompt=` at construction time.
- `ForesightExtractor`: single `MemCell` → `list[Foresight]`; async with sync bridge via `asgiref.async_to_sync`.
- `AtomicFactExtractor`: single `MemCell` → `list[AtomicFact]`; async with sync bridge.
- `ProfileExtractor`: chronological `list[MemCell]` (last element = most recent) → single `Profile`; single-shot LLM snapshot.
- English and Chinese prompts for all four extractors under `user_memory/prompts/{en,zh}/`.
- `DetectionResult` re-export from `everalgo.boundary`.

### Changed

- `WorkspaceMemCellExtractor` is no longer re-exported from `everalgo.user_memory` (`__all__`). It was an unimplemented stub that raised `NotImplementedError`; it now lives only in `everalgo.boundary.workspace`. Re-adding it once implemented is a non-breaking addition.
- `BoundaryDetector` renamed from `UserBoundaryDetector` (which was itself renamed from `ChatBoundaryDetector`) to match the no-prefix naming convention used across the package.
- `EpisodeExtractor.aextract` parameter `owner_id` renamed to `sender_id` to align with `ChatMessage.sender_id`.
- `ProfileExtractor` signature changed from separate `memcell` + `cluster_episodes` parameters to a single `memcells: Sequence[MemCell]` list, matching the other extractor contracts.
- `Episode`, `Foresight`, `AtomicFact`, `Profile` schemas dropped `parent_id` / `parent_type` fields and the `id` field; schemas now carry only the minimal required fields plus `ConfigDict(extra="allow")`.

[Unreleased]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-user-memory/v0.3.2...HEAD
[0.3.2]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-user-memory/v0.3.1...everalgo-user-memory/v0.3.2
[0.3.1]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-user-memory/v0.3.0...everalgo-user-memory/v0.3.1
[0.3.0]: https://github.com/EverMind-AI/EverAlgo/compare/everalgo-user-memory/v0.2.0...everalgo-user-memory/v0.3.0
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-user-memory/v0.2.0
