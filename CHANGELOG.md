# Changelog

All notable changes to the EverAlgo monorepo are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Each distribution follows its
own [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html) cadence — there is no umbrella
version.

Per-distribution changelogs are the source of truth. This file is a navigation index.

## Distribution status

The table tracks the current version declared in each `packages/everalgo-*/pyproject.toml`. All eight distributions are published to PyPI.

| Distribution | Version | Changelog |
|---|---|---|
| `everalgo-core` | 0.5.0 | [packages/everalgo-core/CHANGELOG.md](packages/everalgo-core/CHANGELOG.md) |
| `everalgo-boundary` | 0.3.0 | [packages/everalgo-boundary/CHANGELOG.md](packages/everalgo-boundary/CHANGELOG.md) |
| `everalgo-clustering` | 0.2.1 | [packages/everalgo-clustering/CHANGELOG.md](packages/everalgo-clustering/CHANGELOG.md) |
| `everalgo-rank` | 0.4.1 | [packages/everalgo-rank/CHANGELOG.md](packages/everalgo-rank/CHANGELOG.md) |
| `everalgo-user-memory` | 0.7.0 | [packages/everalgo-user-memory/CHANGELOG.md](packages/everalgo-user-memory/CHANGELOG.md) |
| `everalgo-agent-memory` | 0.5.0 | [packages/everalgo-agent-memory/CHANGELOG.md](packages/everalgo-agent-memory/CHANGELOG.md) |
| `everalgo-parser` | 0.2.1 | [packages/everalgo-parser/CHANGELOG.md](packages/everalgo-parser/CHANGELOG.md) |
| `everalgo-knowledge` | 0.1.1 | [packages/everalgo-knowledge/CHANGELOG.md](packages/everalgo-knowledge/CHANGELOG.md) |

## Minor release — 2026-08-20 (user-memory)

`ProfileExtractor`'s three prompts are rewritten around one definition of what a profile is: a portrait of a
person, not a log of what happened. The reported symptom was a real profile holding 22 `explicit_info` items
with 20 of them under two category names; the cause was not label choice but that every operation in a
conversation was being restated as a standing capability — 8.0 items per run on a purely operational corpus,
86% of them claims like `负责测试执行` that the person never made. Category collision followed from those items
all being the same kind of thing.

No public API changes. What changes is what the extractor writes: an operational conversation can now
correctly yield nothing; a stated fact lands in `explicit_info` rather than being rephrased as a disposition;
no item names the subject; facts about one dimension merge into one item; transient statements are dropped.
The bump is minor rather than patch because compaction now runs at the item limit instead of 50% beyond it, so
a profile between 31 and 45 items makes one more LLM call than it used to.

Verified across five models from four vendors and three corpora (the tuned Chinese software-engineering one, a
non-technical Chinese one, and an English one): 45 must-keep checks all at 20/20, capability phrasing and
subject naming at 0.00 per run everywhere. One limitation stands and is documented in the module docstring — a
record of the work still reaches `explicit_info` at 1.4-1.8 items per run on a purely operational corpus, and
four rules written against it each regressed something else.

| Distribution | Version | Bump |
|---|---|---|
| `everalgo-user-memory` | 0.7.0 | minor — profile prompt rewrite; compaction threshold now the item limit |

Detail in [packages/everalgo-user-memory/CHANGELOG.md](packages/everalgo-user-memory/CHANGELOG.md).

## Minor release — 2026-08-19 (boundary)

Three distributions, one change. Boundary detection asks the model whether the trailing segment it could
not close is even interpretable yet — `should_wait` — and has always required the answer to be present,
rejecting a response that omits it. `detect_boundaries` then read the boundary indices and dropped that
answer on the floor, so no caller since 0.1 could see a verdict the model was computing and the library
was enforcing. It matters because a non-empty tail is the ordinary case, while `should_wait` marks the
narrower one: a trailing segment holding only a media placeholder, a bare "ok", or a system notification,
which a caller should not extract from and cannot otherwise distinguish.

`DetectionResult` therefore gains a third field, which breaks two-value unpacking — `cells, tail = ...`
now raises. Named access is unaffected. The field is `bool | None` rather than `bool`: the single-step
detection path answers a different question and never evaluates the trailing segment, so it reports
`None` instead of guessing. Inverting its `should_end` was considered and rejected — both flags default to
"no" for unrelated reasons, so the inversion would report "wait" on nearly every step and, on the steps
that do close an episode, claim "no need to wait" about a tail it never looked at.

The two consumer distributions move for the same reason and raise their `everalgo-boundary` floor to
`>=0.3.0`. Unlike the previous release, an older boundary here is a hard failure rather than something
absorbed silently: `DetectionResult` is a `NamedTuple`, and the arity mismatch raises. Per-distribution
detail in each package's CHANGELOG.

| Distribution | Version | Bump |
|---|---|---|
| `everalgo-boundary` | 0.3.0 | minor (breaking) — `DetectionResult.should_wait`; two-value unpacking removed |
| `everalgo-user-memory` | 0.6.0 | minor (breaking) — three-field `DetectionResult`; `adetect_step` reports `should_wait=None` |
| `everalgo-agent-memory` | 0.5.0 | minor (breaking) — three-field `DetectionResult`; verdict forwarded through trajectory remapping |

## Minor release — 2026-08-19 (episode summary)

Two distributions updated together, because the change spans both. `Episode.summary` was supposed to be a
short preview of an episode; nothing ever wrote one. The consumer side was ported from an opensource
extractor whose LLM contract had the field, the prompt side was not, so for every release from 0.1 through
0.4 the field was a blind `episode[:200]` slice — cut mid-word in English, and in Chinese, where 200
characters is most of an episode, a verbatim copy of the body. Callers were reading that as a summary.

The extractor and the reflector now require the model to write one, and `Episode.summary` is a declared,
required field rather than an `extra="allow"` passenger. Required is deliberate: it makes the type-checkers
name every construction site that forgets it, where a default would let the field decay back into the blank
preview being fixed. Both distributions are therefore breaking, and `everalgo-user-memory` raises its
`everalgo-core` floor to `>=0.5.0` — the guarantee it advertises is not true against an older core.

The same `everalgo-user-memory` release carries the output-language work: an `output_language` argument on
every LLM-backed operator, backed by roughly 14000 measured extractions, and the removal of the `prompts/zh/`
tree it makes redundant. Naming the language measured zero wrong-language output; leaving the model to infer
it bottoms out around 10%. Per-distribution detail in each package's CHANGELOG.

| Distribution | Version | Bump |
|---|---|---|
| `everalgo-core` | 0.5.0 | minor (breaking) — `Episode.summary` declared and required; `assert_episode_shape` checks it |
| `everalgo-user-memory` | 0.5.0 | minor (breaking) — model-written `summary` on episode + reflect; `output_language` on every operator; `prompts/zh/` removed |

## Patch release — 2026-08-04

One distribution updated. `ProfileExtractor`'s UPDATE path applied each operation to a list it was mutating as it went, so a delete shifted the index every later operation addressed — silently editing or removing the wrong profile item and returning a structurally valid profile with wrong contents that a caller could not detect, work around, or recover from. Indices now resolve against the profile snapshot the model was actually shown. The same release closes five routes to duplicate profile items, including two that no operation could reach before: the INIT path, which wrote the LLM's output through unchecked, and compaction, the very mechanism meant to collapse duplication. Because an update now deduplicates the whole bucket, a profile already polluted before this release heals on its next update. Per-distribution detail in the package CHANGELOG.

| Distribution | Version | Bump |
|---|---|---|
| `everalgo-user-memory` | 0.4.1 | patch — profile update indices resolved against the original snapshot; duplicate items removed on every write path; per-op rejection warnings |

## Minor release — 2026-07-30

Two distributions updated, independently. Every prompt in `everalgo-user-memory` now judges output language from what the conversation participants themselves write — with an operational test for what counts as pasted material — and every absolute clock time an extractor emits carries the `UTC` label in a 24-hour format; the `zh` prompt set reached parity with `en`. `everalgo-agent-memory` gains `aextract_with_reason`, which returns a typed rejection reason instead of leaving an empty result only explicable from the log. Per-distribution detail in each package's CHANGELOG.

| Distribution | Version | Bump |
|---|---|---|
| `everalgo-user-memory` | 0.4.0 | minor — participant-anchored output language; `UTC`-labelled 24-hour times; `prompts/zh/atomic_fact_from_text.py` added |
| `everalgo-agent-memory` | 0.4.0 | minor — `aextract_with_reason` + typed `CaseSkipReason` / `SkillSkipReason`; `asgiref` declared explicitly |

## Patch release — 2026-07-21

One distribution updated. Per-distribution detail in the package CHANGELOG.

| Distribution | Version | Bump |
|---|---|---|
| `everalgo-user-memory` | 0.3.2 | patch — `ProfileExtractor` scoped to `sender_id`; cross-owner leakage fix |

## Patch release — 2026-06-24

Three distributions updated. Per-distribution detail in each package's CHANGELOG.

| Distribution | Version | Bump |
|---|---|---|
| `everalgo-core` | 0.4.0 | minor — BREAKING: removed `retry_on_json_parse_failure` + `allm_judge`/`JudgeResult` |
| `everalgo-rank` | 0.4.1 | patch — internal variable rename (`member_to_cluster`) |
| `everalgo-user-memory` | 0.3.1 | patch — episode prompt timestamp anchor fix |

## Coordinated minor — 2026-06-16

Three distributions updated. `everalgo-knowledge` is published to PyPI for the first time. Per-distribution detail in each package's CHANGELOG.

| Distribution | Version | Bump |
|---|---|---|
| `everalgo-core` | 0.3.0 | minor — `CategorySpec` type + `KnowledgeMemory.category_id` field |
| `everalgo-rank` | 0.4.0 | minor — category-aware retrieval (`acategory_retrieve`) |
| `everalgo-knowledge` | 0.1.1 | **first public release** — `KnowledgeExtractor` pipeline + `aclassify_category` |

## Coordinated patch + minor — 2026-06-15

Seven distributions updated. All relicensed from MIT to Apache-2.0. Per-distribution detail in each package's CHANGELOG.

| Distribution | Version | Bump |
|---|---|---|
| `everalgo-core` | 0.2.1 | patch |
| `everalgo-boundary` | 0.2.1 | patch |
| `everalgo-clustering` | 0.2.1 | patch |
| `everalgo-rank` | 0.3.1 | patch |
| `everalgo-parser` | 0.2.1 | patch |
| `everalgo-user-memory` | 0.3.0 | minor — `EpisodeReflector` |
| `everalgo-agent-memory` | 0.3.0 | minor — `AgentProfileExtractor` + case filter |

## [everalgo-rank/0.3.0] - 2026-05-28

Per-distribution minor bump of `everalgo-rank` only. Other distributions are unchanged.

### Changed (breaking)

- `everalgo-rank`: skill ranker's standalone post-rerank verify stage is removed; relevance grading folds into a single LLM pass with a new `min_rerank_score` hard-threshold gate. Public API `averify` / `verify` and the `enable_verify` / `verify_threshold` / `verify_prompt` parameters are gone. See [packages/everalgo-rank/CHANGELOG.md](packages/everalgo-rank/CHANGELOG.md#030---2026-05-28) for migration.

### Changed

- License relicensed from MIT to Apache-2.0.

## [0.2.0] - 2026-05-27

Coordinated release-readiness baseline across the seven published distributions. Per-distribution detail lives in each package's CHANGELOG; highlights:

### Changed (breaking)

- `everalgo-clustering`: `cluster_by_geometry` is now synchronous (pure compute, no I/O).
- `everalgo-boundary` / `everalgo-user-memory`: the unimplemented `WorkspaceMemCellExtractor` stub is removed from the public API (reachable only via explicit submodule import).
- `everalgo-core`: the empty `everalgo.protocols` placeholder is removed.

### Changed

- Packaging metadata enriched across all distributions (keywords, full classifiers, author email).
- Coverage gate centralised in `pyproject.toml`; CI now tests Python 3.12 / 3.13 / 3.14; tag-triggered PyPI publish via Trusted Publishing added.
- Community health files added (`SECURITY.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `CITATION.cff`).
- `everalgo-knowledge` marked not-for-publish (`Private :: Do Not Upload`); namespace reserved only.

## [0.1.0] - 2026-05-17

### Added

- **Monorepo scaffold** — 8-distribution uv virtual workspace with a shared lockfile and PEP 420 native namespace packages under `everalgo.*`.
- **Release infrastructure** — `cliff.toml` for per-distribution CHANGELOG generation, `py.typed` markers, pre-commit hook (`ruff check --fix` + `ruff format` + standard sanitisers).
- **Logging conventions (ADR-013)** — `NullHandler` on every subpackage logger; `SensitiveHeadersFilter` on the `everalgo.llm` logger; ruff rules `G` + `LOG` + `TRY` enforce library-safe logging patterns.
- **6 runnable quickstart examples** under `examples/` covering boundary detection, geometry clustering, Episode extraction, AgentCase extraction, rerank, and the full user-memory pipeline. All use `FakeLLMClient`; no API key required.
- **Cross-package integration tests** under `tests/integration/` for the full user-memory pipeline and full agent-memory pipeline.
- **`Cluster.members: list[str]`** — caller-supplied entity ids; algorithm appends on merge, never inspects semantics. Caller uses this to track which business entities belong to each cluster.
- **`EpisodeExtractor` generic mode** — pass `sender_id=None` to extract one whole-memcell generic episode using `EPISODE_GENERATION_PROMPT` (cheaper than per-user fan-out). Pass a user id to use the USER-focused `USER_EPISODE_GENERATION_PROMPT`.
- **`AtomicFactExtractor` generic mode** — pass `sender_id=None` to extract facts not bound to any user.
- **`ProfileExtractor` UPDATE mode** — pass `old_profile=Profile` to use the incremental ops-based update path. When the merged item count exceeds an internal compact threshold, a second LLM pass runs automatically (caller-transparent). `old_profile=None` continues to trigger full INIT extraction.
- **`AgentSkillExtractor` expanded kwargs** — all policy thresholds exposed as per-call keyword arguments: `skip_quality_threshold`, `skip_maturity_scoring`, `maturity_threshold`, `retire_confidence`, `failure_quality_threshold`, `max_case_history`, `max_description_tokens`, `max_content_tokens`, `maturity_trivial_change_ratio`, `maturity_reeval_change_ratio`. `SkillConfig` public class removed; `_SkillCfg` is now an internal dataclass.
- **`skip_quality_threshold=0.2` short-circuit** in `AgentSkillExtractor` — cases with `quality_score < threshold` return `[]` without calling the LLM.
- **Time-format utilities down to core** — `format_message_timestamp` / `format_natural_language_time` live in `everalgo.llm.format` and are shared across all subpackages.
- **Robust LLM JSON parser down to core** — `everalgo.llm.parse.parse_llm_json_object` (fence → direct loads → outermost-braces fallback) shared across all subpackages.
- **Boundary prompt ISO 8601 UTC format** — all boundary detection prompts (EN and ZH) now use `YYYY-MM-DDTHH:MM:SSZ` for timestamps.
- **Instance-only LLM injection** — `llm=` is passed at constructor time only; the 4-layer global resolution (global default / scoped context / per-call override) has been removed.
- **`AgentBoundaryDetector`** — filter → detect → remap pipeline for mixed `ConversationItem` trajectories; tool-call items are preserved in output `MemCell.items`.
- **`BoundaryDetector` renamed** — `ChatBoundaryDetector` → `UserBoundaryDetector` → `BoundaryDetector` (final, matching the no-prefix convention used by all other facades).
- **Unified `MemCell` contract** — `MemCell.items: list[ConversationItem]` replaces the earlier split between text-only and tool-aware cell types.
- **Google-style docstrings** workspace-wide.

### Changed

- **Clustering API redesigned** — `cluster_by_geometry` / `cluster_by_llm` now accept `(new_cluster: Cluster, existing_clusters: list[Cluster]) → Cluster | None`; `ClusterState` removed. The caller owns `list[Cluster]`, stamps IDs, and serialises. `ClusterConfig` removed; per-function kwargs replace it.
- **`AgentSkillExtractor.aextract` signature change** — `cluster_id` parameter removed (caller stamps `AgentSkill.cluster_id` after extraction); `case_history` renamed to `supporting_cases`.
- **`AgentSkill.cluster_id`** defaults to `""` (caller stamps after extraction; algo emits empty).
- **Operator-side retry/fallback removed** — retries belong to the LLM SDK or to the caller; EverAlgo operators make exactly one LLM call and propagate errors.

### Removed

- **`Profile.sources` field** — removed from all profile prompts and the `Profile` type surface; the field was LLM-hallucinated and provided no reliable traceability.
- **`SkillConfig` public class** — replaced by per-call keyword arguments on `AgentSkillExtractor.aextract`.
- **`ClusterState`** — replaced by caller-owned `list[Cluster]`; see Changed above.
- **`ClusterConfig`** — replaced by per-function keyword arguments on `cluster_by_geometry` / `cluster_by_llm`.
- **`TextMessage` type** — merged into `ChatMessage` with multimodal `content: str | list[ContentBlock]`.
- **Dead `_conversation_item_adapter`** from `agent_memory/case.py`.

[everalgo-rank/0.3.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-rank/v0.3.0
[0.2.0]: https://github.com/EverMind-AI/EverAlgo/releases/tag/everalgo-core/v0.2.0
[0.1.0]: https://github.com/EverMind-AI/EverAlgo/tree/everalgo-core/v0.2.0
