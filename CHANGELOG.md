# Changelog

All notable changes to the EverAlgo monorepo are documented here. Format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). Each distribution follows its
own [Semantic Versioning 2.0](https://semver.org/spec/v2.0.0.html) cadence — there is no umbrella
version.

Per-distribution changelogs are the source of truth. This file is a navigation index.

## Distribution status

All distributions have been published to PyPI. The table below tracks the latest released version
declared in each `packages/everalgo-*/pyproject.toml`.

| Distribution | Version | Changelog |
|---|---|---|
| `everalgo-core` | 0.1.0 | [packages/everalgo-core/CHANGELOG.md](packages/everalgo-core/CHANGELOG.md) |
| `everalgo-boundary` | 0.1.0 | [packages/everalgo-boundary/CHANGELOG.md](packages/everalgo-boundary/CHANGELOG.md) |
| `everalgo-clustering` | 0.1.0 | [packages/everalgo-clustering/CHANGELOG.md](packages/everalgo-clustering/CHANGELOG.md) |
| `everalgo-rank` | 0.1.1 | [packages/everalgo-rank/CHANGELOG.md](packages/everalgo-rank/CHANGELOG.md) |
| `everalgo-user-memory` | 0.1.0 | [packages/everalgo-user-memory/CHANGELOG.md](packages/everalgo-user-memory/CHANGELOG.md) |
| `everalgo-agent-memory` | 0.1.0 | [packages/everalgo-agent-memory/CHANGELOG.md](packages/everalgo-agent-memory/CHANGELOG.md) |
| `everalgo-parser` | 0.1.0 | [packages/everalgo-parser/CHANGELOG.md](packages/everalgo-parser/CHANGELOG.md) |
| `everalgo-knowledge` | 0.1.0 | [packages/everalgo-knowledge/CHANGELOG.md](packages/everalgo-knowledge/CHANGELOG.md) |

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

[0.1.0]: https://github.com/EverMind-AI/EverAlgo/compare/main...HEAD
