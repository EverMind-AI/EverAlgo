# EverCore

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

EverCore is the **algorithm library** behind EverMind's memory system — stateless, dependency-free of any storage, focused on extraction and ranking only. Orchestration, persistence, and routing live upstream in EverOS.

## Why split EverCore from EverOS?

- **Algorithm engineers iterate fast.** EverCore is "the algorithm team's home base" — every change to extraction strategies, prompts, fusion math, ranker weights happens here without going through service-layer ceremony.
- **Pure functions, easy to reason about.** No DB, no filesystem, no business state. All operators are plain in-memory transforms with explicit input / output types.
- **One codebase serves both the open-source and the commercial cloud builds.** The same `evercore.*` packages are consumed by both editions.

The full architecture rationale lives in [`docs/design.md`](docs/design.md).

## Repository layout

This repo is a **monorepo** of 8 publishable distributions sharing the `evercore.*` namespace via [PEP 420](https://peps.python.org/pep-0420/), managed with [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):

| Distribution | What it provides |
|---|---|
| [`evercore-core`](packages/evercore-core/) | Types, LLM client + providers, prompt validators, testing helpers |
| [`evercore-boundary`](packages/evercore-boundary/) | MemCell extractors (`Chat` / `Workspace` / `Agent`) + tokenize / split |
| [`evercore-clustering`](packages/evercore-clustering/) | `cluster_by_geometry` / `cluster_by_llm` over `ClusterState` |
| [`evercore-rank`](packages/evercore-rank/) | 4 rankers (episodic / profile / case / skill) over fusion / weight / rerank |
| [`evercore-parser`](packages/evercore-parser/) | Multimodal raw-file → `ParsedContent` |
| [`evercore-user-memory`](packages/evercore-user-memory/) | `Episode` / `Foresight` / `AtomicFact` / `Profile` extractors |
| [`evercore-agent-memory`](packages/evercore-agent-memory/) | `AgentCase` / `AgentSkill` extractors |
| [`evercore-knowledge`](packages/evercore-knowledge/) | `KnowledgeMemory` extractors |

## Quick start

```bash
git clone git@gitlab.com:npc-work/aic/ai/evercore.git
cd evercore

uv sync --all-packages          # editable-install all 8 packages into a shared venv
uv run pytest                   # run the workspace-wide test suite
```

Install only what you need on the consumer side:

```bash
pip install evercore-user-memory      # auto-pulls evercore-core + boundary + clustering
pip install evercore-rank             # auto-pulls evercore-core
pip install evercore-knowledge        # auto-pulls evercore-core + parser
```

## Releasing

Every distribution is released **independently**: each `packages/evercore-*/pyproject.toml` carries its own `version = "..."` and follows its own SemVer cadence. There is no umbrella version — bumping `evercore-rank` does not require bumping anything else. (Rationale: see [`docs/design.md`](docs/design.md) §1.3 "Why no meta package".)

### Cutting a release

1. Bump the `version = "..."` field in the relevant `packages/evercore-<name>/pyproject.toml` and land the change on `main` via MR.
2. From `main`, push a per-package tag:
   ```bash
   git tag evercore-clustering/v0.2.0
   git push origin evercore-clustering/v0.2.0
   ```
3. The `.gitlab-ci.yml` `build` stage runs `uv build` for the matching package; the publish stage uploads the wheel + sdist to PyPI using `UV_PUBLISH_TOKEN` from CI variables.

### Tag naming

Format: `<dist-name>/v<semver>`. The slash separator keeps per-distribution tags unambiguous from any future repository-wide tag.

```text
evercore-core/v0.1.0
evercore-rank/v0.2.0
evercore-user-memory/v0.1.3
```

### Local dry-run

```bash
cd packages/evercore-clustering
uv build                                 # writes packages/<dist>/dist/*.whl + *.tar.gz
uv publish --dry-run                     # validate without uploading
uv publish --token "$PYPI_TOKEN"         # actual upload (admins only)
```

### Pre-flight checklist

Before pushing a release tag:

- [ ] `uv run pytest` is green on `main`.
- [ ] `uv run ruff check . && uv run ruff format --check .` pass.
- [ ] The bumped `version` honours SemVer relative to the previous tag (no breaking change in a minor / patch).
- [ ] Downstream packages' `>=X.Y,<2.0` ranges still allow the new version.

### CI / pipeline status

The current `.gitlab-ci.yml` `build` job builds **every** package on any tag push. Before the first real release, the rule needs to be tightened to filter on the tag's `<dist-name>` prefix so only the matching package is built and published. Tracked as a follow-up.

## For AI coding assistants

Read [`AGENTS.md`](AGENTS.md) — the single source of truth for assistant context. `CLAUDE.md` and `.cursorrules` are symlinks to it.

## Documentation

- [`docs/design.md`](docs/design.md) — full architecture (read this before challenging any design choice)
- [`docs/decisions/`](docs/decisions/) — ADRs (Architecture Decision Records)
- [`AGENTS.md`](AGENTS.md) — how to onboard, how to add an operator, how to add an LLM provider

## License

[MIT](LICENSE).
