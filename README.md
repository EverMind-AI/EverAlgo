# EverAlgo

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![PyPI - everalgo-core](https://img.shields.io/pypi/v/everalgo-core)](https://pypi.org/project/everalgo-core/)

EverAlgo is the **algorithm library** behind EverMind's memory system — stateless, storage-free, focused on extraction and ranking only. Orchestration, persistence, and routing live upstream in EverOS.

## Why split EverAlgo from EverOS?

- **Algorithm engineers iterate fast.** EverAlgo is the algorithm team's home base — every change to extraction strategies, prompts, fusion math, and ranker weights happens here without going through service-layer ceremony.
- **Pure functions, easy to reason about.** No DB, no filesystem, no business state. All operators are plain in-memory transforms with explicit input / output types.
- **One codebase serves both the open-source and the commercial cloud builds.** The same `everalgo.*` packages are consumed by both editions.

The full architecture lives in [`docs/concepts/architecture.md`](docs/concepts/architecture.md).

## Repository layout

This repo is a **monorepo** of 8 distributions sharing the `everalgo.*` namespace via [PEP 420](https://peps.python.org/pep-0420/), managed with [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) (seven published to PyPI; `everalgo-knowledge` is namespace-reserved and not yet published — see [Status & known limitations](#status--known-limitations)):

| Distribution | What it provides |
|---|---|
| [`everalgo-core`](packages/everalgo-core/) | Types, LLM client + providers, prompt helpers, testing utilities |
| [`everalgo-boundary`](packages/everalgo-boundary/) | `detect_boundaries` + `DetectionResult` + token helpers |
| [`everalgo-clustering`](packages/everalgo-clustering/) | `Cluster` value object + `cluster_by_geometry` / `cluster_by_llm` operators |
| [`everalgo-rank`](packages/everalgo-rank/) | 4 rankers (episodic / profile / case / skill) over fusion / weight / rerank toolkit |
| [`everalgo-parser`](packages/everalgo-parser/) | Multimodal raw-file → `ParsedContent` (image, audio, PDF, HTML, email, office, URL; video deferred) |
| [`everalgo-user-memory`](packages/everalgo-user-memory/) | `BoundaryDetector` + `Episode` / `Foresight` / `AtomicFact` / `Profile` extractors |
| [`everalgo-agent-memory`](packages/everalgo-agent-memory/) | `AgentBoundaryDetector` + `AgentCase` / `AgentSkill` extractors |
| [`everalgo-knowledge`](packages/everalgo-knowledge/) | `KnowledgeMemory` extractor (NOT YET IMPLEMENTED — not published) |

## 60-second quickstart

Install all packages into a shared editable venv, then run the offline examples — no API key required:

```bash
git clone git@github.com:EverMind-AI/EverAlgo.git
cd everalgo

uv sync --all-packages          # editable-install all 8 packages into a shared venv
uv run pre-commit install       # register the git hook (ruff + standard sanitisers)
ls .git/hooks/pre-commit        # MUST exist — see AGENTS.md §3 "Common pitfall"

uv run python examples/01_boundary_chat.py          # Chat → MemCell
uv run python examples/03_user_memory_episode.py    # MemCell → Episode
uv run python examples/04_agent_memory_case.py      # Agent trajectory → AgentCase
uv run python examples/06_full_user_memory_pipeline.py   # Full pipeline
uv run pytest                                        # workspace-wide test suite
```

> [!NOTE]
> If `.git/hooks/pre-commit` does not exist after the `install` step, hooks won't fire at `git commit` time. Verify the file every clone. Details in [`AGENTS.md`](AGENTS.md) §3.

## Code example — full user-memory pipeline

```python
import asyncio
import json
import numpy as np

from everalgo.clustering import Cluster, cluster_by_geometry
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage
from everalgo.user_memory import (
    BoundaryDetector, EpisodeExtractor, ForesightExtractor,
    AtomicFactExtractor, ProfileExtractor,
)

_BOUNDARY_JSON = json.dumps({"reasoning": "single topic", "boundaries": [], "should_wait": False})
_EPISODE_JSON  = json.dumps({"title": "Alice asks about async", "content": "Alice explored async patterns."})
_FORE_JSON     = json.dumps([{"content": "Alice will read the follow-up doc", "evidence": "assistant promised a doc", "start_time": "2023-11-14", "end_time": "2023-11-21", "duration_days": 7}])
_FACT_JSON     = json.dumps({"atomic_facts": {"time": "Nov 14 2023", "atomic_fact": ["Alice is learning Python async."]}})
_PROFILE_JSON  = json.dumps({"explicit_info": [], "implicit_traits": [{"category": "Technical", "description": "Python developer."}]})


async def main() -> None:
    fake = FakeLLMClient(responses=[
        ChatResponse(content=_BOUNDARY_JSON, model="fake"),
        ChatResponse(content=_EPISODE_JSON,  model="fake"),
        ChatResponse(content=_FORE_JSON,     model="fake"),
        ChatResponse(content=_FACT_JSON,     model="fake"),
        ChatResponse(content=_PROFILE_JSON,  model="fake"),
    ])

    messages = [
        ChatMessage(id="m1", role="user",      content="I want to learn Python async retry patterns.", timestamp=1_700_000_000_000, sender_id="u_alice", sender_name="Alice"),
        ChatMessage(id="m2", role="assistant",  content="Sure — I'll send a follow-up doc next week.", timestamp=1_700_000_001_000, sender_id="assistant"),
    ]

    # Step 1 — boundary detection
    result = await BoundaryDetector(llm=fake).adetect(messages, is_final=True)
    mc = result.cells[0]

    # Steps 2–4 — extract user-memory products (sender_id is required, not inferred)
    episode    = await EpisodeExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    foresights = await ForesightExtractor(llm=fake).aextract(mc, sender_id="u_alice")
    facts      = await AtomicFactExtractor(llm=fake).aextract(mc, sender_id="u_alice")

    # Step 5 — cluster the cell (caller wraps as size-1 Cluster, no LLM), then extract profile
    vector = np.random.rand(2560).astype(np.float32)
    existing: list[Cluster] = []
    new_cluster = Cluster(centroid=vector, last_ts=mc.timestamp)
    merged = cluster_by_geometry(new_cluster, existing)
    if merged is None:
        existing.append(new_cluster.model_copy(update={"id": "cid_001"}))

    profile = await ProfileExtractor(llm=fake).aextract([mc], sender_id="u_alice")

    print(f"episode: {episode.subject!r}")
    print(f"foresights: {len(foresights)}, facts: {len(facts)}")
    print(f"profile summary: {profile.summary!r}")


asyncio.run(main())
```

## Install — consumers

Install only the distributions you need; transitive deps are pulled automatically:

```bash
pip install everalgo-user-memory    # pulls core + boundary
pip install everalgo-agent-memory   # pulls core + boundary + clustering
pip install everalgo-rank           # pulls core
pip install everalgo-clustering     # pulls core
```

## Status & known limitations

Seven of the eight distributions are published on PyPI (most at `0.2.0`, `everalgo-rank` at `0.3.0`); `everalgo-knowledge` is intentionally **not published** (namespace reserved only). Three operators are **unimplemented placeholders that raise `NotImplementedError`** — they are deliberately kept out of the public `__all__` (import the submodule directly if you want the reserved stub) and must not be relied on yet:

| Placeholder | Reachable at | Status |
|---|---|---|
| `WorkspaceMemCellExtractor` | `everalgo.boundary.workspace` | Jira / Email / Confluence slicing — not implemented |
| `KnowledgeExtractor` | `everalgo.knowledge` | whole `everalgo-knowledge` distribution unpublished |
| video parsing | `everalgo.parser.video` | deferred pending an ADR (Gemini Video vs Whisper + frame sampling) |

Everything else is fully implemented and tested: boundary detection, both clustering operators, all four rankers, the user-memory extractors (Episode / Foresight / AtomicFact / Profile), and the agent-memory extractors (Case / Skill).

## Releasing

Every distribution is released **independently**: each `packages/everalgo-*/pyproject.toml` carries its own `version = "..."` and follows its own SemVer cadence. There is no umbrella version — bumping `everalgo-rank` does not require bumping anything else.

The repo uses a **two-tier CHANGELOG**:
- [`CHANGELOG.md`](CHANGELOG.md) at the root — current-version overview table.
- `packages/everalgo-<dist>/CHANGELOG.md` per distribution — full history; ships inside the wheel.

Per-distribution changelogs are generated by [git-cliff](https://git-cliff.org/) from Conventional Commit messages on `main` (see [`cliff.toml`](cliff.toml) for the parser config; see [`AGENTS.md`](AGENTS.md) §6 for the commit-message contract).

### Cutting a release

> **The full, verified step-by-step procedure is in [`docs/releasing.md`](docs/releasing.md)** — including the canary-first / one-tag-at-a-time lessons learned in the 0.2.0 release. The steps below are a quick reference.

Steps for releasing `everalgo-clustering v0.2.0`:

1. **Bump the version.** Open an MR editing `packages/everalgo-clustering/pyproject.toml`; merge to `main` via squash.

2. **Generate the changelog fragment.**
   ```bash
   git checkout main && git pull
   uv run git-cliff \
     --tag everalgo-clustering/v0.2.0 \
     --include-path 'packages/everalgo-clustering/**' \
     --unreleased \
     --prepend packages/everalgo-clustering/CHANGELOG.md
   ```

3. **Review and polish.** git-cliff gives a draft, not a final — edit for clarity and add Breaking-Changes call-outs.

4. **Update the root CHANGELOG** table row for this distribution.

5. **Commit + tag + push.**
   ```bash
   git add packages/everalgo-clustering/CHANGELOG.md CHANGELOG.md
   git commit -m "📝 docs(clustering): release notes for v0.2.0"
   git push origin main
   git tag everalgo-clustering/v0.2.0
   git push origin everalgo-clustering/v0.2.0
   ```

6. **Publish to PyPI.** The `.gitlab-ci.yml` `publish` stage triggers automatically on `<dist>/v<semver>` tag push using [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/) (GitLab OIDC, no long-lived tokens). See the CI file for prerequisites and mechanism details.

Tag format: `<dist-name>/v<semver>` — e.g. `everalgo-core/v0.1.0`, `everalgo-rank/v0.2.0`.

### Pre-flight checklist

- [ ] `uv run pytest` is green on `main`.
- [ ] `uv run ruff check . && uv run ruff format --check .` pass.
- [ ] `uv run mypy . && uv run pyright` pass.
- [ ] The bumped `version` honours SemVer relative to the previous tag.
- [ ] The `packages/everalgo-<dist>/CHANGELOG.md` section has been reviewed and edited.
- [ ] The root `CHANGELOG.md` table row has been updated.

## For AI coding assistants

Read [`AGENTS.md`](AGENTS.md) — the single source of truth for assistant context. `CLAUDE.md` and `.cursorrules` are symlinks to it.

## Documentation

- [`docs/index.md`](docs/index.md) — project landing page
- [`docs/installation.md`](docs/installation.md) — install + prerequisites
- [`docs/getting-started.md`](docs/getting-started.md) — 5-minute quickstart
- [`docs/concepts/`](docs/concepts/) — architecture, stateless-design, async/sync bridge
- [`docs/api/`](docs/api/) — per-package API reference
- [`docs/version-policy.md`](docs/version-policy.md) — SemVer + supported Python versions
- [`docs/contributing.md`](docs/contributing.md) — how to contribute
- [`examples/`](examples/) — runnable quickstart scripts (01 through 07)
- [`AGENTS.md`](AGENTS.md) — onboarding for AI assistants + contributors

## License

[Apache License 2.0](LICENSE).
