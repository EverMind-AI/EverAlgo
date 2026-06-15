# Contributing

Contributions are welcome — bug reports, algorithm improvements, new extractors, new LLM providers, documentation fixes, and test coverage.

---

## Before you start

Read **[AGENTS.md](../AGENTS.md)** first.
It is the single source of truth for AI assistants and human contributors alike.
Key sections:

- §3 — Quick start and the pre-commit hook pitfall
- §5 — Code style (naming convention, async contract, prompt storage)
- §6 — Branching and commit format
- §7 — Checklist for adding a new algorithm operator
- §8 — Checklist for adding a new LLM provider

---

## Development setup

```bash
git clone git@github.com:EverMind-AI/EverAlgo.git
cd everalgo

uv sync --all-packages --group dev
uv run pre-commit install
ls .git/hooks/pre-commit              # must exist
```

See [Installation](installation.md) for details.

---

## Running checks locally

Before opening a Merge Request (GitLab) or Pull Request (GitHub mirror), run the full check suite:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pyright
uv run pytest
```

All five commands must pass.
CI runs the same checks and will reject the MR / PR if any fail.

---

## Commit format

EverAlgo uses **Gitmoji + Conventional Commits**:

```
<emoji> <type>(<scope>): <description>
```

Examples:

```
✨ feat(user-memory): add ProfileExtractor prompt zh variant
🐛 fix(boundary): correct token count for emoji-only messages
♻️ refactor(rank): extract shared fusion helper
✅ test(clustering): cover cluster_by_llm fast-path edge case
📝 docs(concepts): clarify stateless-design operator contract
```

The scope is the distribution name without the `everalgo-` prefix (e.g. `core`, `boundary`, `user-memory`).
For cross-cutting changes use `ci`, `repo`, or `docs`.

The **MR / PR title** is load-bearing: the primary GitLab repo squash-merges the MR title verbatim onto `main`, and `git-cliff` parses these messages for CHANGELOGs. Keep the same format when contributing via the GitHub mirror.

---

## CHANGELOG

Every MR that adds, changes, or removes **user-visible behaviour** must include a one-line entry in the affected package's `packages/everalgo-<dist>/CHANGELOG.md` under the `## [Unreleased]` section. Use the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) subsection that fits: `### Added`, `### Changed`, `### Fixed`, `### Removed`, or `### Performance`.

Internal refactors, test-only changes, and CI tweaks do not need an entry.

At release time, the `[Unreleased]` section is promoted to the new version header. Writing the entry in the MR — when the author has full context — produces better release notes than reconstructing from git history later. See [releasing.md](releasing.md) for the full policy.

---

## Branching

Work on short-lived branches off `main`:

```
feat/<topic>
fix/<bug>
docs/<topic>
refactor/<topic>
```

Open a Merge Request (GitLab, primary) or Pull Request (GitHub mirror) and squash-merge into `main`.
`main` is protected; direct push is denied.

---

## Adding a new operator

Follow the checklist in [AGENTS.md §7](../AGENTS.md).
In summary:

1. Pick the right subpackage based on the algorithm axis.
2. Write the operator as a module-level function or a stateless class implementing the relevant `Protocol`.
3. Store prompt strings as module-level constants in `<subpkg>/prompts/en/<name>.py` (and `zh/` if applicable).
4. Re-export the operator in `<subpkg>/__init__.py` and add it to `__all__`.
5. Write tests using `FakeLLMClient` — no real network calls in default `pytest`.
6. Add a CHANGELOG entry under `## [Unreleased]` in `packages/everalgo-<dist>/CHANGELOG.md` (see [releasing.md](releasing.md)).
7. Run the full check suite before opening the MR / PR.

---

## Questions

Open an issue or discussion on the primary GitLab repo or the GitHub mirror.
