# Contributing to EverAlgo

Thanks for your interest in contributing! This file is the entry point; the detailed guides live
alongside the code so they stay in sync with it.

- **Development workflow** (environment setup, tests, lint, type-checking, pre-commit) —
  [`docs/contributing.md`](docs/contributing.md).
- **Architecture, conventions, and the contract between human and AI contributors** —
  [`AGENTS.md`](AGENTS.md) (the canonical source of truth; `CLAUDE.md` / `.cursorrules` are symlinks).

## Quick reference

```bash
uv sync --all-packages --group dev   # install the workspace (editable) + dev tools
uv run pre-commit install            # register the git hook — REQUIRED per clone
uv run pytest                        # run the test suite
uv run ruff check . && uv run ruff format --check .
uv run mypy . && uv run pyright      # both type-checkers (CI gate)
```

## Ground rules

- **Branching is trunk-based.** `main` is protected; land changes via a Merge Request that
  squash-merges into `main`. Branch names: `feat/<topic>`, `fix/<bug>`, `docs/<topic>`,
  `refactor/<topic>`.
- **Commit messages: Gitmoji + Conventional Commits** — `<emoji> <type>(<scope>): <description>`.
  The MR title is load-bearing (it becomes the squash commit). See [`AGENTS.md`](AGENTS.md) §6.
- **English only** in code, comments, configuration, and commit messages.
- By contributing you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

For anything ambiguous, open a draft MR or an issue and ask — early feedback beats a large
rewrite later.
