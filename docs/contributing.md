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

Before opening a Merge Request, run the full check suite:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pyright
uv run pytest
```

All five commands must pass.
CI runs the same checks and will reject the MR if any fail.

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

The **MR title** is load-bearing: GitLab squash-merges the MR title verbatim onto `main`, and `git-cliff` parses these messages for CHANGELOGs.

---

## Branching

Work on short-lived branches off `main`:

```
feat/<topic>
fix/<bug>
docs/<topic>
refactor/<topic>
```

Open a Merge Request and squash-merge into `main`.
`main` is GitLab-protected; direct push is denied.

---

## Adding a new operator

Follow the checklist in [AGENTS.md §7](../AGENTS.md).
In summary:

1. Pick the right subpackage based on the algorithm axis.
2. Write the operator as a module-level function or a stateless class implementing the relevant `Protocol`.
3. Store prompt strings as module-level constants in `<subpkg>/prompts/en/<name>.py` (and `zh/` if applicable).
4. Re-export the operator in `<subpkg>/__init__.py` and add it to `__all__`.
5. Write tests using `FakeLLMClient` — no real network calls in default `pytest`.
6. Run the full check suite before opening the MR.

---

## Questions

Open a GitLab issue or discussion.
For design decisions, review the ADRs in `local/decisions/` — each one covers a concrete architectural choice with alternatives considered.
