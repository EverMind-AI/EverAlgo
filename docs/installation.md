# Installation

## Prerequisites

- **Python 3.12 or later.**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or `pip` / `pipx`.

---

## Install a single distribution (most users)

Install only the distribution that covers your use case.
Each distribution automatically pulls its required dependencies.

```bash
# User-memory pipeline: boundary detection + 4 extractors
pip install everalgo-user-memory

# Agent-memory pipeline: case and skill extractors
pip install everalgo-agent-memory

# Retrieval re-ranking only
pip install everalgo-rank

# Multimodal file ingestion to KnowledgeMemory
pip install everalgo-knowledge

# Raw parsing only (OCR, ASR, document layout)
pip install everalgo-parser

# Low-level building blocks (types, LLM client, testing helpers)
pip install everalgo-core
```

With uv:

```bash
uv add everalgo-user-memory
```

---

## Development install (contributors and algorithm engineers)

Clone the monorepo and install all 8 packages as editable into a shared venv:

```bash
git clone git@github.com:EverMind-AI/EverAlgo.git
cd everalgo

uv sync --all-packages --group dev   # shared venv, all 8 packages editable
uv run pre-commit install             # register the commit-time lint hook
ls .git/hooks/pre-commit              # MUST exist — see AGENTS.md §3 for the common pitfall
```

Verify the install:

```bash
uv run python -c "from everalgo.user_memory import EpisodeExtractor; print('ok')"
uv run pytest
```

If you are working on a single package and want a faster sync:

```bash
uv sync --package everalgo-clustering
```

---

## LLM provider dependencies

`openai` is a mandatory dependency of `everalgo-core` — it is installed automatically when you install any EverAlgo distribution. Only the OpenAI-compatible provider (`OpenAICompatClient`) is currently implemented; it works with any OpenAI-API-compatible endpoint (OpenAI, OpenRouter, vLLM, DeepSeek, Azure).

`everalgo.testing.fake_llm.FakeLLMClient` is always available without any real API key — use it for local development and tests.

---

## Troubleshooting

**`ImportError: cannot import name 'EpisodeExtractor'`**
Verify that `everalgo-user-memory` is installed, not just `everalgo-core`.
Each distribution is independent: `pip show everalgo-user-memory` should list it.

**Pre-commit hook not firing after clone**
`uv run pre-commit run --all-files` passing does not mean the hook is installed.
Run `uv run pre-commit install` and confirm with `ls .git/hooks/pre-commit`.
If the file is missing, every `git commit` silently bypasses lint — see AGENTS.md §3.

**`uv sync` fails with `conflict`**
Try `uv sync --reinstall` or delete `.venv/` and re-run `uv sync --all-packages`.
