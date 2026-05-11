# `render_prompt` helper — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the `(prompt or DEFAULT).format(**fields)` pattern from the 2 existing operators into a shared `render_prompt` helper in `everalgo-core/src/everalgo/prompts/render.py`. Implements `docs/superpowers/specs/2026-05-11-render-prompt-helper-design.md`.

**Architecture:** Single module-level function `render_prompt(default: str, prompt: str | None, /, **fields: Any) -> str`. Lives next to existing `prompts/validator.py`. Public via `everalgo.prompts.render_prompt`.

**Tech Stack:** Plain Python (stdlib only); `pytest` for tests.

---

## File Structure

**Create:**
- `packages/everalgo-core/src/everalgo/prompts/render.py` — the helper
- `packages/everalgo-core/tests/prompts/__init__.py` — empty (if not already present)
- `packages/everalgo-core/tests/prompts/test_render.py` — unit tests

**Modify:**
- `packages/everalgo-core/src/everalgo/prompts/__init__.py` — re-export `render_prompt`
- `packages/everalgo-boundary/src/everalgo/boundary/chat.py` (lines 55-58) — call site rewrite
- `packages/everalgo-user-memory/src/everalgo/user_memory/episode.py` (lines 48-51) — call site rewrite

---

## Task 1: Build `render_prompt` + tests + 2 call-site rewrites

**Files** (as listed in "File Structure" above).

- [ ] **Step 1: Create `packages/everalgo-core/src/everalgo/prompts/render.py`**

```python
"""Render a prompt template with caller-provided fields, falling back to a default.

Centralises the ``(prompt or DEFAULT).format(**fields)`` pattern used by
every LLM-calling operator in EverAlgo, so that future cross-cutting
concerns (prompt logging, escape rules, i18n switching, instrumentation)
have a single edit point.
"""

from __future__ import annotations

from typing import Any


def render_prompt(default: str, prompt: str | None, /, **fields: Any) -> str:
    """Render ``prompt`` with ``fields``; if ``prompt`` is None, use ``default``.

    Designed for operators that accept an optional caller-override of the
    prompt template while still shipping a sensible default. ``default``
    and ``prompt`` are positional-only so callers cannot accidentally
    swap their order via keyword arguments.

    Args:
        default: Module-level prompt constant shipped with the operator
            (for example ``CHAT_BOUNDARY_DETECT_PROMPT_EN``).
        prompt: Caller override; if ``None``, falls back to ``default``.
        **fields: Keyword arguments substituted into the template via
            :py:meth:`str.format`. The template is responsible for naming
            each placeholder; missing placeholders raise :class:`KeyError`.

    Returns:
        The rendered prompt string, ready to send to the LLM.

    Raises:
        KeyError: If the template references a placeholder not in ``fields``.

    Example:
        >>> render_prompt("Hello, {name}!", None, name="world")
        'Hello, world!'
        >>> render_prompt("Hello, {name}!", "Hi {name}", name="world")
        'Hi world'
    """
    return (prompt or default).format(**fields)
```

- [ ] **Step 2: Update `packages/everalgo-core/src/everalgo/prompts/__init__.py`**

Replace the file's contents with:

```python
"""Prompt utilities shared across the library.

Per ``docs/design.md`` §1.4: concrete prompt strings live next to their
algorithm in each subpackage at ``<subpkg>/prompts/{en,zh}/<name>.py`` as
module-level constants — not in this directory. This package only ships
the cross-cutting helpers (rendering, validation) every operator reuses.
"""

from everalgo.prompts.render import render_prompt

__all__ = ["render_prompt"]
```

- [ ] **Step 3: Create `packages/everalgo-core/tests/prompts/__init__.py`** if it does not already exist

```bash
test -f packages/everalgo-core/tests/prompts/__init__.py || touch packages/everalgo-core/tests/prompts/__init__.py
```

- [ ] **Step 4: Create `packages/everalgo-core/tests/prompts/test_render.py`**

```python
"""Tests for ``everalgo.prompts.render``."""

from __future__ import annotations

import pytest

from everalgo.prompts import render_prompt


def test_uses_default_when_prompt_is_none() -> None:
    result = render_prompt("Hello, {name}!", None, name="world")
    assert result == "Hello, world!"


def test_uses_override_when_prompt_is_provided() -> None:
    result = render_prompt("Hello, {name}!", "Hi {name}", name="world")
    assert result == "Hi world"


def test_empty_string_prompt_falls_back_to_default() -> None:
    """An empty string is falsy in Python, so it should fall back to ``default``.

    This matches ``(prompt or default)`` semantics and matters because some
    callers may pass ``prompt=""`` to mean "no override"; we preserve that
    behaviour rather than rendering an empty template silently.
    """
    result = render_prompt("Hello, {name}!", "", name="world")
    assert result == "Hello, world!"


def test_missing_placeholder_in_fields_raises_key_error() -> None:
    with pytest.raises(KeyError):
        render_prompt("Hello, {name}!", None)


def test_extra_fields_are_silently_ignored() -> None:
    """``str.format`` ignores kwargs that the template does not reference."""
    result = render_prompt("Hello, {name}!", None, name="world", unused="extra")
    assert result == "Hello, world!"


def test_default_and_prompt_are_positional_only() -> None:
    """The ``/`` separator forbids passing ``default=`` or ``prompt=`` as kwargs."""
    with pytest.raises(TypeError):
        render_prompt(default="Hello, {name}!", prompt=None, name="world")  # type: ignore[misc, call-arg]
```

- [ ] **Step 5: Run the new tests**

```bash
cd /Users/admin/Documents/evermemos/evercore
uv run pytest packages/everalgo-core/tests/prompts/test_render.py -v
```

Expected: 6 passed.

If the `positional-only` assertion fails, double-check that the `render_prompt` signature in Step 1 has `/` between `prompt` and `**fields`.

- [ ] **Step 6: Replace the call in `packages/everalgo-boundary/src/everalgo/boundary/chat.py`**

Read the file first. Then change lines 55-58 from:

```python
        rendered = (prompt or CHAT_BOUNDARY_DETECT_PROMPT_EN).format(
            messages=_format_messages_for_prompt(messages),
            token_count=count_tokens(_concat_messages(messages)),
        )
```

to:

```python
        rendered = render_prompt(
            CHAT_BOUNDARY_DETECT_PROMPT_EN,
            prompt,
            messages=_format_messages_for_prompt(messages),
            token_count=count_tokens(_concat_messages(messages)),
        )
```

Also add an import alongside the existing `from everalgo.boundary.prompts.en.chat import CHAT_BOUNDARY_DETECT_PROMPT_EN`:

```python
from everalgo.prompts import render_prompt
```

(Place the new import in the correct isort group — `everalgo.prompts` is now first-party.)

- [ ] **Step 7: Replace the call in `packages/everalgo-user-memory/src/everalgo/user_memory/episode.py`**

Read the file first. Then change lines 48-51 from:

```python
        rendered = (prompt or EPISODE_EXTRACT_PROMPT_EN).format(
            memcell_text=_render_memcell_text(memcell),
            timestamp=memcell.timestamp,
        )
```

to:

```python
        rendered = render_prompt(
            EPISODE_EXTRACT_PROMPT_EN,
            prompt,
            memcell_text=_render_memcell_text(memcell),
            timestamp=memcell.timestamp,
        )
```

Add the corresponding import:

```python
from everalgo.prompts import render_prompt
```

- [ ] **Step 8: Run the full test suite**

```bash
uv run pytest
```

Expected: 173 passed (167 pre-existing + 6 new). If any pre-existing test fails, the most likely cause is the call-site rewrite swapping `default` and `prompt` — re-read Steps 6 and 7.

- [ ] **Step 9: Lint + format + type-check**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

Expected: all three green. If `ruff check` reports `I001` import sort issues at `chat.py` or `episode.py`, run `uv run ruff check --fix .` once and re-verify.

- [ ] **Step 10: Commit**

```bash
git add -A
git status   # confirm staged set: 4 modified + 3 new
git commit -m "$(cat <<'EOF'
♻️ refactor(core): extract render_prompt helper for the (prompt or DEFAULT).format pattern

Adds `everalgo.prompts.render_prompt(default, prompt, /, **fields)` that
centralises the `(prompt or DEFAULT).format(**fields)` pattern used by
every LLM-calling operator. Replaces the two existing call sites
(boundary/chat.py and user_memory/episode.py); the 15 stub operators
will adopt the helper as they land.

Why now: 17 operator signatures already accept `prompt: str | None`, so
the pattern is locked in. Centralising lets us add prompt logging /
escape / i18n / validation at one site later. Function signature is
positional-only on `default` and `prompt` so callers cannot accidentally
swap their order via keyword args.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git log --oneline -3
```

If the pre-commit hook fails or auto-fixes files, fix the cause (never `--no-verify`); re-stage and re-commit.

### Self-review before reporting DONE

```bash
git show HEAD --stat                                          # 4 modified + 3 new files
grep -n 'prompt or .*PROMPT' packages/everalgo-boundary/src/everalgo/boundary/chat.py    # empty
grep -n 'prompt or .*PROMPT' packages/everalgo-user-memory/src/everalgo/user_memory/episode.py  # empty
grep -n 'render_prompt' packages/everalgo-boundary/src/everalgo/boundary/chat.py packages/everalgo-user-memory/src/everalgo/user_memory/episode.py
uv run pytest -q | tail -3
```

## Task 2: Push + open MR

- [ ] **Step 1: Push**

```bash
git push -u origin feat/extract-render-prompt
```

- [ ] **Step 2: Open MR**

```bash
glab mr create \
  --title "♻️ refactor(core): extract render_prompt helper for prompt overrides" \
  --target-branch main \
  --remove-source-branch \
  --description "$(cat <<'EOF'
## Summary

Extract the `(prompt or DEFAULT).format(**fields)` pattern from the 2 existing operators into a shared helper `everalgo.prompts.render_prompt(default, prompt, /, **fields)`. The 15 stub operators (boundary/user-memory/agent-memory/knowledge) will adopt the helper as they land.

- New: `everalgo-core/src/everalgo/prompts/render.py` (~30 lines)
- New: `everalgo-core/tests/prompts/test_render.py` (6 tests)
- Updated: `everalgo-core/src/everalgo/prompts/__init__.py` re-exports `render_prompt`
- Updated: `boundary/chat.py:55-58` and `user_memory/episode.py:48-51` adopt the helper

Spec: `docs/superpowers/specs/2026-05-11-render-prompt-helper-design.md`
Plan: `docs/superpowers/plans/2026-05-11-render-prompt-helper.md`

## Why

17 operator signatures already accept `prompt: str | None = None`. Centralising the render path means future prompt logging / escape / i18n / validation hooks change one site, not 17. Function signature is positional-only on `default` and `prompt` to prevent accidental swap via keyword args.

## Verification

- `uv run pytest` — 173 passed (167 prior + 6 new)
- `uv run ruff check .` / `ruff format --check .` / `mypy .` — all green

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Confirm MR URL and pipeline running**

```bash
glab mr view
glab ci status --branch feat/extract-render-prompt
```
