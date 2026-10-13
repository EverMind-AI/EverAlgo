"""Checks that a caller's bad input surfaces as ``InvalidInputError``, never as a bare ``ValueError``.

A host decides what to do with an error by its type: ``LLMOutputError`` means the model's reply was bad (retry may
help), ``InvalidInputError`` means the caller's data was bad (the same input fails the same way, so it is a terminal
outcome). A bare ``ValueError`` from an operator could be either, or a bug, so the host cannot tell. Source files
are read with ``ast``, so the check needs no package to be importable, and it spans every distribution like
``test_core_layering.py`` does.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORE_SRC = "packages/everalgo-core/src/everalgo"

#: Files that may still raise a bare ``ValueError``, each with why it is not a caller's-input error.
_BARE_VALUE_ERROR_ALLOWED: dict[str, str] = {
    # The schema a host hands over is checked against the strict subset an LLM API accepts: a programming error in
    # the host's reply model, found when the model class is first used, not data a caller passes per call.
    f"{_CORE_SRC}/llm/schema.py": "reply-model schema outside the strict subset (a coding error)",
    # Retry counts and response-format settings are client construction arguments, not operator input.
    f"{_CORE_SRC}/llm/openai_compat.py": "client construction arguments",
    # A prompt template with a bad placeholder is its author's mistake, found when the prompt is rendered.
    f"{_CORE_SRC}/utils/templates.py": "prompt template placeholder errors (the prompt author's coding error)",
}
#: Whole directories: test doubles and their misuse are not operator input.
_BARE_VALUE_ERROR_ALLOWED_DIRS: dict[str, str] = {
    f"{_CORE_SRC}/testing/": "test doubles; misusing one is a test-writing error",
}


def _bare_value_error_raises(path: Path) -> list[int]:
    """Line of each ``raise ValueError`` / ``raise ValueError(...)`` in ``path``."""
    lines = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        raised = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if isinstance(raised, ast.Name) and raised.id == "ValueError":
            lines.append(node.lineno)
    return lines


def _is_allowed(relative: str) -> bool:
    return relative in _BARE_VALUE_ERROR_ALLOWED or relative.startswith(tuple(_BARE_VALUE_ERROR_ALLOWED_DIRS))


def test_no_distribution_raises_a_bare_value_error_outside_the_allowed_files() -> None:
    offenders = []
    for path in sorted(_REPO_ROOT.glob("packages/*/src/**/*.py")):
        relative = path.relative_to(_REPO_ROOT).as_posix()
        if not _is_allowed(relative):
            offenders += [f"{relative}:{line}: raise ValueError" for line in _bare_value_error_raises(path)]
    assert offenders == [], f"raise InvalidInputError (everalgo.utils) for the caller's bad input: {offenders}"


def test_every_allowed_file_still_raises_a_bare_value_error() -> None:
    """A stale entry would silently widen the allowance, so each one must still be needed."""
    stale = [relative for relative in _BARE_VALUE_ERROR_ALLOWED if not _bare_value_error_raises(_REPO_ROOT / relative)]
    stale += [
        directory
        for directory in _BARE_VALUE_ERROR_ALLOWED_DIRS
        if not any(_bare_value_error_raises(path) for path in (_REPO_ROOT / directory).glob("**/*.py"))
    ]
    assert stale == [], stale


def test_the_scan_finds_a_bare_raise_in_either_spelling_and_ignores_other_errors(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "def f(x):\n"
        "    if x == 1:\n"
        "        raise ValueError('called')\n"
        "    if x == 2:\n"
        "        raise ValueError\n"
        "    if x == 3:\n"
        "        raise TypeError('other')\n"
        "    try:\n"
        "        pass\n"
        "    except ValueError:\n"
        "        raise\n",
        encoding="utf-8",
    )
    assert _bare_value_error_raises(source) == [3, 5]
