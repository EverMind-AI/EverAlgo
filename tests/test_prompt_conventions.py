"""No prompt constant anywhere in the workspace may use the single-brace placeholder syntax.

``render_prompt`` already rejects a ``{name}`` template at render time, but only for the prompts a
test happens to render. This test walks every ``prompts`` package in the workspace instead, so a
template added or edited in any package is held to the one syntax whether or not it has a rendering
test of its own — the gate that stops the repo splitting into two placeholder dialects again.

Importing is what makes it total: a constant assembled by concatenation or ``.replace`` at import
time is scanned in its final form, which reading the source text would miss.

Lives at the repo-root ``tests/`` because it spans every distribution, like ``test_logging_conventions.py``.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

# The ``everalgo`` namespace package each algorithm package contributes a ``prompts`` subpackage to.
PROMPT_PACKAGES = (
    "everalgo.agent_memory.prompts",
    "everalgo.boundary.prompts",
    "everalgo.clustering.prompts",
    "everalgo.knowledge.prompts",
    "everalgo.parser.prompts",
    "everalgo.rank.prompts",
    "everalgo.user_memory.prompts",
)

# A prompt shorter than this is a label or a fragment, not a template; the bound keeps the sweep off
# module dunders and short constants that happen to be uppercase.
_MIN_PROMPT_LENGTH = 100

# The pre-migration placeholder shape. A literal JSON shard such as ``{"key": ...}`` does not match:
# the brace has to be followed by a bare identifier and a closing brace.
_LEGACY_PLACEHOLDER = re.compile(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})")


def _iter_prompt_modules() -> list[ModuleType]:
    """Import every module under the workspace's prompt packages."""
    modules: list[ModuleType] = []
    for name in PROMPT_PACKAGES:
        package = importlib.import_module(name)
        modules.append(package)
        modules.extend(
            importlib.import_module(info.name) for info in pkgutil.walk_packages(package.__path__, f"{name}.")
        )
    return modules


def _prompt_constants(module: ModuleType) -> list[tuple[str, str]]:
    """Return the module's ``(name, value)`` pairs that look like prompt templates."""
    return [
        (name, value)
        for name, value in vars(module).items()
        if name.isupper() and isinstance(value, str) and len(value) > _MIN_PROMPT_LENGTH
    ]


_PROMPT_CONSTANTS = [
    pytest.param(value, id=f"{module.__name__}.{name}")
    for module in _iter_prompt_modules()
    for name, value in _prompt_constants(module)
]


def test_the_sweep_found_prompts_to_check() -> None:
    """Guard the guard: a broken import list would make every other case vacuously pass."""
    assert len(_PROMPT_CONSTANTS) > 50


@pytest.mark.parametrize("template", _PROMPT_CONSTANTS)
def test_prompt_constant_has_no_single_brace_placeholder(template: str) -> None:
    found = sorted(set(_LEGACY_PLACEHOLDER.findall(template)))
    assert not found, f"single-brace placeholders {found}; write them as {{{{name}}}}"
