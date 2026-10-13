"""Checks that the ``everalgo.boundary`` modules import one another in one direction only.

``_detector`` sits over ``_segments``, ``_render``, ``_replies`` and the prompts; ``_segments`` over ``_render``;
``_render`` and ``_replies`` over no sibling; ``__init__`` re-exports from ``_detector`` alone.

Imports are read from the source with ``ast``, so a check needs nothing importable beyond the package's location.
"""

from __future__ import annotations

import ast
from pathlib import Path

import everalgo.boundary as boundary_package

_PACKAGE = "everalgo.boundary"
_PACKAGE_DIR = Path(boundary_package.__file__).parent

#: The sibling modules each top-level module may import.
_ALLOWED_SIBLINGS: dict[str, frozenset[str]] = {
    "__init__": frozenset({"_detector"}),
    "_detector": frozenset({"_segments", "_render", "_replies", "prompts.en.chat"}),
    "_segments": frozenset({"_render"}),
    "_render": frozenset(),
    "_replies": frozenset(),
    "workspace": frozenset(),
}


def _tree(module: str) -> ast.Module:
    return ast.parse((_PACKAGE_DIR / f"{module}.py").read_text(encoding="utf-8"))


def _sibling_imports(module: str) -> set[str]:
    siblings = set()
    for node in ast.walk(_tree(module)):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        siblings |= {name.removeprefix(f"{_PACKAGE}.") for name in names if name.startswith(f"{_PACKAGE}.")}
    return siblings


def test_the_rule_covers_every_top_level_module() -> None:
    assert {path.stem for path in _PACKAGE_DIR.glob("*.py")} == set(_ALLOWED_SIBLINGS)


def test_modules_import_their_siblings_in_one_direction_only() -> None:
    offenders = {
        module: sorted(_sibling_imports(module) - allowed)
        for module, allowed in _ALLOWED_SIBLINGS.items()
        if _sibling_imports(module) - allowed
    }
    assert offenders == {}


def test_no_module_imports_its_siblings_relatively() -> None:
    relative = [
        module
        for module in _ALLOWED_SIBLINGS
        for node in ast.walk(_tree(module))
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert relative == []
