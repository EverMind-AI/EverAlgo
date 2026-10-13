"""Checks that the ``everalgo.user_memory.episode`` package keeps the layout its docstring describes.

- Its modules import one another in one direction only: ``_extractor`` and ``_summarizer`` each over ``_render``,
  ``_replies`` and ``_summary``, which import no sibling.
- Every logger in the package is the package's own, the name hosts collect the log contract under.

Imports are read from the source with ``ast``, so a check needs nothing importable beyond the package's location.
"""

from __future__ import annotations

import ast
from pathlib import Path

import everalgo.user_memory.episode as episode_package

_PACKAGE = "everalgo.user_memory.episode"
_PACKAGE_DIR = Path(episode_package.__file__).parent

#: The sibling modules each module may import; ``__init__`` re-exports from the two operator modules only.
_ALLOWED_SIBLINGS: dict[str, frozenset[str]] = {
    "__init__": frozenset({"_extractor", "_summarizer"}),
    "_extractor": frozenset({"_render", "_replies", "_summary"}),
    "_summarizer": frozenset({"_render", "_replies", "_summary"}),
    "_render": frozenset(),
    "_replies": frozenset(),
    "_summary": frozenset(),
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


def test_the_rule_covers_every_module_in_the_package() -> None:
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


def test_every_logger_in_the_package_is_the_package_logger() -> None:
    names = [
        (module, ast.unparse(call.args[0]) if call.args else "<none>")
        for module in _ALLOWED_SIBLINGS
        for call in ast.walk(_tree(module))
        if isinstance(call, ast.Call) and ast.unparse(call.func) == "logging.getLogger"
    ]
    assert names, "the package logs nothing"
    assert {name for _, name in names} == {repr(_PACKAGE)}, names
