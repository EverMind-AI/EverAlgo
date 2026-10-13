"""Checks that everalgo-core keeps the layout its README describes.

- ``llm``, ``types`` and ``utils`` import none of each other; ``testing`` may import ``llm`` and ``types``.
- Core imports no other distribution, and no third-party library beyond ``httpx``, ``pydantic`` and ``tiktoken``.
- No distribution imports a private name from another distribution: they are released separately, so a private name
  could change under a dependent without a version bump.
- Only ``everalgo.utils.timestamps`` renders or parses a time, so every prompt shows a model one shape.
- Prompt markup (``tagged`` / ``paragraphs``), the chat line (``speaker`` / ``render_chat_line``) and reply parsing
  (``parse_reply`` / ``describe_errors``) come from core: no distribution defines a private copy of its own.

The imports are read from the source with ``ast``, so a check needs no package to be importable. Lives at the
repo-root ``tests/`` because the last rule spans every distribution, like ``test_logging_conventions.py``.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORE_SRC = _REPO_ROOT / "packages/everalgo-core/src/everalgo"

#: Which core subpackages each one may import; everything else inside ``everalgo.`` is off limits.
_ALLOWED_CORE_IMPORTS: dict[str, frozenset[str]] = {
    "llm": frozenset({"llm"}),
    "types": frozenset({"types"}),
    "utils": frozenset({"utils"}),
    "testing": frozenset({"testing", "llm", "types"}),
}
_CORE_THIRD_PARTY = frozenset({"httpx", "pydantic", "tiktoken"})

_TIME_MODULE = _REPO_ROOT / "packages/everalgo-core/src/everalgo/utils/timestamps.py"
_TIME_CODECS = frozenset({"fromtimestamp", "strftime", "strptime", "utcfromtimestamp"})
#: A strftime directive such as ``%Y``; a bare ``%`` (``{ratio:.1%}``) is the percent format, not a time.
_STRFTIME_DIRECTIVE = re.compile(r"%[A-Za-z]")

#: The private names the core helpers had while each package kept its own copy.
_CORE_HELPER_COPIES = frozenset(
    {"_tagged", "_paragraphs", "_open_tag_body", "_speaker", "_parse_reply", "_describe_errors"}
)
#: Files that held such a copy; each must still be scanned, so a moved or renamed file cannot pass vacuously.
_FORMER_COPY_HOLDERS = frozenset(
    _REPO_ROOT / f"packages/{relative}"
    for relative in (
        "everalgo-boundary/src/everalgo/boundary/_render.py",
        "everalgo-boundary/src/everalgo/boundary/_replies.py",
        "everalgo-boundary/src/everalgo/boundary/prompts/en/chat.py",
        "everalgo-user-memory/src/everalgo/user_memory/_replies.py",
        "everalgo-user-memory/src/everalgo/user_memory/episode/_replies.py",
        "everalgo-user-memory/src/everalgo/user_memory/profile/_replies.py",
        "everalgo-user-memory/src/everalgo/user_memory/prompts/en/atomic_fact_from_text.py",
        "everalgo-user-memory/src/everalgo/user_memory/prompts/en/episode.py",
        "everalgo-user-memory/src/everalgo/user_memory/prompts/en/profile_from_episode_texts.py",
        "everalgo-user-memory/src/everalgo/user_memory/prompts/en/reflect.py",
    )
)


def _imported_modules(path: Path) -> list[str]:
    modules = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.append(node.module)
            modules += [f"{node.module}.{alias.name}" for alias in node.names]
    return modules


def _core_modules() -> list[tuple[str, Path]]:
    """Each core source file with the subpackage it belongs to; top-level modules belong to none and are skipped."""
    return [(path.relative_to(_CORE_SRC).parts[0], path) for path in _CORE_SRC.glob("*/**/*.py")]


def test_core_subpackages_import_only_what_their_layer_allows() -> None:
    offenders = []
    for subpackage, path in _core_modules():
        allowed = _ALLOWED_CORE_IMPORTS[subpackage]
        for module in _imported_modules(path):
            parts = module.split(".")
            if parts[0] == "everalgo" and len(parts) > 1 and parts[1] not in allowed:
                offenders.append(f"{path.relative_to(_REPO_ROOT)} imports {module}")
    assert offenders == [], offenders


def test_core_depends_on_no_other_distribution_and_three_libraries_only() -> None:
    offenders = []
    for _, path in _core_modules():
        for module in _imported_modules(path):
            top = module.split(".")[0]
            if top == "everalgo" or top in sys.stdlib_module_names or top in _CORE_THIRD_PARTY:
                continue
            offenders.append(f"{path.relative_to(_REPO_ROOT)} imports {module}")
    assert offenders == [], offenders


def test_no_distribution_imports_a_private_name_of_another() -> None:
    offenders = []
    for path in _REPO_ROOT.glob("packages/*/src/everalgo/**/*.py"):
        own = path.relative_to(_REPO_ROOT / "packages").parts[0]
        for module in _imported_modules(path):
            parts = module.split(".")
            if parts[0] != "everalgo" or not any(part.startswith("_") for part in parts[1:]):
                continue
            if _distribution_of(parts[1]) not in (own, None):
                offenders.append(f"{path.relative_to(_REPO_ROOT)} imports {module}")
    assert offenders == [], offenders


def _distribution_of(subpackage: str) -> str | None:
    """Return the distribution that ships ``everalgo.<subpackage>``, or ``None`` if none does."""
    for candidate in (_REPO_ROOT / "packages").glob(f"*/src/everalgo/{subpackage}"):
        return candidate.relative_to(_REPO_ROOT / "packages").parts[0]
    for candidate in (_REPO_ROOT / "packages").glob(f"*/src/everalgo/{subpackage}.py"):
        return candidate.relative_to(_REPO_ROOT / "packages").parts[0]
    return None


def test_only_the_timestamps_module_renders_or_parses_a_time() -> None:
    offenders = []
    for path in sorted(_REPO_ROOT.glob("packages/*/src/**/*.py")):
        if path == _TIME_MODULE:
            continue
        offenders += [f"{path.relative_to(_REPO_ROOT)}:{line}: {what}" for line, what in _time_codecs(path)]
    assert offenders == [], offenders


def test_a_percent_format_spec_is_not_a_time_but_a_strftime_directive_is(tmp_path: Path) -> None:
    percent, date = tmp_path / "percent.py", tmp_path / "date.py"
    percent.write_text('ratio_text = f"{ratio:.1%}"\n', encoding="utf-8")
    date.write_text('date_text = f"{dt:%Y-%m-%d}"\n', encoding="utf-8")
    assert _time_codecs(percent) == []
    assert [line for line, _ in _time_codecs(date)] == [1]


def _time_codecs(path: Path) -> list[tuple[int, str]]:
    """Each ``strftime`` / ``strptime`` / ``fromtimestamp`` call and f-string spec with a ``%Y``-style directive."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _TIME_CODECS:
            found.append((node.lineno, f".{node.func.attr}()"))
        elif (
            isinstance(node, ast.FormattedValue)
            and node.format_spec is not None
            and _STRFTIME_DIRECTIVE.search(ast.unparse(node.format_spec))
        ):
            found.append((node.lineno, f"f-string format spec {ast.unparse(node.format_spec)}"))
    return found


def test_no_distribution_defines_its_own_copy_of_a_core_prompt_helper() -> None:
    sources = sorted(_REPO_ROOT.glob("packages/*/src/**/*.py"))
    offenders = [
        f"{path.relative_to(_REPO_ROOT)}:{node.lineno}: def {node.name}"
        for path in sources
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name in _CORE_HELPER_COPIES
    ]
    assert _FORMER_COPY_HOLDERS.issubset(sources)
    assert offenders == [], offenders
