"""Cross-distribution tokenizer convention checks.

Every tiktoken encode in a PUBLISHED DISTRIBUTION (``packages/*/src``) must go through
``everalgo._tokenize.encode``, which passes ``disallowed_special=()``.

WHY THIS IS A BUILD-BREAKING RULE RATHER THAN A STYLE PREFERENCE.
``Encoding.encode`` defaults to ``disallowed_special="all"``, so it raises ValueError the
moment the input contains a literal such as ``<|endoftext|>`` or ``<|fim_prefix|>``. That
default is meant for a caller ASSEMBLING A PROMPT, where a user-supplied control token
would be a real injection. This library never assembles a prompt at the tokenizer layer —
it measures and slices text — so the default instead makes ordinary content
unprocessable: a conversation that merely *quotes* those literals (any discussion of code
completion or tokenizers does) fails, and keeps failing, because the caller retries the
same text forever.

Three separate modules had grown their own raw ``encode`` call, and only one of them was
found by the incident. A convention test is the cheapest way to stop the fourth.

``benchmarks/`` is a workspace member but is deliberately out of scope. It is internal, never
published, and ``benchmarks.common.metrics`` takes the encoding name as an argument, so it cannot
route through the o200k-only shared wrapper without losing that. It passes ``disallowed_special=()``
at its own call sites instead; the rule there is carried by a comment, not by this test.

Lives at the repo-root ``tests/`` rather than under one distribution because the contract
spans every distribution — the same reasoning as ``test_logging_conventions.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOKENIZE_MODULE = "packages/everalgo-core/src/everalgo/_tokenize.py"

#: Binding a tokenizer object and calling ``.encode`` on it directly — the shape that
#: skips the wrapper even without importing tiktoken (``from everalgo._tokenize import
#: _get_tokenizer``; ``_get_tokenizer().encode(text)``).
_DIRECT_ENCODE = re.compile(r"(?:_get_tokenizer\(\)|\btokenizer|\benc)\s*\.encode\(")


def _source_files() -> list[Path]:
    return sorted(_REPO_ROOT.glob("packages/*/src/**/*.py"))


def test_tiktoken_is_imported_in_exactly_one_module() -> None:
    """Only the shared tokenizer module may reach for tiktoken.

    You cannot call ``Encoding.encode`` without first obtaining an ``Encoding``, so
    confining the import confines the defect.
    """
    offenders = [
        str(path.relative_to(_REPO_ROOT))
        for path in _source_files()
        if re.search(r"^\s*import tiktoken\b|^\s*from tiktoken\b", path.read_text(encoding="utf-8"), re.M)
    ]
    assert offenders == [_TOKENIZE_MODULE], (
        "in a published distribution tiktoken must be imported only by the shared tokenizer "
        f"module ({_TOKENIZE_MODULE}); found: {offenders}. Use everalgo._tokenize.encode / "
        "count_tokens instead — a direct encode raises on text containing "
        "<|endoftext|> and friends."
    )


def test_no_module_calls_encode_on_a_tokenizer_directly() -> None:
    """Even with the import confined, a module can still fetch the shared Encoding.

    ``_tokenize.py`` itself is exempt: it owns the one sanctioned call, and it passes
    ``disallowed_special=()``.
    """
    offenders = []
    for path in _source_files():
        rel = str(path.relative_to(_REPO_ROOT))
        if rel == _TOKENIZE_MODULE:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _DIRECT_ENCODE.search(line):
                offenders.append(f"{rel}:{lineno}")
    assert offenders == [], (
        f"direct tokenizer .encode() outside {_TOKENIZE_MODULE}: {offenders}. "
        "Call everalgo._tokenize.encode (or count_tokens) so special-token literals "
        "are treated as ordinary text."
    )


def test_the_sanctioned_call_still_disables_the_check() -> None:
    """The exemption above is only safe while the one call it exempts stays safe."""
    source = (_REPO_ROOT / _TOKENIZE_MODULE).read_text(encoding="utf-8")
    encode_calls = re.findall(r"\.encode\([^)]*\)", source)
    assert encode_calls, "expected at least one encode call in the shared tokenizer module"
    assert all("disallowed_special=()" in call for call in encode_calls), (
        f"every encode in {_TOKENIZE_MODULE} must pass disallowed_special=(); got {encode_calls}"
    )
