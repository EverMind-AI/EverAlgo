"""Unit tests for ``parse_llm_json`` — extraction shapes and the invalid-escape repair."""

from __future__ import annotations

import json

from everalgo.knowledge._llm_json import parse_llm_json


def test_parses_a_fenced_json_block() -> None:
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parses_bare_json_inside_prose() -> None:
    assert parse_llm_json('Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}


def test_returns_none_without_a_json_object() -> None:
    assert parse_llm_json("no object here") is None
    assert parse_llm_json("") is None


def test_returns_none_for_a_non_dict_top_level() -> None:
    assert parse_llm_json("[1, 2, 3]") is None


def test_recovers_latex_backslashes_in_a_string_value() -> None:
    r"""The reported failure: LaTeX copied verbatim out of the source document.

    ``\k`` / ``\l`` / ``\s`` are not legal JSON escapes, so a complete, well-formed
    response was being discarded and the document extracted to nothing.
    """
    raw = '{"summary": "sensitivity $\\kappa$, gates $\\lambda_t$, softmax $\\sigma$"}'
    assert parse_llm_json(raw) == {"summary": "sensitivity $\\kappa$, gates $\\lambda_t$, softmax $\\sigma$"}


def test_latex_colliding_with_a_legal_escape_stays_lossy() -> None:
    r"""Known limit: ``\theta`` / ``\frac`` / ``\bar`` are indistinguishable from tab / FF / BS.

    At the JSON layer ``\t`` IS a tab, so the repair must leave it alone or it would corrupt
    genuine control-character escapes. The document is recovered; a few characters inside
    formulas are not. Measured on the reported sample: 0-7 such characters per response
    against ~5,000 characters of summary. The clean fix is upstream -- ask the model for
    ``response_format={"type": "json_object"}`` so the escapes are valid to begin with.
    """
    # The response carries a literal ``\pi`` and a literal ``\theta``. ``\p`` is repaired back
    # into a backslash; ``\t`` is a legal escape and lands as a tab, taking the ``h`` with it.
    assert parse_llm_json(r'{"s": "$\pi_\theta$"}') == {"s": "$\\pi_\theta$"}


def test_recovery_works_inside_a_fenced_block_too() -> None:
    assert parse_llm_json('```json\n{"s": "$\\kappa$"}\n```') == {"s": "$\\kappa$"}


def test_valid_escapes_survive_the_repair() -> None:
    r"""A correctly escaped backslash must not be split by the repair.

    The repair matches the escape PAIR; a lone-backslash scan would consume the first half
    of ``\\`` and turn the second half into a broken ``\p``.
    """
    raw = r'{"win": "C:\\path\\to", "quote": "she said \"hi\"", "nl": "a\nb", "u": "\u00e9", "tex": "\kappa"}'
    parsed = parse_llm_json(raw)
    assert parsed == {"win": r"C:\path\to", "quote": 'she said "hi"', "nl": "a\nb", "u": "é", "tex": "\\kappa"}


def test_well_formed_json_is_never_rewritten() -> None:
    """The repair is a fallback, so anything that already parses is returned untouched."""
    payload = {"summary": "a\\b", "topics": [{"topic": "t", "block_refs": "0-3"}]}
    assert parse_llm_json(json.dumps(payload)) == payload


def test_still_none_when_the_damage_is_not_escapes() -> None:
    """A set literal for ``block_refs`` (observed from one provider) stays unparseable."""
    assert parse_llm_json('{"topics": [{"block_refs": {230, 231}}]}') is None
