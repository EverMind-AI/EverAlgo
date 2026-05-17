"""Unit tests for :func:`everalgo.llm.parse.parse_llm_json_object`."""

from __future__ import annotations

import pytest

from everalgo.llm.parse import parse_llm_json_object


def test_direct_happy_path() -> None:
    """Tier-2 (direct loads) succeeds on a clean JSON-mode response."""
    assert parse_llm_json_object('{"a": 1}') == {"a": 1}


def test_direct_happy_path_with_whitespace() -> None:
    """Tier-2 succeeds when the response has leading/trailing whitespace."""
    assert parse_llm_json_object('  {"x": "hello"}  ') == {"x": "hello"}


def test_fenced_block() -> None:
    """Tier-1 (fence) extracts the object from a ```json block."""
    raw = '```json\n{"a": 1}\n```'
    assert parse_llm_json_object(raw) == {"a": 1}


def test_fenced_block_with_prose_around() -> None:
    """Tier-1 succeeds when prose surrounds the fenced block."""
    raw = 'Sure, here is the JSON:\n```json\n{"a": 1}\n```\nHope this helps.'
    assert parse_llm_json_object(raw) == {"a": 1}


def test_prose_wrapped_braces_no_fence() -> None:
    """Tier-3 (outermost braces) rescues a JSON object embedded in prose."""
    assert parse_llm_json_object('Sure! {"a": 1} Hope this helps.') == {"a": 1}


def test_nested_braces_in_outer_brace_mode() -> None:
    """Tier-3 correctly identifies the outermost braces when nested objects are present."""
    assert parse_llm_json_object('prose {"a": {"b": 2}} more') == {"a": {"b": 2}}


def test_top_level_array_raises() -> None:
    """Top-level arrays are rejected — only objects are accepted."""
    with pytest.raises(ValueError, match="Failed to parse LLM response as a JSON object"):
        parse_llm_json_object("[1, 2, 3]")


def test_bare_nonsense_raises() -> None:
    """Unparseable strings raise ValueError."""
    with pytest.raises(ValueError, match="Failed to parse LLM response as a JSON object"):
        parse_llm_json_object("not json at all")


def test_empty_string_raises() -> None:
    """Empty string raises ValueError."""
    with pytest.raises(ValueError, match="Failed to parse LLM response as a JSON object"):
        parse_llm_json_object("")
