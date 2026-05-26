"""Unit tests for :func:`everalgo.llm.parse.parse_llm_json_object`."""

from __future__ import annotations

import json

import pytest

from everalgo.llm.parse import extract_final_answer, parse_llm_json_object, retry_on_json_parse_failure


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


# ---------------------------------------------------------------------------
# retry_on_json_parse_failure
# ---------------------------------------------------------------------------


async def test_retry_succeeds_on_first_try() -> None:
    calls = 0

    @retry_on_json_parse_failure(max_retries=3)
    async def fn() -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"ok": 1}

    result = await fn()
    assert result == {"ok": 1}
    assert calls == 1


async def test_retry_recovers_after_one_failure() -> None:
    calls = 0

    @retry_on_json_parse_failure(max_retries=3)
    async def fn() -> dict[str, int]:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise json.JSONDecodeError("bad", "", 0)
        return {"ok": 1}

    result = await fn()
    assert result == {"ok": 1}
    assert calls == 2


async def test_retry_exhausts_then_raises() -> None:
    """After max_retries failed attempts, raise the last exception."""
    calls = 0

    @retry_on_json_parse_failure(max_retries=2)
    async def fn() -> dict[str, int]:
        nonlocal calls
        calls += 1
        raise json.JSONDecodeError("bad", "", 0)

    with pytest.raises(json.JSONDecodeError):
        await fn()
    assert calls == 2, "max_retries=2 should produce exactly 2 attempts"


async def test_retry_retries_on_value_error_by_default() -> None:
    calls = 0

    @retry_on_json_parse_failure(max_retries=3)
    async def fn() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise ValueError("bad value")
        return "ok"

    result = await fn()
    assert result == "ok"
    assert calls == 2


async def test_retry_does_not_swallow_non_retryable_exceptions() -> None:
    """Non-retryable exceptions propagate immediately without consuming retry budget."""
    calls = 0

    @retry_on_json_parse_failure(max_retries=3)
    async def fn() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError):
        await fn()
    assert calls == 1, "Non-retryable exceptions should NOT trigger retries"


async def test_retry_max_retries_3_invokes_exactly_3_times() -> None:
    """max_retries=3 means 3 total attempts (first call + 2 retries)."""
    calls = 0

    @retry_on_json_parse_failure(max_retries=3)
    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise json.JSONDecodeError("always", "", 0)

    with pytest.raises(json.JSONDecodeError):
        await fn()
    assert calls == 3, f"Expected exactly 3 attempts, got {calls}"


async def test_retry_custom_retryable_exc_narrows_retry_set() -> None:
    """Passing retryable_exc=(RuntimeError,) makes RuntimeError retryable and json.JSONDecodeError NOT retryable."""
    calls = 0

    @retry_on_json_parse_failure(max_retries=3, retryable_exc=(RuntimeError,))
    async def retries_runtime_error() -> str:
        nonlocal calls
        calls += 1
        if calls < 2:
            raise RuntimeError("transient")
        return "ok"

    result = await retries_runtime_error()
    assert result == "ok"
    assert calls == 2


async def test_retry_custom_retryable_exc_excludes_default_types() -> None:
    """When retryable_exc=(RuntimeError,), json.JSONDecodeError propagates immediately."""
    calls = 0

    @retry_on_json_parse_failure(max_retries=3, retryable_exc=(RuntimeError,))
    async def fn() -> str:
        nonlocal calls
        calls += 1
        raise json.JSONDecodeError("not in retryable_exc", "", 0)

    with pytest.raises(json.JSONDecodeError):
        await fn()
    assert calls == 1, "JSONDecodeError should not be retried when not in retryable_exc"


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------


def test_extract_final_answer_simple() -> None:
    raw = "Some reasoning here.\nFinal answer: Paris"
    assert extract_final_answer(raw) == "Paris"


def test_extract_final_answer_no_marker_returns_full_text_stripped() -> None:
    raw = "  no marker here  "
    assert extract_final_answer(raw) == "no marker here"


def test_extract_final_answer_custom_marker() -> None:
    raw = "...\n## Answer ##\nfoo"
    assert extract_final_answer(raw, marker="## Answer ##") == "foo"


def test_extract_final_answer_marker_at_start() -> None:
    raw = "Final answer: directly"
    assert extract_final_answer(raw) == "directly"


def test_extract_final_answer_marker_with_trailing_whitespace() -> None:
    raw = "Final answer:    multiple   spaces   "
    assert extract_final_answer(raw) == "multiple   spaces"


def test_extract_final_answer_uses_last_occurrence_when_marker_repeats() -> None:
    """When marker appears in reasoning prose AND in the final answer block, take last."""
    raw = "Reasoning: Final answer: was rejected\n...\nFinal answer: actual answer"
    assert extract_final_answer(raw) == "actual answer"
