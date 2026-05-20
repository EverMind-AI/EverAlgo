"""Tests for benchmark metric aggregation."""

from __future__ import annotations

from typing import Any

import tiktoken

from benchmarks.common.metrics import avg_context_tokens, stage_summary
from benchmarks.common.stages.types import StageStats


def test_avg_context_tokens_uses_o200k_base():
    """Default encoding is o200k_base; verify tokens > 0 for non-empty contexts."""
    answers = [
        {"formatted_context": "hello world"},
        {"formatted_context": "this is a longer test context with more tokens"},
    ]
    avg = avg_context_tokens(answers)
    # Spot check: avg should be roughly len(text)/4 in tokens
    enc = tiktoken.get_encoding("o200k_base")
    expected_total = len(enc.encode("hello world")) + len(enc.encode("this is a longer test context with more tokens"))
    assert avg == expected_total // 2  # integer mean


def test_avg_context_tokens_skips_empty_contexts():
    answers: list[dict[str, Any]] = [
        {"formatted_context": "hello"},
        {"formatted_context": ""},
        {"formatted_context": None},
        {},  # no formatted_context key
    ]
    avg = avg_context_tokens(answers)
    # Only one non-empty context counted
    enc = tiktoken.get_encoding("o200k_base")
    assert avg == len(enc.encode("hello"))


def test_avg_context_tokens_empty_list_returns_zero():
    assert avg_context_tokens([]) == 0


def test_avg_context_tokens_custom_encoding():
    answers = [{"formatted_context": "hello"}]
    avg = avg_context_tokens(answers, encoding="cl100k_base")
    enc = tiktoken.get_encoding("cl100k_base")
    assert avg == len(enc.encode("hello"))


def test_stage_summary_aggregates_per_stage():
    stages = [
        StageStats(
            stage_name="extract",
            duration_seconds=10.0,
            prompt_tokens=100,
            completion_tokens=20,
            http_calls=5,
            success=10,
            failed=0,
        ),
        StageStats(
            stage_name="answer",
            duration_seconds=5.0,
            prompt_tokens=50,
            completion_tokens=10,
            http_calls=3,
            success=3,
            failed=1,
        ),
    ]
    summary = stage_summary(stages)
    assert summary["extract"]["duration_seconds"] == 10.0
    assert summary["extract"]["prompt_tokens"] == 100
    assert summary["extract"]["completion_tokens"] == 20
    assert summary["extract"]["http_calls"] == 5
    assert summary["extract"]["success"] == 10
    assert summary["extract"]["failed"] == 0
    assert summary["answer"]["duration_seconds"] == 5.0
    assert summary["answer"]["failed"] == 1


def test_stage_summary_empty_list_returns_empty_dict():
    assert stage_summary([]) == {}
