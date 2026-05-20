"""Aggregation utilities for benchmark reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import tiktoken

if TYPE_CHECKING:
    from benchmarks.common.stages.types import StageStats


def estimate_tokens(text: str, encoding: str = "o200k_base") -> int:
    """Estimate token count using tiktoken.

    Used for stages where actual API usage is not tracked (Stage 1 and
    Stage 2).  The estimate is tiktoken-based and is documented as
    approximate in the summary banner.

    Args:
        text: Text to estimate tokens for.
        encoding: tiktoken encoding name (default ``o200k_base`` for GPT-4o family).

    Returns:
        Estimated token count; ``0`` for empty input.
    """
    if not text:
        return 0
    enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(text))


def avg_context_tokens(
    answers: list[dict[str, Any]],
    *,
    encoding: str = "o200k_base",
) -> int:
    """Average tiktoken count of ``formatted_context`` across non-empty answers.

    Args:
        answers: List of answer records (each must have ``formatted_context`` key).
        encoding: tiktoken encoding name (default ``o200k_base`` for GPT-4o family).

    Returns:
        Integer mean token count; ``0`` if no non-empty contexts.
    """
    enc = tiktoken.get_encoding(encoding)
    counts = [len(enc.encode(ctx)) for a in answers if (ctx := a.get("formatted_context"))]
    return sum(counts) // len(counts) if counts else 0


def stage_summary(stages: list[StageStats]) -> dict[str, dict[str, Any]]:
    """Convert a list of StageStats into a per-stage summary dict for reports."""
    return {
        s.stage_name: {
            "duration_seconds": s.duration_seconds,
            "prompt_tokens": s.prompt_tokens,
            "completion_tokens": s.completion_tokens,
            "http_calls": s.http_calls,
            "success": s.success,
            "failed": s.failed,
        }
        for s in stages
    }
