"""Aggregation utilities for benchmark reports.

Every ``encode`` here passes ``disallowed_special=()``: ``Encoding.encode`` otherwise raises
``ValueError`` on a transcript that merely quotes ``<|endoftext|>`` / ``<|fim_prefix|>``, which would
abort a benchmark run over a corpus it is only supposed to be measuring. The library side routes
through ``everalgo.utils.tokens.encode`` and ``tests/test_tokenizer_conventions.py`` enforces that; this
module is exempt from the shared wrapper because it takes the encoding name as an argument, so the
rule is carried here by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import tiktoken

from everalgo.llm import TextPart

if TYPE_CHECKING:
    from collections.abc import Sequence

    from benchmarks.common.stages.types import StageStats
    from everalgo.llm import LLMMessage


def estimate_tokens(text: str, encoding: str = "o200k_base") -> int:
    """Estimate token count using tiktoken.

    ``LLMClient.achat`` returns only the reply text, so every stage reports
    estimated rather than billed usage; the summary banner says so.

    Args:
        text: Text to estimate tokens for.
        encoding: tiktoken encoding name (default ``o200k_base`` for GPT-4o family).

    Returns:
        Estimated token count; ``0`` for empty input.
    """
    if not text:
        return 0
    enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(text, disallowed_special=()))


def estimate_chat_tokens(messages: Sequence[LLMMessage], reply: str) -> tuple[int, int]:
    """Estimate the prompt and completion tokens of one ``achat`` call.

    Only text is counted: a multimodal part other than text contributes nothing.

    Args:
        messages: The messages the call sent.
        reply: The reply text the call returned.

    Returns:
        ``(prompt_tokens, completion_tokens)``.
    """
    prompt_tokens = sum(estimate_tokens(text) for message in messages for text in _message_texts(message))
    return prompt_tokens, estimate_tokens(reply)


def _message_texts(message: LLMMessage) -> list[str]:
    """The text of a message: its string content, or the text of each of its text parts."""
    if isinstance(message.content, str):
        return [message.content]
    return [part.text for part in message.content if isinstance(part, TextPart)]


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
    counts = [len(enc.encode(ctx, disallowed_special=())) for a in answers if (ctx := a.get("formatted_context"))]
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
