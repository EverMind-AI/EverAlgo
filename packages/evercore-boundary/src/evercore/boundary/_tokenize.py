"""Token counting helper for boundary extractors.

Minimal reference implementation using a 4-character heuristic (roughly
matching English GPT tokens). For production use, replace with tiktoken or a
real tokenizer (planned for a future SemVer minor bump).

NOT exposed in __all__ — module-private utility for boundary algorithms.
"""

from __future__ import annotations

_CHARS_PER_TOKEN_HEURISTIC = 4


def count_tokens(text: str) -> int:
    """Estimate token count — minimal reference impl.

    Uses ``len(text) // CHARS_PER_TOKEN`` as a rough proxy for GPT-style
    tokenization. Accuracy is sufficient for "is this MemCell larger than
    the LLM context window?" decisions but NOT for billing / quota.

    Args:
        text: Input string. Empty string returns 0.

    Returns:
        Estimated token count (always >= 0).
    """
    return len(text) // _CHARS_PER_TOKEN_HEURISTIC
