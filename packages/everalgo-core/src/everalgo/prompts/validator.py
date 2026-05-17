"""Prompt validators — fail-fast checks for prompt templates.

Designed to be called at module import time after a prompt constant is defined, so that template typos are
caught before any LLM call.
"""

import string
from collections.abc import Callable, Iterable


def check_placeholders(prompt: str, *, required: Iterable[str]) -> None:
    """Assert that ``prompt`` contains every placeholder in ``required``; attribute/index access collapses to root.

    Raises:
        ValueError: If any required placeholder is missing; error message also lists any extra placeholders.
    """
    found: set[str] = set()
    for _, field_name, _, _ in string.Formatter().parse(prompt):
        if not field_name:
            continue
        # Reduce ``user.name`` / ``items[0]`` to root identifier.
        root = field_name.split(".", 1)[0].split("[", 1)[0]
        if root:
            found.add(root)

    required_set = set(required)
    missing = required_set - found
    if not missing:
        return

    extras = found - required_set
    msg = f"Missing required placeholders: {sorted(missing)}"
    if extras:
        msg += f" (extra placeholders present: {sorted(extras)})"
    raise ValueError(msg)


def _default_token_estimator(text: str) -> int:
    """Return a coarse over-estimate (~4 chars/token); intentionally conservative so CJK text never slips through."""
    return max(1, len(text) // 4 + 1)


def check_length(
    prompt: str,
    *,
    max_tokens: int,
    tokenizer: Callable[[str], int] | None = None,
) -> None:
    """Assert that ``prompt`` is at most ``max_tokens`` tokens long.

    Args:
        prompt: Rendered prompt (post-format).
        max_tokens: Hard ceiling — typically context window minus response reserve.
        tokenizer: Token counter; ``None`` falls back to the conservative character-based heuristic.

    Raises:
        ValueError: If the token count exceeds ``max_tokens``.
    """
    counter = tokenizer if tokenizer is not None else _default_token_estimator
    actual = counter(prompt)
    if actual > max_tokens:
        raise ValueError(f"Prompt length {actual} tokens exceeds max_tokens={max_tokens}")
