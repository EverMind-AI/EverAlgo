"""Prompt validators — fail-fast checks for prompt templates.

Designed to be called at module import time after a prompt constant is
defined, so that template typos are caught before any LLM call.
"""

import string
from collections.abc import Callable, Iterable


def check_placeholders(prompt: str, *, required: Iterable[str]) -> None:
    """Assert that ``prompt`` contains every Python format placeholder in ``required``.

    The check uses :class:`string.Formatter` so attribute access
    (``{user.name}``) and indexing (``{items[0]}``) collapse to the root
    identifier (``user`` / ``items``).

    Args:
        prompt: Template string with ``{placeholder}`` markers.
        required: Names that must appear as ``{name}`` in the template.

    Raises:
        ValueError: If any required placeholder is missing. The diagnostic
            message lists the missing names and, when present, any extra
            placeholders the template carries — useful for catching typos
            such as ``{nme}`` instead of ``{name}``.
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
    """Coarse-but-safe over-estimate (~ 4 characters per token, English baseline).

    This intentionally over-counts (especially for CJK text) so that a
    too-long prompt is never silently allowed to pass. Callers wanting an
    accurate token count should pass a real tokenizer (for example
    ``tiktoken.encoding_for_model("gpt-4").encode``).
    """
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
        max_tokens: Hard ceiling — typically the model context window minus
            the response reserve.
        tokenizer: Token counter callable. ``None`` (default) falls back to
            an over-counting heuristic; for precise token accounting pass
            an accurate tokenizer.

    Raises:
        ValueError: If the estimated token count exceeds ``max_tokens``.
            The message includes both the actual count and the cap.
    """
    counter = tokenizer if tokenizer is not None else _default_token_estimator
    actual = counter(prompt)
    if actual > max_tokens:
        raise ValueError(f"Prompt length {actual} tokens exceeds max_tokens={max_tokens}")
