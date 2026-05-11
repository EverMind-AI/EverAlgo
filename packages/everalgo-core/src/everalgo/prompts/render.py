"""Render a prompt template with caller-provided fields, falling back to a default.

Centralises the ``(prompt or DEFAULT).format(**fields)`` pattern used by
every LLM-calling operator in EverAlgo, so that future cross-cutting
concerns (prompt logging, escape rules, i18n switching, instrumentation)
have a single edit point.
"""

from __future__ import annotations

from typing import Any


def render_prompt(default: str, prompt: str | None, /, **fields: Any) -> str:
    """Render ``prompt`` with ``fields``; if ``prompt`` is None, use ``default``.

    Designed for operators that accept an optional caller-override of the
    prompt template while still shipping a sensible default. ``default``
    and ``prompt`` are positional-only so callers cannot accidentally
    swap their order via keyword arguments.

    Args:
        default: Module-level prompt constant shipped with the operator
            (for example ``CHAT_BOUNDARY_DETECT_PROMPT_EN``).
        prompt: Caller override; if ``None``, falls back to ``default``.
        **fields: Keyword arguments substituted into the template via
            :py:meth:`str.format`. The template is responsible for naming
            each placeholder; missing placeholders raise :class:`KeyError`.

    Returns:
        The rendered prompt string, ready to send to the LLM.

    Raises:
        KeyError: If the template references a placeholder not in ``fields``.

    Example:
        >>> render_prompt("Hello, {name}!", None, name="world")
        'Hello, world!'
        >>> render_prompt("Hello, {name}!", "Hi {name}", name="world")
        'Hi world'
    """
    return (prompt or default).format(**fields)
