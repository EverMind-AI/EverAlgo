"""Render a prompt template with caller-provided fields, falling back to a default.

Uses ``str.replace`` to substitute ``{key}`` placeholders so the template body can contain
literal ``{`` / ``}`` (e.g. JSON examples) without escaping.
"""

from __future__ import annotations

from typing import Any


def render_prompt(default: str, prompt: str | None, /, **fields: Any) -> str:
    """Render ``prompt`` with ``fields``; if ``prompt`` is None, use ``default``.

    Substitutes each ``{key}`` placeholder in the template with ``str(value)``. Template bodies
    may freely contain literal braces (e.g. JSON examples) — only the exact ``{key}`` substring
    is replaced.

    Args:
        default: Module-level prompt constant shipped with the operator.
        prompt: Caller override; if ``None``, falls back to ``default``.
        **fields: Each key=value renders ``{key}`` → ``str(value)``. Missing placeholders are left
            in the output verbatim (no KeyError) so a typo'd template degrades gracefully rather
            than crashing the LLM call.

    Returns:
        Rendered prompt string.

    Examples:
        >>> render_prompt("Hello, {name}!", None, name="world")
        'Hello, world!'
        >>> render_prompt("Hello, {name}!", "Hi {name}", name="world")
        'Hi world'
        >>> render_prompt('Output JSON like {"key": "value"} for {name}', None, name="Alice")
        'Output JSON like {"key": "value"} for Alice'
    """
    template = prompt or default
    for key, value in fields.items():
        template = template.replace("{" + key + "}", str(value))
    return template
