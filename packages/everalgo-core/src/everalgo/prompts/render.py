"""Render a prompt template with caller-provided fields, falling back to a default.

Mirrors Python's :py:meth:`str.format` brace-escape semantics: ``{{`` / ``}}`` collapse to literal
``{`` / ``}``, while ``{key}`` placeholders are substituted from kwargs. Prompts can be authored
exactly the same way they would be for ``.format`` (including JSON examples written as
``{{...}}``) and pass through this renderer byte-identical.
"""

from __future__ import annotations

from typing import Any

# Sentinels for {{ }} brace-escape masking. NUL-flanked so they cannot collide with any prompt
# content (LLM prompts are text and NUL bytes are forbidden in practice). The mask hop matters
# because a naive ``replace("{key}", value)`` over a template containing ``{{...}}`` JSON examples
# would otherwise leave the doubled braces in place, and downstream LLM-as-judge parses
# ``{{"label": ...}}`` as malformed JSON.
_LB_SENTINEL = "\x00\x00EVERALGO_LB\x00\x00"
_RB_SENTINEL = "\x00\x00EVERALGO_RB\x00\x00"


def render_prompt(default: str, prompt: str | None, /, **fields: Any) -> str:
    """Render ``prompt`` with ``fields``; if ``prompt`` is None, use ``default``.

    Mirrors :py:meth:`str.format`: ``{{`` becomes literal ``{``, ``}}`` becomes literal ``}``,
    and each ``{key}`` placeholder is replaced with ``str(value)``. Unescaped single ``{`` /
    ``}`` (e.g. unbalanced ``{"key": value}`` shards) are left verbatim rather than raising —
    this is the only deliberate deviation from ``.format``, which would ``KeyError`` on them.

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
        >>> render_prompt('Output JSON like {{"key": "value"}} for {name}', None, name="Alice")
        'Output JSON like {"key": "value"} for Alice'
    """
    template = prompt or default
    template = template.replace("{{", _LB_SENTINEL).replace("}}", _RB_SENTINEL)
    for key, value in fields.items():
        template = template.replace("{" + key + "}", str(value))
    return template.replace(_LB_SENTINEL, "{").replace(_RB_SENTINEL, "}")
