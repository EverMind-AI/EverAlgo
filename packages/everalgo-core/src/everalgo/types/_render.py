"""Content rendering helpers shared across EverAlgo packages.

Internal module (``_render``-prefixed) — consumed by ``everalgo-user-memory`` and
``everalgo-agent-memory`` to render ``ChatMessage.content`` into a plain string for LLM prompt
assembly.  Not re-exported from ``everalgo.types.__init__`` because it is an implementation detail,
not a public data contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everalgo.types.content import TextContent


def render_content(content: str | list[TextContent]) -> str:  # type: ignore[type-arg]
    """Render ``ChatMessage.content`` to a single string for LLM prompt assembly.

    Args:
        content: ``str`` — returned as-is (string shorthand for plain text).
            ``list[ContentBlock]`` — block ``text`` attribute concatenated with a space; blocks without
            a ``text`` attribute (future ``ImageContent`` / ``AudioContent``) rendered as
            ``[<type>]`` placeholders.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        # ``ContentBlock`` is currently ``TextContent``; future union members (ImageContent, etc.)
        # will not have a `.text` attribute, so getattr is the forward-compatible branch.
        text_val: str | None = getattr(block, "text", None)
        if text_val:
            parts.append(text_val)
        else:
            block_type: str = getattr(block, "type", "unknown")
            parts.append(f"[{block_type}]")
    return " ".join(parts)
