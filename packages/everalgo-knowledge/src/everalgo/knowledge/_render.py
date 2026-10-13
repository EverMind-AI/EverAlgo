"""The close of every knowledge system message: the caller's ``<custom_instructions>`` block, when there is one."""

from __future__ import annotations

__all__ = ["append_custom_instructions"]

from everalgo.utils import paragraphs, tagged


def append_custom_instructions(rules: str, custom_instructions: str | None) -> str:
    """Close a system message with the caller's ``<custom_instructions>`` block.

    The block adds to the rules, never replaces them, and is always the message's last block. Its text is placed
    verbatim and never rendered as a template, so a ``{{...}}`` in it stays literal.

    Args:
        rules: The system message's rules, fully assembled.
        custom_instructions: The caller's instructions; None or a blank string leaves ``rules`` unchanged.

    Returns:
        The system message.
    """
    if custom_instructions is None or not custom_instructions.strip():
        return rules
    return paragraphs(rules, tagged("custom_instructions", custom_instructions))
