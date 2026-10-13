"""XML-style markup for prompt text: a block wrapped in its tag, and blocks joined into one message."""

from __future__ import annotations

__all__ = ["paragraphs", "tagged"]


def tagged(tag: str, body: str, /, **attributes: str) -> str:
    """Wrap a block in its XML-style tag, open and close tag each alone on its line.

    Args:
        tag: Descriptive snake_case tag name.
        body: Block text, placed verbatim between the tags.
        **attributes: Rendered on the open tag as ``name="value"``, in the order given, unescaped.

    Returns:
        The tagged block, e.g. ``<messages count="3">``, a newline, ``body``, a newline, ``</messages>``.
    """
    opening = " ".join([tag, *(f'{name}="{value}"' for name, value in attributes.items())])
    return f"<{opening}>\n{body}\n</{tag}>"


def paragraphs(*blocks: str) -> str:
    """Join blocks into one text, one blank line between them."""
    return "\n\n".join(blocks)
