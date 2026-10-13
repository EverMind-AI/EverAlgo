"""Tests for ``everalgo.utils.markup``."""

from __future__ import annotations

from everalgo.utils import paragraphs, tagged


def test_tagged_puts_each_tag_alone_on_its_line_and_the_body_verbatim() -> None:
    assert tagged("rules", "line one\n\nline two") == "<rules>\nline one\n\nline two\n</rules>"


def test_tagged_renders_attributes_on_the_open_tag_in_the_order_given() -> None:
    assert tagged("messages", "{{messages}}", count="3", order="chronological") == (
        '<messages count="3" order="chronological">\n{{messages}}\n</messages>'
    )


def test_tagged_takes_attributes_named_like_its_own_parameters() -> None:
    """``tag`` and ``body`` are positional-only, so an attribute may share their names."""
    assert tagged("item", "x", tag="label", body="b") == '<item tag="label" body="b">\nx\n</item>'


def test_paragraphs_puts_one_blank_line_between_blocks() -> None:
    assert paragraphs("a", "b\nc", "d") == "a\n\nb\nc\n\nd"


def test_paragraphs_of_one_block_is_that_block() -> None:
    assert paragraphs("only") == "only"
