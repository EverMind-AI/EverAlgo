"""Tests for ``everalgo.utils.markup``."""

from __future__ import annotations

import pytest

from everalgo.utils import append_custom_instructions, paragraphs, tagged


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


def test_append_custom_instructions_closes_the_system_message_with_the_tagged_block() -> None:
    assert append_custom_instructions("rules", "Be brief.") == (
        "rules\n\n<custom_instructions>\nBe brief.\n</custom_instructions>"
    )


@pytest.mark.parametrize("blank", [None, "", " ", "\n\t  \n"])
def test_append_custom_instructions_leaves_the_system_message_unchanged_for_none_or_blank(blank: str | None) -> None:
    assert append_custom_instructions("rules", blank) == "rules"


def test_append_custom_instructions_places_the_text_verbatim_and_never_renders_it() -> None:
    text = "  keep {{top_k}} and {x}  \n"
    assert (
        append_custom_instructions("rules", text) == f"rules\n\n<custom_instructions>\n{text}\n</custom_instructions>"
    )
