"""Unit tests for ``everalgo.knowledge._block_split``."""

from __future__ import annotations

from everalgo.knowledge._block_split import (
    TABLE_END_MARKER,
    TABLE_START_MARKER,
    format_numbered_paragraphs,
    preprocess_content,
    split_and_batch_content,
    split_content_to_blocks,
)

# ── preprocess_content ───────────────────────────────────────────────


def test_preprocess_content_empty_returns_empty() -> None:
    assert preprocess_content("") == ""


def test_preprocess_content_strips_outer_whitespace() -> None:
    assert preprocess_content("\n\n  hello world  \n\n") == "hello world"


def test_preprocess_content_collapses_blank_line_runs() -> None:
    assert preprocess_content("a\n\n\n\n\nb") == "a\n\nb"


def test_preprocess_content_preserves_single_blank_line() -> None:
    assert preprocess_content("a\n\nb") == "a\n\nb"


# ── split_content_to_blocks ──────────────────────────────────────────


def test_split_empty_content() -> None:
    assert split_content_to_blocks("") == []


def test_split_basic_paragraphs() -> None:
    content = "first paragraph\n\nsecond paragraph\n\nthird paragraph"
    blocks = split_content_to_blocks(content)
    assert blocks == [
        (0, "first paragraph"),
        (1, "second paragraph"),
        (2, "third paragraph"),
    ]


def test_split_table_markers_merge_to_one_block() -> None:
    content = f"intro\n\n{TABLE_START_MARKER}\nrow1\nrow2\nrow3\n{TABLE_END_MARKER}\n\nouter"
    blocks = split_content_to_blocks(content)
    assert blocks == [
        (0, "intro"),
        (1, "row1\nrow2\nrow3"),
        (2, "outer"),
    ]


def test_split_markdown_table_rows_merge() -> None:
    content = "intro\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nafter"
    blocks = split_content_to_blocks(content)
    assert blocks == [
        (0, "intro"),
        (1, "| a | b |\n|---|---|\n| 1 | 2 |"),
        (2, "after"),
    ]


def test_split_list_items_merge_dash_and_star() -> None:
    content = "intro\n\n- item1\n- item2\n* item3\n\nafter"
    blocks = split_content_to_blocks(content)
    assert blocks == [
        (0, "intro"),
        (1, "- item1\n- item2\n* item3"),
        (2, "after"),
    ]


def test_split_skips_blank_lines_inside_runs() -> None:
    content = "a\n\n\nb"
    blocks = split_content_to_blocks(content)
    assert blocks == [(0, "a"), (1, "b")]


# ── split_and_batch_content ──────────────────────────────────────────


def test_batch_empty_atoms_returns_empty() -> None:
    assert split_and_batch_content([]) == []


def test_batch_under_limit_keeps_single_batch() -> None:
    atoms = [(0, "hello"), (1, "world")]
    batches = split_and_batch_content(atoms, max_tokens=1000)
    assert len(batches) == 1
    assert batches[0] == atoms


def test_batch_splits_at_token_limit() -> None:
    # Each single-letter atom encodes to 1 ``o200k_base`` token, so max_tokens=1
    # forces every atom into its own batch.
    atoms = [(0, "a"), (1, "b"), (2, "c")]
    batches = split_and_batch_content(atoms, max_tokens=1)
    assert batches == [[(0, "a")], [(1, "b")], [(2, "c")]]


def test_batch_oversize_atom_stays_in_own_batch() -> None:
    # An atom larger than max_tokens still produces one batch rather than being dropped.
    atoms = [(0, "x" * 50)]
    batches = split_and_batch_content(atoms, max_tokens=1)
    assert len(batches) == 1
    assert batches[0] == atoms


# ── format_numbered_paragraphs ───────────────────────────────────────


def test_format_numbered_paragraphs_basic() -> None:
    atoms = [(0, "first"), (1, "second")]
    assert format_numbered_paragraphs(atoms) == "0: first\n1: second"


def test_format_numbered_paragraphs_empty() -> None:
    assert format_numbered_paragraphs([]) == ""
