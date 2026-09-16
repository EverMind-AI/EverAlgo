"""Unit tests for ``everalgo.knowledge._block_split``."""

from __future__ import annotations

import pytest

from everalgo._tokenize import count_tokens
from everalgo.knowledge._block_split import (
    DEFAULT_MAX_TOKENS_PER_ATOM,
    DEFAULT_MAX_TOKENS_PER_BATCH,
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


# ── oversized-atom splitting (rule 5) ────────────────────────────────


def _table(rows: int, *, header: bool = True) -> str:
    lines = ["| Variant | Source | Rarity | Notes |", "|---|---|---|---|"] if header else []
    lines += [
        f"| Shiny variant #{i:04d} | Wandering merchant, biome {i % 40} | Rare | Winter update. |" for i in range(rows)
    ]
    return "\n".join(lines)


def test_unbounded_table_merge_no_longer_yields_an_over_budget_atom() -> None:
    """A page that is mostly one table used to collapse into a single mega-atom."""
    doc = f"# Title\n\n## Variants\n\n{_table(6000)}\n\n## References\n\nSee the index.\n"

    atoms = split_content_to_blocks(doc)

    assert len(atoms) > 6, "the table must no longer be one atom"
    assert all(count_tokens(t) <= DEFAULT_MAX_TOKENS_PER_ATOM for _, t in atoms)
    assert [i for i, _ in atoms] == list(range(len(atoms))), "ids stay dense and sequential"


def test_oversized_atom_pieces_repeat_the_table_header() -> None:
    doc = _table(6000)
    pieces = [t for _, t in split_content_to_blocks(doc)]

    assert len(pieces) > 1
    for piece in pieces:
        assert piece.startswith("| Variant | Source | Rarity | Notes |")
        assert piece.splitlines()[1] == "|---|---|---|---|"


def test_oversized_atom_pieces_preserve_every_row() -> None:
    """Splitting must not drop or reorder content — only the header is duplicated."""
    doc = _table(6000)
    pieces = [t for _, t in split_content_to_blocks(doc)]

    body = [ln for p in pieces for ln in p.splitlines() if ln.startswith("| Shiny variant")]
    assert body == doc.splitlines()[2:]


def test_long_list_is_split_without_a_header() -> None:
    doc = "\n".join(f"- Item {i}: the polished lens recipe was introduced here." for i in range(8000))

    atoms = split_content_to_blocks(doc)

    assert len(atoms) > 1
    assert all(count_tokens(t) <= DEFAULT_MAX_TOKENS_PER_ATOM for _, t in atoms)
    assert [ln for _, t in atoms for ln in t.splitlines()] == doc.splitlines()


def test_ordinary_document_is_untouched_by_the_cap() -> None:
    """The cap is loose enough that normal content sees byte-identical behaviour."""
    doc = f"# Title\n\nIntro paragraph.\n\n{_table(20)}\n\n- a\n- b\n\nClosing line.\n"

    assert split_content_to_blocks(doc) == split_content_to_blocks(doc, max_atom_tokens=0)


def test_cap_is_configurable_and_zero_disables_it() -> None:
    doc = _table(2000)

    assert len(split_content_to_blocks(doc, max_atom_tokens=0)) == 1
    tight = split_content_to_blocks(doc, max_atom_tokens=5_000)
    assert len(tight) > 1
    assert all(count_tokens(t) <= 5_000 for _, t in tight)


# ── adversarial: ways an atom could still slip past the cap ──────────


@pytest.mark.parametrize(
    ("label", "unit"),
    [
        ("ascii", "the quick brown fox jumps over the lazy dog "),
        ("simplified", "洗涤剂的总活性物含量决定去污能力和溶解性能"),
        ("traditional", "洗滌劑的總活性物含量決定去污能力與溶解性能"),  # >1 token per char
        ("rare_cjk", "翧翴翵翸翽耰耲耷耹耺耼聁聃聇聈聉"),  # ~2 tokens per char
        ("emoji", "🧺🧼🫧👕👖🧦"),  # ~2.7 tokens per char
        ("mixed", "洗衣 laundry 🧺 2,080,537,117.56 "),
    ],
)
def test_cap_holds_for_every_script(label: str, unit: str) -> None:
    """A character-count pre-filter is unsound: CJK and emoji exceed 1 token/char."""
    cap = 4_000
    doc = "\n".join(f"- {unit}" for _ in range(2_000))

    atoms = split_content_to_blocks(doc, max_atom_tokens=cap)

    oversized = [(i, count_tokens(t)) for i, t in atoms if count_tokens(t) > cap]
    assert not oversized, f"{label}: atoms over cap {cap}: {oversized}"


def test_single_line_longer_than_the_cap_is_split_on_token_boundaries() -> None:
    """CJK from a PDF often arrives as one very long unwrapped line."""
    cap = 1_000
    doc = "洗涤剂的总活性物含量决定去污能力和溶解性能。" * 3_000  # one line, no breaks

    atoms = split_content_to_blocks(doc, max_atom_tokens=cap)

    assert len(atoms) > 1
    assert all(count_tokens(t) <= cap for _, t in atoms)
    assert "".join(t for _, t in atoms) == doc.strip(), "no content may be lost"


def test_table_marker_path_is_also_capped() -> None:
    """Rule 1 (TABLE_START/END) is unbounded too, not just the native-row merge."""
    cap = 4_000
    rows = "\n".join(f"<tr><td>Variant {i}</td><td>Wandering merchant</td></tr>" for i in range(3_000))
    doc = f"# Title\n\n{TABLE_START_MARKER}\n{rows}\n{TABLE_END_MARKER}\n\nAfter.\n"

    atoms = split_content_to_blocks(doc, max_atom_tokens=cap)

    assert all(count_tokens(t) <= cap for _, t in atoms)
    assert atoms[-1][1] == "After."


def test_table_without_a_separator_row_still_splits() -> None:
    """Header detection must not be a precondition for the cap to apply."""
    cap = 4_000
    doc = "\n".join(f"| Variant {i} | Wandering merchant, biome {i % 40} |" for i in range(3_000))

    atoms = split_content_to_blocks(doc, max_atom_tokens=cap)

    assert len(atoms) > 1
    assert all(count_tokens(t) <= cap for _, t in atoms)


def test_header_larger_than_the_cap_does_not_loop_or_drop_content() -> None:
    """Degenerate: the repeated header alone already exceeds the budget."""
    header = "| " + " | ".join(f"column_{i}" for i in range(600)) + " |"
    doc = header + "\n|" + "---|" * 600 + "\n" + "\n".join(f"| row {i} |" for i in range(500))

    atoms = split_content_to_blocks(doc, max_atom_tokens=500)

    assert atoms, "must not return empty"
    assert all(count_tokens(t) <= 500 for _, t in atoms)


def test_ids_stay_dense_after_splitting_several_atoms() -> None:
    """``block_refs`` indexes this list positionally — gaps would corrupt every ref."""
    cap = 3_000
    table = "\n".join(f"| Variant {i} | merchant |" for i in range(2_000))
    lst = "\n".join(f"- Item {i}: the polished lens recipe." for i in range(2_000))
    doc = f"# Title\n\n{table}\n\nMiddle paragraph.\n\n{lst}\n\nEnd.\n"

    atoms = split_content_to_blocks(doc, max_atom_tokens=cap)

    assert [i for i, _ in atoms] == list(range(len(atoms)))


def test_split_atoms_batch_within_budget_end_to_end() -> None:
    """The whole point: no batch may exceed the window after splitting."""
    rows = "\n".join(
        f"| Shiny variant #{i:04d} | Trading with the wandering merchant in biome {i % 40} "
        f"| {'Rare' if i % 3 else 'Common'} | Introduced in the winter update. |"
        for i in range(6_000)
    )
    doc = f"# Shiny skin\n\n## Obtainable variants\n\n{rows}\n\n## References\n\nSee the index.\n"

    batches = split_and_batch_content(split_content_to_blocks(preprocess_content(doc)))

    over = [count_tokens(format_numbered_paragraphs(b)) for b in batches]
    assert all(n <= DEFAULT_MAX_TOKENS_PER_BATCH for n in over), f"over-budget batches: {over}"


def test_split_handles_special_token_literals() -> None:
    """Block splitting counts tokens, so it shared the tokenizer defect.

    A knowledge document about code completion quotes these literals; before the shared
    encode wrapper that document could not be ingested at all.
    """
    content = "The Qwen Coder FIM template is `<|fim_prefix|>{prefix}<|fim_suffix|>`.\n\n" * 20
    blocks = split_content_to_blocks(content)
    assert blocks  # split, not raised
