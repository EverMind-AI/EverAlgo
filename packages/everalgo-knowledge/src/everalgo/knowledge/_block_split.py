"""Block splitting and token batching for ``KnowledgeExtractor``.

Pure-compute stage 1 of the extraction pipeline. No LLM calls, no I/O.
Normalizes parsed text, splits it into numbered atom blocks (paragraphs
with table / list merging), and groups atoms into token-bounded batches
ready for sequential LLM windows.

Higher-level normalization (HTML -> Markdown, table-marker injection) is
the parser's responsibility, not knowledge's. Callers that have already
emitted ``TABLE_START_MARKER`` / ``TABLE_END_MARKER`` around HTML-derived
tables get whole-table atoms; callers that have not pay no cost.

NOT exposed in the package ``__all__`` — these utilities are internal to
the knowledge extractor pipeline.
"""

from __future__ import annotations

import re
from functools import lru_cache

import tiktoken

__all__ = [
    "TABLE_END_MARKER",
    "TABLE_START_MARKER",
    "format_numbered_paragraphs",
    "preprocess_content",
    "split_and_batch_content",
    "split_content_to_blocks",
]

# Input convention markers — callers (typically the parser) emit these around
# HTML-derived tables so that an entire table survives as a single atom.
TABLE_START_MARKER = "TABLE_START"
TABLE_END_MARKER = "TABLE_END"

_O200K_ENCODING_NAME = "o200k_base"

# Default token budget per LLM-window batch. 80K leaves headroom on a 128K-context
# model after the topic-extraction prompt + JSON response. Tunable per call.
DEFAULT_MAX_TOKENS_PER_BATCH = 80_000


@lru_cache(maxsize=1)
def _get_tokenizer() -> tiktoken.Encoding:
    """Return the shared ``o200k_base`` encoding, initialising on first call."""
    return tiktoken.get_encoding(_O200K_ENCODING_NAME)


def preprocess_content(content: str) -> str:
    """Normalize parsed text before block splitting.

    Strips outer whitespace and collapses runs of three or more blank lines
    to a single blank-line separator. Returns an empty string for empty
    input.
    """
    if not content:
        return ""
    text = content.strip()
    return re.sub(r"\n{3,}", "\n\n", text)


_LIST_ITEM_RE = re.compile(r"^[-*]\s")


def _consume_table_marker_atom(lines: list[str], i: int) -> tuple[str, int]:
    """Consume an explicit ``TABLE_START`` ... ``TABLE_END`` block.

    Assumes ``lines[i].strip() == TABLE_START_MARKER`` on entry. Returns the
    merged atom text (may be empty if the bracket pair is empty) and the index
    just past the closing marker.
    """
    j = i + 1
    table_lines: list[str] = []
    while j < len(lines) and lines[j].strip() != TABLE_END_MARKER:
        stripped = lines[j].strip()
        if stripped:
            table_lines.append(stripped)
        j += 1
    if j < len(lines):
        j += 1  # consume TABLE_END
    return "\n".join(table_lines), j


def _consume_native_table_rows(lines: list[str], i: int) -> tuple[str, int]:
    """Consume a run of consecutive native markdown table rows.

    Assumes ``lines[i].strip().startswith('|')`` on entry. Returns merged text
    and the index of the first non-row line.
    """
    table_lines = [lines[i].strip()]
    j = i + 1
    while j < len(lines) and lines[j].strip().startswith("|"):
        table_lines.append(lines[j].strip())
        j += 1
    return "\n".join(table_lines), j


def _consume_list_items(lines: list[str], i: int) -> tuple[str, int]:
    """Consume a run of consecutive markdown list items (``-`` or ``*`` prefix).

    Assumes ``_LIST_ITEM_RE`` matches ``lines[i].strip()`` on entry. Returns
    merged text and the index of the first non-item line.
    """
    list_lines = [lines[i].strip()]
    j = i + 1
    while j < len(lines) and _LIST_ITEM_RE.match(lines[j].strip()):
        list_lines.append(lines[j].strip())
        j += 1
    return "\n".join(list_lines), j


def split_content_to_blocks(content: str) -> list[tuple[int, str]]:
    """Split ``content`` into indexed atom blocks ``[(id, text), ...]``.

    Merging rules (applied in this order):

    1. ``TABLE_START_MARKER`` / ``TABLE_END_MARKER`` enclose one atom — the
       entire bracketed body becomes a single block.
    2. Consecutive native markdown table rows (lines starting with ``|``) merge
       into one atom.
    3. Consecutive list items (lines starting with ``-`` or ``*`` followed by
       whitespace) merge into one atom.
    4. Otherwise each non-empty line becomes its own atom.

    Blank lines act only as separators and never become atoms.
    """
    if not content:
        return []

    lines = content.split("\n")
    blocks: list[tuple[int, str]] = []
    idx = 0
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line == TABLE_START_MARKER:
            merged, i = _consume_table_marker_atom(lines, i)
        elif line.startswith("|"):
            merged, i = _consume_native_table_rows(lines, i)
        elif _LIST_ITEM_RE.match(line):
            merged, i = _consume_list_items(lines, i)
        else:
            merged = line
            i += 1

        if merged:
            blocks.append((idx, merged))
            idx += 1

    return blocks


def split_and_batch_content(
    atoms: list[tuple[int, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_BATCH,
) -> list[list[tuple[int, str]]]:
    """Greedy bin-packing of atoms into batches by ``o200k_base`` token count.

    Each returned batch fits within ``max_tokens`` tokens — except for the
    pathological case of a single atom that already exceeds the limit, which
    becomes its own (over-budget) batch rather than being silently dropped.
    Atoms are never split across batches; ordering is preserved.

    An empty ``atoms`` list returns an empty list (zero batches).
    """
    if not atoms:
        return []

    enc = _get_tokenizer()
    batches: list[list[tuple[int, str]]] = []
    current_batch: list[tuple[int, str]] = []
    current_tokens = 0

    for atom in atoms:
        atom_tokens = len(enc.encode(atom[1]))
        if current_batch and current_tokens + atom_tokens > max_tokens:
            batches.append(current_batch)
            current_batch = [atom]
            current_tokens = atom_tokens
        else:
            current_batch.append(atom)
            current_tokens += atom_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def format_numbered_paragraphs(atoms: list[tuple[int, str]]) -> str:
    """Format atoms as ``'ID: content'`` lines for LLM prompt input."""
    return "\n".join(f"{idx}: {text}" for idx, text in atoms)
