"""Token-aware text utilities for agent_memory extractors.

Builds on :mod:`everalgo._tokenize` (the shared ``o200k_base`` encoder + ``count_tokens`` / ``force_split``)
and adds head/tail-preserving truncation used by both Case and Skill extractors. Module-private — not
exported from ``agent_memory.__init__``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from everalgo._tokenize import _get_tokenizer, count_tokens

__all__ = ["count_tokens", "json_default", "truncate_text"]


def truncate_text(
    text: str,
    max_tokens: int,
    *,
    head_ratio: float = 0.7,
    suffix: str | None = None,
) -> str:
    r"""Truncate ``text`` to ``max_tokens`` tokens, preserving head + tail with a marker.

    Two modes:

    - ``suffix is None`` (default): keep a ``head_ratio`` slice from the start and the remainder from the end,
      joined by ``"\\n[... trimmed N tokens ...]\\n"`` (when ``head_ratio < 1.0``) or a literal ``"..."``
      append (when ``head_ratio == 1.0``). Mirrors opensource ``_truncate_text`` line 153-171.
    - ``suffix is not None``: keep only the head ``max_tokens`` tokens and append ``suffix`` (the
      ``"... [omitted]"`` shape used by opensource skill extractor :125).

    Parameters
    ----------
    text : str
        Source text; empty / non-string returns input unchanged.
    max_tokens : int
        Hard cap on output token count (not counting ``suffix`` if provided).
    head_ratio : float, optional
        Fraction of ``max_tokens`` allocated to the head when no ``suffix`` override is set. Default ``0.7``
        matches the opensource ``case`` extractor; pass ``1.0`` for head-only with ``"..."`` append.
    suffix : str or None, optional
        When provided, switches to head-only + suffix mode (skill extractor pattern).

    Returns
    -------
    str
        Truncated text. Short / empty inputs are returned unchanged.
    """
    if not text:
        return text
    tokenizer = _get_tokenizer()
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text

    if suffix is not None:
        head = tokenizer.decode(tokens[:max_tokens])
        return head.rstrip() + suffix

    head_count = int(max_tokens * head_ratio)
    tail_count = max_tokens - head_count
    head_text = tokenizer.decode(tokens[:head_count])
    if tail_count <= 0:
        return head_text.rstrip() + "..."
    tail_text = tokenizer.decode(tokens[-tail_count:])
    trimmed = len(tokens) - max_tokens
    return f"{head_text}\n[... trimmed {trimmed} tokens ...]\n{tail_text}"


def json_default(obj: Any) -> Any:
    """``json.dumps(default=...)`` fallback for non-serialisable values.

    Converts :class:`datetime.datetime` to ISO 8601 strings; falls back to :func:`str` for anything else, so
    no value can crash the LLM prompt rendering.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)
