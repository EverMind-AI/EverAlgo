"""Token-aware text utilities for agent_memory extractors (head/tail truncation, json_default)."""

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

    When ``suffix`` is ``None``: keep ``head_ratio`` of tokens from the start and the remainder from
    the end, joined by ``"\\n[... trimmed N tokens ...]\\n"`` (or ``"..."`` when ``head_ratio == 1.0``).
    When ``suffix`` is set: keep the head only and append ``suffix``.
    Short / empty inputs are returned unchanged.
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
    """``json.dumps(default=...)`` fallback: datetime → ISO string, anything else → ``str``."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)
