"""Shared tokenizer utilities using OpenAI ``o200k_base`` encoding (GPT-4o / o-series).

Module-private — internal shared implementation; not part of the public ``everalgo.*`` API.
"""

from __future__ import annotations

import tiktoken

_ENCODING_NAME = "o200k_base"
_tokenizer_cache: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    """Return the shared ``o200k_base`` encoding, initialising on first call."""
    global _tokenizer_cache
    if _tokenizer_cache is None:
        _tokenizer_cache = tiktoken.get_encoding(_ENCODING_NAME)
    return _tokenizer_cache


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` under ``o200k_base``. Empty string returns 0."""
    if not text:
        return 0
    return len(_get_tokenizer().encode(text))


def force_split(text: str, *, max_tokens: int) -> list[str]:
    """Split ``text`` into chunks of at most ``max_tokens`` tokens (no semantic awareness).

    Returns ``[]`` for empty input, ``[text]`` when the whole string already fits.

    Raises:
        ValueError: If ``max_tokens <= 0``.
    """
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if not text:
        return []
    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= max_tokens:
        return [text]
    return [tokenizer.decode(token_ids[i : i + max_tokens]) for i in range(0, len(token_ids), max_tokens)]
