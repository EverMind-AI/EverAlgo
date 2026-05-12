"""Token counting + force-split helpers for boundary extractors.

Token counting uses OpenAI's ``o200k_base`` encoding via :mod:`tiktoken` — the same encoding the GPT-4o /
GPT-5 / o-series tokenizers use, and what most production callers measure their context budgets against.
Earlier versions of this module shipped a char/4 heuristic; the upgrade is a SemVer minor bump (signature
preserved).

NOT exposed in ``__all__`` — module-private utilities for boundary algorithms.
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
    """Count tokens in ``text`` under the ``o200k_base`` encoding.

    Parameters
    ----------
    text : str
        Input string. Empty string returns 0.

    Returns
    -------
    int
        Token count (always >= 0). Suitable for context-budget gating; matches what OpenAI bills against on
        chat-completions calls using GPT-4o / GPT-5 / o-series models.
    """
    if not text:
        return 0
    return len(_get_tokenizer().encode(text))


def force_split(text: str, *, max_tokens: int) -> list[str]:
    """Force-split ``text`` into chunks each containing at most ``max_tokens`` tokens.

    No semantic awareness — chunks break wherever the tokenizer's encoding boundaries permit. Intended as a
    last-resort guardrail for caller-side prompt fitting; semantic boundaries belong to
    :class:`ChatMemCellExtractor` / future ``WorkspaceMemCellExtractor`` / ``AgentMemCellExtractor`` instead.

    Parameters
    ----------
    text : str
        Input string. Empty input returns ``[]``.
    max_tokens : int
        Maximum tokens per output chunk. Must be positive.

    Returns
    -------
    list[str]
        Each element has ``count_tokens(element) <= max_tokens``. Returns ``[]`` for empty input and
        ``[text]`` when the whole string already fits.

    Raises
    ------
    ValueError
        If ``max_tokens <= 0``.
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
