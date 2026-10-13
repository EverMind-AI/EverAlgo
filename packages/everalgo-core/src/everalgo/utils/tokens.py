"""Token counting and truncation under OpenAI's ``o200k_base`` encoding (GPT-4o and the o-series).

The counts are a budget estimate for any model: other tokenizers split the same text differently.
"""

from __future__ import annotations

__all__ = ["count_tokens", "encode", "truncate_text"]

import tiktoken

_ENCODING_NAME = "o200k_base"
_tokenizer_cache: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    """Return the shared ``o200k_base`` encoding, initialising on first call."""
    global _tokenizer_cache
    if _tokenizer_cache is None:
        _tokenizer_cache = tiktoken.get_encoding(_ENCODING_NAME)
    return _tokenizer_cache


def encode(text: str) -> list[int]:
    """Encode ``text``, treating special-token literals such as ``<|endoftext|>`` as ordinary text.

    tiktoken's default raises on those literals, which guards a caller assembling a prompt but leaves any text that
    merely quotes one unprocessable for good; this module only measures and slices text. Every tiktoken encode in a
    published distribution goes through here (``tests/test_tokenizer_conventions.py``).
    """
    return _get_tokenizer().encode(text, disallowed_special=())


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` under ``o200k_base``. Empty string returns 0."""
    if not text:
        return 0
    return len(encode(text))


def truncate_text(text: str, max_tokens: int, *, head_ratio: float = 0.7, suffix: str | None = None) -> str:
    r"""Cut ``text`` down to ``max_tokens`` tokens; text that already fits comes back unchanged.

    Without ``suffix`` the head (``head_ratio`` of the budget) and the tail are kept, joined by
    ``"\n[... trimmed N tokens ...]\n"``, or followed by ``"..."`` when ``head_ratio`` leaves no tail. With
    ``suffix`` only the head is kept and ``suffix`` is appended.
    """
    tokens = encode(text)
    if len(tokens) <= max_tokens:
        return text
    tokenizer = _get_tokenizer()
    if suffix is not None:
        return tokenizer.decode(tokens[:max_tokens]).rstrip() + suffix
    head_count = int(max_tokens * head_ratio)
    tail_count = max_tokens - head_count
    head_text = tokenizer.decode(tokens[:head_count])
    if tail_count <= 0:
        return head_text.rstrip() + "..."
    tail_text = tokenizer.decode(tokens[-tail_count:])
    return f"{head_text}\n[... trimmed {len(tokens) - max_tokens} tokens ...]\n{tail_text}"
