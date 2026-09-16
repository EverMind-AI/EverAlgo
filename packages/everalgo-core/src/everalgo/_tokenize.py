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


def encode(text: str) -> list[int]:
    """Encode ``text``, treating tiktoken special tokens as ordinary text.

    ``Encoding.encode`` defaults to ``disallowed_special="all"``, so it raises
    ``ValueError`` as soon as the input contains a literal such as ``<|endoftext|>`` or
    ``<|fim_prefix|>``. That default exists to protect a caller who is ASSEMBLING A PROMPT
    from letting a user smuggle a control token into a model's input. Nothing in this
    library assembles a prompt here — this module only measures and slices text — so the
    default turns ordinary content into a crash instead: any conversation that merely
    *mentions* those literals (a code-completion discussion, a tokenizer bug report) is
    unprocessable, and the failure is permanent because the same text is retried forever.

    ``disallowed_special=()`` encodes the literal as the several ordinary tokens it is
    made of, rather than as the one control token it would be inside a real prompt. For
    measuring and slicing that is the honest reading, and it is the only one that lets the
    text through.

    Every tiktoken ``encode`` in the workspace must go through here — see
    ``tests/test_tokenizer_conventions.py``, which fails the build on a direct call.
    """
    return _get_tokenizer().encode(text, disallowed_special=())


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` under ``o200k_base``. Empty string returns 0."""
    if not text:
        return 0
    return len(encode(text))


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
    token_ids = encode(text)
    if len(token_ids) <= max_tokens:
        return [text]
    return [tokenizer.decode(token_ids[i : i + max_tokens]) for i in range(0, len(token_ids), max_tokens)]
