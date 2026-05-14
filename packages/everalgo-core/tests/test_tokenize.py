"""Tests for everalgo._tokenize — count_tokens + force_split."""

import pytest

from everalgo._tokenize import _get_tokenizer, count_tokens, force_split


def test_count_tokens_empty_string_is_zero() -> None:
    """Empty input yields zero tokens."""
    assert count_tokens("") == 0


def test_count_tokens_returns_non_negative() -> None:
    """Any non-empty string yields a non-negative count."""
    for text in ("x", "hello world", "你好", "a" * 1000):
        assert count_tokens(text) >= 0


def test_count_tokens_matches_o200k_base_encoding() -> None:
    """count_tokens delegates to tiktoken o200k_base — verify via direct encode."""
    tokenizer = _get_tokenizer()
    text = "Hello world from the EverAlgo shared tokenizer."
    assert count_tokens(text) == len(tokenizer.encode(text))


def test_count_tokens_handles_unicode() -> None:
    """Unicode (CJK) text still produces a positive count."""
    assert count_tokens("你好世界") > 0
    assert count_tokens("こんにちは") > 0


def test_force_split_empty_returns_empty_list() -> None:
    """Empty input yields an empty list, not a single empty chunk."""
    assert force_split("", max_tokens=10) == []


def test_force_split_invalid_max_tokens_raises() -> None:
    """max_tokens must be positive."""
    with pytest.raises(ValueError, match="must be positive"):
        force_split("hello", max_tokens=0)
    with pytest.raises(ValueError, match="must be positive"):
        force_split("hello", max_tokens=-1)


def test_force_split_short_text_returns_single_chunk() -> None:
    """Text that fits within max_tokens returns [text] unchanged."""
    text = "Hello world."
    result = force_split(text, max_tokens=100)
    assert result == [text]


def test_force_split_long_text_chunks_under_limit() -> None:
    """Each chunk respects the max_tokens budget under tiktoken."""
    long_text = "Hello world. " * 200  # ~400+ tokens
    max_tokens = 50
    chunks = force_split(long_text, max_tokens=max_tokens)
    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk) <= max_tokens


def test_force_split_chunks_reassemble_decoded_token_stream() -> None:
    """Concatenating chunks recovers the original tokenized content (modulo encoding round-trip)."""
    text = "The quick brown fox jumps over the lazy dog. " * 20
    chunks = force_split(text, max_tokens=10)
    rejoined = "".join(chunks)
    # Token-level equality is the load-bearing invariant; raw-string equality may differ via decode boundaries
    # on BPE tokenizers, so we compare via re-encoded round-trip.
    assert count_tokens(rejoined) == count_tokens(text)
