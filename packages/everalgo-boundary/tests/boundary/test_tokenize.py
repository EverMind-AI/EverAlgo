"""Tests for everalgo._tokenize — count_tokens and force_split.

Both helpers are internal (module-private ``_tokenize``), but they underpin
the force-split loop in the boundary package so coverage here prevents regressions.

Design note: these are pure-compute functions (no I/O, no LLM calls). They use
the real ``o200k_base`` tiktoken encoding so results reflect actual token counts.
"""

from __future__ import annotations

import pytest

from everalgo._tokenize import count_tokens, force_split

# ===========================================================================
# count_tokens — basic contract
# ===========================================================================


def test_empty_string_returns_zero() -> None:
    assert count_tokens("") == 0


def test_single_ascii_word_returns_positive_count() -> None:
    assert count_tokens("hello") > 0


def test_longer_text_returns_larger_count_than_shorter() -> None:
    short = "hello"
    long = "hello " * 20
    assert count_tokens(long) > count_tokens(short)


def test_whitespace_only_returns_positive_count() -> None:
    """Whitespace alone is still tokenised."""
    assert count_tokens("   ") > 0


# ===========================================================================
# count_tokens — known token counts
# ===========================================================================


def test_known_short_sentence_token_count_in_expected_range() -> None:
    """'Hello, world!' is a well-known short string; sanity-check range [3, 6]."""
    count = count_tokens("Hello, world!")
    assert 3 <= count <= 6


def test_repeated_short_words_count_scales_roughly_linearly() -> None:
    """Doubling the repetitions should roughly double the token count (within 10%)."""
    base = count_tokens("word " * 10)
    doubled = count_tokens("word " * 20)
    # Allow 10% slack for tokenizer merging at boundaries.
    assert doubled >= int(base * 1.8)


# ===========================================================================
# count_tokens — multilingual content
# ===========================================================================


def test_cjk_characters_are_tokenised() -> None:
    """Chinese text tokenises to a positive count (tiktoken CJK support)."""
    assert count_tokens("你好，世界") > 0  # noqa: RUF001


def test_mixed_ascii_and_cjk_tokenises() -> None:
    text = "Hello 你好 world 世界"
    assert count_tokens(text) > 0


def test_emoji_only_string_tokenises() -> None:
    assert count_tokens("😀🎉🔥") > 0


def test_unicode_symbols_tokenise() -> None:
    assert count_tokens("α β γ δ ε") > 0  # noqa: RUF001


# ===========================================================================
# count_tokens — edge cases
# ===========================================================================


def test_very_long_string_does_not_raise() -> None:
    long_text = "a" * 100_000
    count = count_tokens(long_text)
    assert count > 0


def test_newlines_and_tabs_tokenise() -> None:
    assert count_tokens("\n\t\r\n") > 0


def test_null_bytes_tokenise_without_raising() -> None:
    assert count_tokens("\x00\x01\x02") >= 0


def test_count_is_int_type() -> None:
    result = count_tokens("test string")
    assert isinstance(result, int)


# ===========================================================================
# force_split — basic contract
# ===========================================================================


def test_empty_string_returns_empty_list() -> None:
    assert force_split("", max_tokens=10) == []


def test_short_text_within_limit_returns_single_chunk() -> None:
    text = "short text"
    chunks = force_split(text, max_tokens=1000)
    assert chunks == [text]


def test_zero_max_tokens_raises_value_error() -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        force_split("any text", max_tokens=0)


def test_negative_max_tokens_raises_value_error() -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        force_split("any text", max_tokens=-1)


def test_large_text_is_split_into_multiple_chunks() -> None:
    """Text larger than max_tokens must produce more than one chunk."""
    long_text = "word " * 100  # ~100+ tokens
    chunks = force_split(long_text, max_tokens=10)
    assert len(chunks) > 1


def test_all_chunks_fit_within_max_tokens() -> None:
    """Every chunk in the output must have at most max_tokens tokens."""
    long_text = "the quick brown fox jumps over the lazy dog " * 20
    max_tokens = 15
    chunks = force_split(long_text, max_tokens=max_tokens)
    for chunk in chunks:
        assert count_tokens(chunk) <= max_tokens


def test_reassembling_chunks_covers_original_tokens() -> None:
    """Concatenating all chunks must contain the same token count as the original text."""
    long_text = "hello world " * 50
    max_tokens = 20
    chunks = force_split(long_text, max_tokens=max_tokens)
    total = sum(count_tokens(c) for c in chunks)
    assert total == count_tokens(long_text)


def test_single_max_token_splits_to_individual_tokens() -> None:
    """max_tokens=1 produces one chunk per token."""
    text = "hello world"
    chunks = force_split(text, max_tokens=1)
    assert len(chunks) == count_tokens(text)
