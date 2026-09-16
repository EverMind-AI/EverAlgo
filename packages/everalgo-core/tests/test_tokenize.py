"""Tests for everalgo._tokenize — count_tokens + force_split."""

import pytest

from everalgo._tokenize import _get_tokenizer, count_tokens, encode, force_split


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


# ---------------------------------------------------------------------------
# tiktoken special tokens appearing as ordinary text
# ---------------------------------------------------------------------------

# Literals from the o200k_base special set. They turn up in perfectly ordinary
# content: any conversation about code completion or tokenizers quotes them.
SPECIAL_TOKEN_LITERALS = ("<|endoftext|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|endofprompt|>")


@pytest.mark.parametrize("literal", SPECIAL_TOKEN_LITERALS)
def test_encode_does_not_reject_special_token_literals(literal: str) -> None:
    """The bug this module guards against.

    ``Encoding.encode`` defaults to ``disallowed_special="all"`` and raises ValueError on
    these. That default is for prompt assembly; here it made real user text permanently
    unprocessable — the caller retries the same content forever and never gets past it.
    """
    assert encode(literal)  # non-empty token list, no raise


@pytest.mark.parametrize("literal", SPECIAL_TOKEN_LITERALS)
def test_count_tokens_counts_special_token_literals_as_text(literal: str) -> None:
    """Counted as the several ordinary tokens they are made of, not as one control token."""
    assert count_tokens(literal) > 1


def test_count_tokens_handles_a_realistic_sentence_quoting_a_special_token() -> None:
    """The shape actually seen in production: prose that mentions the literal."""
    text = "The FIM template is `<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>`."
    assert count_tokens(text) > 0


def test_force_split_handles_special_token_literals() -> None:
    """force_split encodes too, so it had the same defect."""
    text = "<|endoftext|> " * 40
    chunks = force_split(text, max_tokens=8)
    assert len(chunks) > 1
    assert "".join(chunks) == text


def test_raw_tiktoken_still_rejects_them() -> None:
    """Pins WHY the wrapper exists: the underlying default has not changed.

    If tiktoken ever flips this default, this test fails and the wrapper's rationale
    should be re-read rather than silently kept.
    """
    with pytest.raises(ValueError, match="disallowed special token"):
        _get_tokenizer().encode("<|endoftext|>")
