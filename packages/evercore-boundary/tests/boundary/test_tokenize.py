"""Tests for evercore.boundary._tokenize — count_tokens helper."""

from evercore.boundary._tokenize import count_tokens


def test_count_tokens_empty_string_is_zero() -> None:
    """Empty input yields zero tokens."""
    assert count_tokens("") == 0


def test_count_tokens_short_text_proportional() -> None:
    """40 chars yields 10 tokens under the 4-char heuristic."""
    assert count_tokens("a" * 40) == 10


def test_count_tokens_returns_non_negative() -> None:
    """Any non-empty string yields a non-negative count."""
    samples = ["x", "hello world", "你好", "a" * 1000]
    for text in samples:
        assert count_tokens(text) >= 0
