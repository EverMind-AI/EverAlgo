"""Tests for ``everalgo.utils.tokens``."""

import pytest

from everalgo.utils import count_tokens, encode, truncate_text
from everalgo.utils.tokens import _get_tokenizer

# o200k_base special-token literals; ordinary content such as tokenizer documentation quotes them.
SPECIAL_TOKEN_LITERALS = ("<|endoftext|>", "<|endofprompt|>")


def _words(count: int) -> str:
    return " ".join(f"word_{i}" for i in range(count))


# ---------------------------------------------------------------------------
# count_tokens / encode
# ---------------------------------------------------------------------------


def test_count_tokens_of_empty_text_is_zero() -> None:
    assert count_tokens("") == 0


def test_count_tokens_matches_o200k_base() -> None:
    text = "Hello world from the EverAlgo shared tokenizer."
    assert count_tokens(text) == len(_get_tokenizer().encode(text))


def test_count_tokens_handles_cjk() -> None:
    assert count_tokens("你好世界") > 0
    assert count_tokens("こんにちは") > 0


@pytest.mark.parametrize("literal", SPECIAL_TOKEN_LITERALS)
def test_encode_treats_a_special_token_literal_as_ordinary_text(literal: str) -> None:
    assert count_tokens(literal) == len(encode(literal)) > 1


def test_raw_tiktoken_still_rejects_special_token_literals() -> None:
    """Pins why ``encode`` exists: if tiktoken changes this default, re-read the wrapper's rationale."""
    with pytest.raises(ValueError, match="disallowed special token"):
        _get_tokenizer().encode("<|endoftext|>")


# ---------------------------------------------------------------------------
# truncate_text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [None, "... [omitted]"])
def test_text_within_the_budget_comes_back_unchanged(suffix: str | None) -> None:
    text = _words(100)
    assert truncate_text(text, max_tokens=count_tokens(text), suffix=suffix) == text
    assert truncate_text("", max_tokens=10, suffix=suffix) == ""


def test_head_and_tail_are_kept_around_a_trim_marker() -> None:
    head, tail = truncate_text(_words(500), max_tokens=100, head_ratio=0.7).split("\n[... trimmed ")

    assert head.startswith("word_0")
    assert tail.endswith("word_499")
    assert tail.startswith(f"{count_tokens(_words(500)) - 100} tokens ...]")


def test_head_ratio_sets_the_share_of_the_budget_kept_as_head() -> None:
    text = _words(500)
    head_70 = truncate_text(text, max_tokens=100, head_ratio=0.7).split("\n[... trimmed")[0]
    head_50 = truncate_text(text, max_tokens=100, head_ratio=0.5).split("\n[... trimmed")[0]

    assert count_tokens(head_70) > count_tokens(head_50)


def test_a_head_ratio_of_one_keeps_the_head_and_an_ellipsis() -> None:
    result = truncate_text(_words(500), max_tokens=50, head_ratio=1.0)

    assert result.endswith("...")
    assert "[... trimmed" not in result


@pytest.mark.parametrize("suffix", ["", "... [omitted]"])
def test_a_suffix_keeps_the_head_only_and_appends_it(suffix: str) -> None:
    text = _words(500)
    result = truncate_text(text, max_tokens=30, suffix=suffix)

    assert result.startswith("word_0")
    assert result.endswith(suffix)
    assert "[... trimmed" not in result
    assert len(result) < len(text)


def test_truncate_text_handles_special_token_literals() -> None:
    text = "The tokenizer rejected the literal <|endoftext|>. " * 50

    assert len(truncate_text(text, max_tokens=32)) < len(text)
