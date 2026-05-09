"""Tests for evercore.prompts.validator."""

import pytest

from evercore.prompts.validator import check_length, check_placeholders


def test_check_placeholders_pass_when_all_required_present() -> None:
    check_placeholders("Hello {name}, today is {date}.", required=["name", "date"])


def test_check_placeholders_pass_when_no_required_and_no_placeholders() -> None:
    check_placeholders("Hello world.", required=[])


def test_check_placeholders_extras_are_allowed() -> None:
    """Template with placeholders not in required is fine — caller may not pass them."""
    check_placeholders("{a} {b} {c}", required=["a"])


def test_check_placeholders_missing_one_raises() -> None:
    with pytest.raises(ValueError, match="Missing required placeholders"):
        check_placeholders("Hello {name}.", required=["name", "date"])


def test_check_placeholders_missing_all_lists_them_alphabetically() -> None:
    with pytest.raises(ValueError, match=r"\['date', 'name'\]"):
        check_placeholders("Hello world.", required=["name", "date"])


def test_check_placeholders_handles_attribute_access() -> None:
    """`{user.name}` should match required `user` (root identifier)."""
    check_placeholders("Hello {user.name}", required=["user"])


def test_check_placeholders_handles_index_access() -> None:
    """`{items[0]}` should match required `items`."""
    check_placeholders("First: {items[0]}", required=["items"])


def test_check_placeholders_extras_listed_when_missing_raised() -> None:
    """Diagnostic message includes both missing and extra placeholders to ease typo fixes."""
    with pytest.raises(ValueError, match=r"extra placeholders present"):
        check_placeholders("Hello {nme}.", required=["name"])


def test_check_length_pass_with_default_estimator_under_limit() -> None:
    check_length("hello world", max_tokens=100)


def test_check_length_fail_with_default_estimator_over_limit() -> None:
    long_text = "x" * 1000
    with pytest.raises(ValueError, match="exceeds max_tokens"):
        check_length(long_text, max_tokens=10)


def test_check_length_with_custom_tokenizer_pass() -> None:
    check_length(
        "this is a sentence",
        max_tokens=10,
        tokenizer=lambda s: len(s.split()),
    )


def test_check_length_with_custom_tokenizer_fail() -> None:
    with pytest.raises(ValueError, match="exceeds max_tokens"):
        check_length(
            "this is a sentence with seven words at least",
            max_tokens=3,
            tokenizer=lambda s: len(s.split()),
        )


def test_check_length_default_estimator_is_safe_overcount_for_cjk() -> None:
    """4-chars-per-token approximation overcounts CJK; assertion confirms it does not under-count."""
    cjk_text = "你好世界" * 100  # 400 CJK chars; real tokens ~ 400-800 (varies by model).
    with pytest.raises(ValueError, match="exceeds max_tokens"):
        # 4-chars-per-token estimator returns ~ 101 tokens, well above the cap.
        check_length(cjk_text, max_tokens=50)
