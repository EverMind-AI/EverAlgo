"""Tests for everalgo.llm.errors.LLMError."""

import pytest

from everalgo.llm.errors import LLMError


def test_llm_error_is_exception_subclass() -> None:
    assert issubclass(LLMError, Exception)


def test_llm_error_with_message() -> None:
    err = LLMError("rate limit exceeded")
    assert str(err) == "rate limit exceeded"


def test_llm_error_chains_cause_via_pep_3134() -> None:
    """The provider layer should attach SDK-native exception via ``raise X from y``.

    Callers can then inspect ``e.__cause__`` to reach the original SDK exception class.
    """
    sdk_native = ValueError("upstream failure")

    with pytest.raises(LLMError) as caught:
        try:
            raise sdk_native
        except ValueError as exc:
            raise LLMError("wrapped failure") from exc

    assert caught.value.__cause__ is sdk_native
