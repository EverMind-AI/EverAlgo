"""Tests for everalgo.llm.errors.LLMError."""

import pytest

from everalgo.llm.errors import LLMError, LLMNotConfiguredError


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


# ---- LLMNotConfiguredError (sub-project 2.5, Task 1) ----------------------


def test_llm_not_configured_error_inherits_runtime_error_not_llm_error() -> None:
    """LLMNotConfiguredError is a misuse error (RuntimeError family).

    Not an SDK call error (LLMError family). See spec §5.3.
    """
    err = LLMNotConfiguredError("test message")
    assert isinstance(err, RuntimeError)
    assert not isinstance(err, LLMError)
