"""Tests for the contract's error (everalgo.llm.errors) and the reference implementation's (openai_compat)."""

import everalgo.llm.errors
from everalgo.llm import LLMOutputError
from everalgo.llm.openai_compat import LLMError


def test_llm_output_error_is_a_plain_exception() -> None:
    assert issubclass(LLMOutputError, Exception)


def test_llm_output_error_is_unrelated_to_the_reference_implementation_error() -> None:
    """A host must be able to tell "the model's reply was unusable" from "the call failed" by type alone."""
    assert not issubclass(LLMOutputError, LLMError)
    assert not issubclass(LLMError, LLMOutputError)


def test_contract_errors_module_defines_only_llm_output_error() -> None:
    assert everalgo.llm.errors.__all__ == ["LLMOutputError"]
    assert not hasattr(everalgo.llm.errors, "LLMError")
