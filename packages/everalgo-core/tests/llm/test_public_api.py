"""Tests for the everalgo.llm public API surface."""

import pytest

import everalgo.llm


def test_top_level_exports_match_all() -> None:
    assert sorted(everalgo.llm.__all__) == sorted(
        [
            # protocol: the client, its request types, and the one error the contract defines
            "LLMClient",
            "LLMMessage",
            "ContentPart",
            "ImageUrlInner",
            "ImageUrlPart",
            "TextPart",
            "image_url_part_from_bytes",
            "LLMOutputError",
            # response_format models: the base class, parsing a reply against one, and the helpers that turn one
            # into request parameters
            "StrictOutputModel",
            "parse_reply",
            "describe_errors",
            "to_openai_response_format",
            "to_anthropic_response_format",
            "to_gemini_response_format",
            "to_ollama_response_format",
        ]
    )


def test_every_name_in_all_resolves() -> None:
    for name in everalgo.llm.__all__:
        assert getattr(everalgo.llm, name) is not None, name


@pytest.mark.parametrize(
    "removed",
    [
        # renamed without an alias
        "ChatMessage",
        # the response wrapper and its parts
        "ChatResponse",
        "Usage",
        # wire-shaped response formats
        "TextFormat",
        "JsonObjectFormat",
        "JsonSchemaSpec",
        "JsonSchemaFormat",
        "ResponseFormat",
        # renamed without an alias
        "json_schema_format",
        # the reference implementation's error, which is not part of the contract
        "LLMError",
        # the reference implementation takes keyword arguments; a config object and a one-target factory added nothing
        "LLMConfig",
        "build_client",
        # lenient fence / braces extraction contradicts strict model_validate_json parsing
        "parse_llm_json_object",
        # prompt timestamp helpers belong to everalgo.utils; the two without a caller are gone
        "Lang",
        "format_atomic_fact_time",
        "format_iso_timestamp",
        "format_message_timestamp",
        "format_natural_language_time",
        # removed outright: nothing raised it
        "UnsupportedParameterError",
    ],
)
def test_removed_names_are_no_longer_exported(removed: str) -> None:
    assert not hasattr(everalgo.llm, removed)
