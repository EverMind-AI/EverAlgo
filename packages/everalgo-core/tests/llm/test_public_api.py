"""Tests for the everalgo.llm public API surface."""


def test_top_level_exports_are_eleven_named_symbols() -> None:
    from everalgo.llm import __all__

    assert sorted(__all__) == sorted(
        [
            "LLMClient",
            "ChatMessage",
            "ChatResponse",
            "Usage",
            "LLMConfig",
            "LLMError",
            "Lang",
            "build_client",
            "format_message_timestamp",
            "format_natural_language_time",
            "parse_llm_json_object",
        ]
    )


def test_top_level_imports_resolve() -> None:
    from everalgo.llm import (
        ChatMessage,
        ChatResponse,
        Lang,
        LLMClient,
        LLMConfig,
        LLMError,
        Usage,
        build_client,
        format_message_timestamp,
        format_natural_language_time,
        parse_llm_json_object,
    )

    assert ChatMessage.__name__ == "ChatMessage"
    assert ChatResponse.__name__ == "ChatResponse"
    assert LLMClient.__name__ == "LLMClient"
    assert LLMConfig.__name__ == "LLMConfig"
    assert LLMError.__name__ == "LLMError"
    assert Usage.__name__ == "Usage"
    assert build_client.__name__ == "build_client"
    assert format_message_timestamp.__name__ == "format_message_timestamp"
    assert format_natural_language_time.__name__ == "format_natural_language_time"
    assert parse_llm_json_object.__name__ == "parse_llm_json_object"
    # Lang is a runtime-accessible Literal alias — verify it is importable
    assert Lang is not None
