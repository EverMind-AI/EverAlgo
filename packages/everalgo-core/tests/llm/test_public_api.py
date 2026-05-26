"""Tests for the everalgo.llm public API surface."""


def test_top_level_exports_match_all() -> None:
    from everalgo.llm import __all__

    assert sorted(__all__) == sorted(
        [
            # core wire types + protocol + factory + errors
            "LLMClient",
            "ChatMessage",
            "ChatResponse",
            "Usage",
            "LLMConfig",
            "LLMError",
            "Lang",
            "build_client",
            "format_atomic_fact_time",
            "format_iso_timestamp",
            "format_message_timestamp",
            "format_natural_language_time",
            "parse_llm_json_object",
            # multimodal content parts (parser-migration)
            "ContentPart",
            "ImageUrlInner",
            "ImageUrlPart",
            "TextPart",
            "image_url_part_from_bytes",
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
        format_atomic_fact_time,
        format_iso_timestamp,
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
    assert format_atomic_fact_time.__name__ == "format_atomic_fact_time"
    assert format_iso_timestamp.__name__ == "format_iso_timestamp"
    assert format_message_timestamp.__name__ == "format_message_timestamp"
    assert format_natural_language_time.__name__ == "format_natural_language_time"
    assert parse_llm_json_object.__name__ == "parse_llm_json_object"
    # Lang is a runtime-accessible Literal alias — verify it is importable
    assert Lang is not None


def test_multimodal_content_part_symbols_importable() -> None:
    """Multimodal content part types added for the parser migration."""
    from everalgo.llm import (
        ContentPart,
        ImageUrlInner,
        ImageUrlPart,
        TextPart,
        image_url_part_from_bytes,
    )

    assert TextPart.__name__ == "TextPart"
    assert ImageUrlInner.__name__ == "ImageUrlInner"
    assert ImageUrlPart.__name__ == "ImageUrlPart"
    # ContentPart is an Annotated[Union], not a class — just verify it's importable.
    assert ContentPart is not None
    assert callable(image_url_part_from_bytes)
