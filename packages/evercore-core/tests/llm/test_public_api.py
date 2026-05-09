"""Tests for the evercore.llm public API surface."""


def test_top_level_exports_are_twelve_named_symbols() -> None:
    from evercore.llm import __all__

    assert sorted(__all__) == sorted(
        [
            # sub-project 2 (LLM Stack — 7)
            "LLMClient",
            "ChatMessage",
            "ChatResponse",
            "Usage",
            "LLMConfig",
            "LLMError",
            "build_client",
            # sub-project 2.5 (3-layer injection — 5)
            "LLMNotConfiguredError",
            "configure",
            "current",
            "resolve",
            "use",
        ]
    )


def test_top_level_imports_resolve() -> None:
    from evercore.llm import (
        ChatMessage,
        ChatResponse,
        LLMClient,
        LLMConfig,
        LLMError,
        Usage,
        build_client,
    )

    assert ChatMessage.__name__ == "ChatMessage"
    assert ChatResponse.__name__ == "ChatResponse"
    assert LLMClient.__name__ == "LLMClient"
    assert LLMConfig.__name__ == "LLMConfig"
    assert LLMError.__name__ == "LLMError"
    assert Usage.__name__ == "Usage"
    assert build_client.__name__ == "build_client"


def test_subproject_2_5_symbols_importable() -> None:
    """Sub-project 2.5 5 new symbols are importable from evercore.llm top level."""
    from evercore.llm import (
        LLMNotConfiguredError,
        configure,
        current,
        resolve,
        use,
    )

    assert callable(configure)
    assert callable(use)
    assert callable(current)
    assert callable(resolve)
    assert issubclass(LLMNotConfiguredError, RuntimeError)
