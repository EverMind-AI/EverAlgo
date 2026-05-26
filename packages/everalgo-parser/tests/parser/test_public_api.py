"""Stub existence tests for everalgo.parser."""

import inspect


def test_top_level_routing_aparse_parse() -> None:
    from everalgo.parser import aparse, parse

    assert inspect.iscoroutinefunction(aparse)
    assert callable(parse)


def test_five_submodules_have_dual_interface() -> None:
    from everalgo.parser import audio, document, image, url, video

    for mod in (image, audio, document, video, url):
        assert inspect.iscoroutinefunction(mod.aparse)
        assert callable(mod.parse)


def test_dunder_all_lists_public_symbols() -> None:
    """Re-exports cover the 5 submodules + dispatch fns + 3 facade types."""
    from everalgo.parser import __all__

    assert sorted(__all__) == sorted(
        [
            "Modality",
            "ParsedContent",
            "RawFile",
            "aparse",
            "audio",
            "document",
            "image",
            "parse",
            "url",
            "video",
        ]
    )
