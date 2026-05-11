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


def test_dunder_all_lists_seven_symbols() -> None:
    from everalgo.parser import __all__

    assert sorted(__all__) == sorted(
        [
            "audio",
            "aparse",
            "document",
            "image",
            "parse",
            "url",
            "video",
        ]
    )
