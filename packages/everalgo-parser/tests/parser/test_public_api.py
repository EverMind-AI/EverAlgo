"""Stub existence tests for everalgo.parser."""

import inspect


def test_top_level_routing_aparse_parse() -> None:
    from everalgo.parser import aparse, parse

    assert inspect.iscoroutinefunction(aparse)
    assert callable(parse)


def test_dunder_all_lists_public_symbols() -> None:
    """``__all__`` lists only facade-level symbols; submodules are internal."""
    from everalgo.parser import __all__

    assert sorted(__all__) == sorted(
        [
            "Modality",
            "ParsedContent",
            "RawFile",
            "aparse",
            "parse",
        ]
    )
