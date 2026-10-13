"""Tests for everalgo.user_memory package-level public API."""

import ast
from pathlib import Path

import everalgo.user_memory
from everalgo.testing.fake_llm import FakeLLMClient


def test_public_symbols_exported() -> None:
    """4 Extractors + EpisodeReflector + EpisodeSummarizer + BoundaryDetector + OutputLanguage + DetectionResult."""
    for name in (
        "AtomicFactExtractor",
        "BoundaryDetector",
        "DetectionResult",
        "EpisodeExtractor",
        "EpisodeReflector",
        "EpisodeSummarizer",
        "ForesightExtractor",
        "OutputLanguage",
        "ProfileExtractor",
    ):
        assert hasattr(everalgo.user_memory, name)


def test_dunder_all_lists_exactly_the_public_symbols() -> None:
    """__all__ lists exactly the symbols the test above checks, no more."""
    assert sorted(everalgo.user_memory.__all__) == sorted(
        [
            "AtomicFactExtractor",
            "BoundaryDetector",
            "DetectionResult",
            "EpisodeExtractor",
            "EpisodeReflector",
            "EpisodeSummarizer",
            "ForesightExtractor",
            "OutputLanguage",
            "ProfileExtractor",
        ]
    )


def test_workspace_extractor_is_not_re_exported() -> None:
    """The unimplemented boundary stub must not leak into user_memory's public surface."""
    assert "WorkspaceMemCellExtractor" not in everalgo.user_memory.__all__
    assert not hasattr(everalgo.user_memory, "WorkspaceMemCellExtractor")


def test_user_memory_extractors_instantiable_with_llm() -> None:
    """Every LLM-backed operator class accepts a required llm= keyword argument at construction."""
    from everalgo.user_memory import (
        AtomicFactExtractor,
        EpisodeExtractor,
        EpisodeReflector,
        EpisodeSummarizer,
        ForesightExtractor,
        ProfileExtractor,
    )

    fake = FakeLLMClient(responses=[])
    assert AtomicFactExtractor(llm=fake).__class__.__name__ == "AtomicFactExtractor"
    assert EpisodeExtractor(llm=fake).__class__.__name__ == "EpisodeExtractor"
    assert EpisodeReflector(llm=fake).__class__.__name__ == "EpisodeReflector"
    assert EpisodeSummarizer(llm=fake).__class__.__name__ == "EpisodeSummarizer"
    assert ForesightExtractor(llm=fake).__class__.__name__ == "ForesightExtractor"
    assert ProfileExtractor(llm=fake).__class__.__name__ == "ProfileExtractor"


def test_boundary_detector_is_the_boundary_package_class() -> None:
    """The product package re-exports ``everalgo.boundary.BoundaryDetector`` itself, not a wrapper or subclass."""
    import everalgo.boundary

    assert everalgo.user_memory.BoundaryDetector is everalgo.boundary.BoundaryDetector


def test_boundary_module_holds_nothing_but_the_re_export() -> None:
    """``boundary.py`` is a docstring, ``__all__`` and one import: no logic of its own to drift from the class."""
    tree = ast.parse((Path(everalgo.user_memory.__file__).parent / "boundary.py").read_text(encoding="utf-8"))
    statements = [ast.unparse(node) for node in tree.body[1:]]

    assert statements == [
        "from __future__ import annotations",
        "__all__ = ['BoundaryDetector']",
        "from everalgo.boundary import BoundaryDetector",
    ]


def test_agent_boundary_detector_is_gone() -> None:
    assert "AgentBoundaryDetector" not in everalgo.user_memory.__all__
    assert not hasattr(everalgo.user_memory, "AgentBoundaryDetector")
