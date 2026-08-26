"""Tests for everalgo.user_memory package-level public API."""

import everalgo.user_memory
from everalgo.testing.fake_llm import FakeLLMClient


def test_public_symbols_exported() -> None:
    """5 Extractors + EpisodeReflector + BoundaryDetector + OutputLanguage + DetectionResult re-export."""
    for name in (
        "AtomicFactExtractor",
        "BoundaryDetector",
        "DecisionExtractor",
        "DetectionResult",
        "EpisodeExtractor",
        "EpisodeReflector",
        "ForesightExtractor",
        "OutputLanguage",
        "ProfileExtractor",
    ):
        assert hasattr(everalgo.user_memory, name)


def test_dunder_all_lists_exactly_the_public_symbols() -> None:
    """__all__ exposes 5 Extractors + EpisodeReflector + BoundaryDetector + OutputLanguage + DetectionResult."""
    assert sorted(everalgo.user_memory.__all__) == sorted(
        [
            "AtomicFactExtractor",
            "BoundaryDetector",
            "DecisionExtractor",
            "DetectionResult",
            "EpisodeExtractor",
            "EpisodeReflector",
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
    """All 5 Extractors + EpisodeReflector accept a required llm= keyword argument at construction."""
    from everalgo.user_memory import (
        AtomicFactExtractor,
        DecisionExtractor,
        EpisodeExtractor,
        EpisodeReflector,
        ForesightExtractor,
        ProfileExtractor,
    )

    fake = FakeLLMClient(responses=[])
    assert AtomicFactExtractor(llm=fake).__class__.__name__ == "AtomicFactExtractor"
    assert DecisionExtractor(llm=fake).__class__.__name__ == "DecisionExtractor"
    assert EpisodeExtractor(llm=fake).__class__.__name__ == "EpisodeExtractor"
    assert EpisodeReflector(llm=fake).__class__.__name__ == "EpisodeReflector"
    assert ForesightExtractor(llm=fake).__class__.__name__ == "ForesightExtractor"
    assert ProfileExtractor(llm=fake).__class__.__name__ == "ProfileExtractor"
