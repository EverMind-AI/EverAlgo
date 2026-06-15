"""Tests for everalgo.user_memory package-level public API."""

import everalgo.user_memory
from everalgo.testing.fake_llm import FakeLLMClient


def test_seven_symbols_exported() -> None:
    """user_memory exposes 4 Extractors + EpisodeReflector + BoundaryDetector + DetectionResult re-export."""
    for name in (
        "AtomicFactExtractor",
        "BoundaryDetector",
        "DetectionResult",
        "EpisodeExtractor",
        "EpisodeReflector",
        "ForesightExtractor",
        "ProfileExtractor",
    ):
        assert hasattr(everalgo.user_memory, name)


def test_dunder_all_lists_seven_symbols() -> None:
    """__all__ exposes exactly 7 symbols (4 Extractors + EpisodeReflector + BoundaryDetector + DetectionResult)."""
    assert sorted(everalgo.user_memory.__all__) == sorted(
        [
            "AtomicFactExtractor",
            "BoundaryDetector",
            "DetectionResult",
            "EpisodeExtractor",
            "EpisodeReflector",
            "ForesightExtractor",
            "ProfileExtractor",
        ]
    )


def test_workspace_extractor_is_not_re_exported() -> None:
    """The unimplemented boundary stub must not leak into user_memory's public surface."""
    assert "WorkspaceMemCellExtractor" not in everalgo.user_memory.__all__
    assert not hasattr(everalgo.user_memory, "WorkspaceMemCellExtractor")


def test_user_memory_extractors_instantiable_with_llm() -> None:
    """All 4 Extractors + EpisodeReflector accept a required llm= keyword argument at construction."""
    from everalgo.user_memory import (
        AtomicFactExtractor,
        EpisodeExtractor,
        EpisodeReflector,
        ForesightExtractor,
        ProfileExtractor,
    )

    fake = FakeLLMClient(responses=[])
    assert AtomicFactExtractor(llm=fake).__class__.__name__ == "AtomicFactExtractor"
    assert EpisodeExtractor(llm=fake).__class__.__name__ == "EpisodeExtractor"
    assert EpisodeReflector(llm=fake).__class__.__name__ == "EpisodeReflector"
    assert ForesightExtractor(llm=fake).__class__.__name__ == "ForesightExtractor"
    assert ProfileExtractor(llm=fake).__class__.__name__ == "ProfileExtractor"
