"""Tests for everalgo.user_memory package-level public API."""

import everalgo.user_memory


def test_seven_symbols_exported() -> None:
    """user_memory exposes 4 user-memory Extractors + 2 boundary re-exports + DetectionOutput."""
    for name in (
        "AtomicFactExtractor",
        "ChatMemCellExtractor",
        "DetectionOutput",
        "EpisodeExtractor",
        "ForesightExtractor",
        "ProfileExtractor",
        "WorkspaceMemCellExtractor",
    ):
        assert hasattr(everalgo.user_memory, name)


def test_dunder_all_lists_seven_symbols() -> None:
    """__all__ exposes exactly 7 symbols (4 Extractors + 2 boundary re-exports + DetectionOutput)."""
    assert sorted(everalgo.user_memory.__all__) == sorted(
        [
            "AtomicFactExtractor",
            "ChatMemCellExtractor",
            "DetectionOutput",
            "EpisodeExtractor",
            "ForesightExtractor",
            "ProfileExtractor",
            "WorkspaceMemCellExtractor",
        ]
    )


def test_user_memory_extractors_instantiable() -> None:
    """All 4 user-memory Extractors can be instantiated without args (stateless)."""
    from everalgo.user_memory import (
        AtomicFactExtractor,
        EpisodeExtractor,
        ForesightExtractor,
        ProfileExtractor,
    )

    assert AtomicFactExtractor().__class__.__name__ == "AtomicFactExtractor"
    assert EpisodeExtractor().__class__.__name__ == "EpisodeExtractor"
    assert ForesightExtractor().__class__.__name__ == "ForesightExtractor"
    assert ProfileExtractor().__class__.__name__ == "ProfileExtractor"
