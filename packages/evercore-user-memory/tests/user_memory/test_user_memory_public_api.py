"""Tests for evercore.user_memory package-level public API."""

import evercore.user_memory


def test_six_symbols_exported() -> None:
    """user_memory exposes 4 user-memory Extractors + 2 boundary re-exports."""
    for name in (
        "AtomicFactExtractor",
        "ChatMemCellExtractor",
        "EpisodeExtractor",
        "ForesightExtractor",
        "ProfileExtractor",
        "WorkspaceMemCellExtractor",
    ):
        assert hasattr(evercore.user_memory, name)


def test_dunder_all_lists_six_symbols() -> None:
    """__all__ exposes exactly 6 symbols (4 Extractors + 2 boundary re-exports)."""
    assert sorted(evercore.user_memory.__all__) == sorted(
        [
            "AtomicFactExtractor",
            "ChatMemCellExtractor",
            "EpisodeExtractor",
            "ForesightExtractor",
            "ProfileExtractor",
            "WorkspaceMemCellExtractor",
        ]
    )


def test_three_stub_extractors_instantiable() -> None:
    """3 stub Extractors can be instantiated without args."""
    from evercore.user_memory import (
        AtomicFactExtractor,
        ForesightExtractor,
        ProfileExtractor,
    )

    assert AtomicFactExtractor().__class__.__name__ == "AtomicFactExtractor"
    assert ForesightExtractor().__class__.__name__ == "ForesightExtractor"
    assert ProfileExtractor().__class__.__name__ == "ProfileExtractor"
