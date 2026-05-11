"""Tests for everalgo.boundary package-level public API."""

import everalgo.boundary


def test_three_memcell_extractors_exported() -> None:
    """Boundary exposes 3 MemCell extractors at top level."""
    assert hasattr(everalgo.boundary, "ChatMemCellExtractor")
    assert hasattr(everalgo.boundary, "WorkspaceMemCellExtractor")
    assert hasattr(everalgo.boundary, "AgentMemCellExtractor")


def test_dunder_all_lists_three_extractors() -> None:
    """__all__ exposes exactly 3 extractor symbols."""
    assert sorted(everalgo.boundary.__all__) == sorted(
        ["ChatMemCellExtractor", "WorkspaceMemCellExtractor", "AgentMemCellExtractor"]
    )
