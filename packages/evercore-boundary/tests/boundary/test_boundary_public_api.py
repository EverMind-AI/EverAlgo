"""Tests for evercore.boundary package-level public API."""

import evercore.boundary


def test_three_memcell_extractors_exported() -> None:
    """Boundary exposes 3 MemCell extractors at top level."""
    assert hasattr(evercore.boundary, "ChatMemCellExtractor")
    assert hasattr(evercore.boundary, "WorkspaceMemCellExtractor")
    assert hasattr(evercore.boundary, "AgentMemCellExtractor")


def test_dunder_all_lists_three_extractors() -> None:
    """__all__ exposes exactly 3 extractor symbols."""
    assert sorted(evercore.boundary.__all__) == sorted(
        ["ChatMemCellExtractor", "WorkspaceMemCellExtractor", "AgentMemCellExtractor"]
    )
