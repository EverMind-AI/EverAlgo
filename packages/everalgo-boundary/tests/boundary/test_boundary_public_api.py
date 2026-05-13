"""Tests for everalgo.boundary package-level public API."""

import everalgo.boundary


def test_three_memcell_extractors_exported() -> None:
    """Boundary exposes 3 MemCell extractors at top level."""
    assert hasattr(everalgo.boundary, "ChatMemCellExtractor")
    assert hasattr(everalgo.boundary, "WorkspaceMemCellExtractor")
    assert hasattr(everalgo.boundary, "AgentMemCellExtractor")


def test_detection_output_exported() -> None:
    """``DetectionOutput`` is exposed at top level as the return type of ChatMemCellExtractor.adetect."""
    assert hasattr(everalgo.boundary, "DetectionOutput")


def test_dunder_all_lists_full_surface() -> None:
    """__all__ exposes 3 extractors + DetectionOutput."""
    assert sorted(everalgo.boundary.__all__) == sorted(
        [
            "AgentMemCellExtractor",
            "ChatMemCellExtractor",
            "DetectionOutput",
            "WorkspaceMemCellExtractor",
        ]
    )
