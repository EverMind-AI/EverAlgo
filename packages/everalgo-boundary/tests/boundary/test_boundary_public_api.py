"""Tests for everalgo.boundary package-level public API."""

import everalgo.boundary


def test_extractors_and_detection_output_exported() -> None:
    """Boundary exposes 3 MemCell extractors + DetectionOutput at top level."""
    assert hasattr(everalgo.boundary, "ChatMemCellExtractor")
    assert hasattr(everalgo.boundary, "WorkspaceMemCellExtractor")
    assert hasattr(everalgo.boundary, "AgentMemCellExtractor")
    assert hasattr(everalgo.boundary, "DetectionOutput")


def test_dunder_all_lists_exact_surface() -> None:
    """__all__ exposes 3 extractors + DetectionOutput (return type for ChatMemCellExtractor.adetect)."""
    assert sorted(everalgo.boundary.__all__) == sorted(
        [
            "AgentMemCellExtractor",
            "ChatMemCellExtractor",
            "DetectionOutput",
            "WorkspaceMemCellExtractor",
        ]
    )
