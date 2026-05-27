"""Tests for everalgo.boundary package-level public API surface.

Verifies:
- __all__ contains exactly the documented exported symbols
- each symbol is importable and has the expected kind (async-coroutinefunction / class)
- detect_boundaries is natively async (not an asgiref sync bridge)
- WorkspaceMemCellExtractor is NOT part of the public API (unimplemented stub)
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

import everalgo.boundary
from everalgo.boundary import DetectionResult, detect_boundaries
from everalgo.testing.fake_llm import FakeLLMClient


def test_dunder_all_lists_exact_public_surface() -> None:
    """__all__ exposes the four documented symbols — no more, no less."""
    assert sorted(everalgo.boundary.__all__) == sorted(
        [
            "BoundaryDecision",
            "DetectionResult",
            "adetect_boundary_step",
            "detect_boundaries",
        ]
    )


def test_detect_boundaries_is_importable_as_top_level_symbol() -> None:
    assert hasattr(everalgo.boundary, "detect_boundaries")
    assert everalgo.boundary.detect_boundaries is detect_boundaries


def test_detect_boundaries_is_async_coroutinefunction() -> None:
    """detect_boundaries is natively async — callers must ``await`` it."""
    assert inspect.iscoroutinefunction(detect_boundaries)


def test_detection_result_is_importable_as_top_level_symbol() -> None:
    assert hasattr(everalgo.boundary, "DetectionResult")
    assert everalgo.boundary.DetectionResult is DetectionResult


def test_detection_result_is_named_tuple_subclass() -> None:
    """DetectionResult must be a NamedTuple so callers can unpack ``cells, tail = ...``."""
    r = DetectionResult(cells=[], tail=[])
    assert isinstance(r, tuple)
    assert r._fields == ("cells", "tail")


def test_workspace_extractor_is_not_in_public_api() -> None:
    """The unimplemented stub must not leak into the package's public surface (it raises on use)."""
    assert "WorkspaceMemCellExtractor" not in everalgo.boundary.__all__
    assert not hasattr(everalgo.boundary, "WorkspaceMemCellExtractor")


def test_workspace_extractor_stub_raises_when_called() -> None:
    """Reachable only via explicit submodule import; calling it fails fast with NotImplementedError."""
    from everalgo.boundary.workspace import WorkspaceMemCellExtractor

    with pytest.raises(NotImplementedError):
        WorkspaceMemCellExtractor().detect(raw_data=None)  # type: ignore[arg-type]


def test_detect_boundaries_coroutine_is_awaitable() -> None:
    """Calling detect_boundaries() without await returns a coroutine object."""
    coro = detect_boundaries([], llm=FakeLLMClient(responses=[]))
    assert asyncio.iscoroutine(coro)
    # Clean up: must close to avoid 'coroutine was never awaited' warning.
    coro.close()
