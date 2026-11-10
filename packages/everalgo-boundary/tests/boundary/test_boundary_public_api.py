"""Tests for everalgo.boundary package-level public API surface.

Verifies:
- __all__ contains exactly the documented exported symbols
- BoundaryDetector is the only entry: two native-async methods, each with a blocking sync bridge
- the module-level entry points of earlier releases are gone
- Cuts is a two-field NamedTuple; to_memcells is a plain synchronous function
- WorkspaceMemCellExtractor is NOT part of the public API (unimplemented stub)
"""

from __future__ import annotations

import inspect
import json

import pytest

import everalgo.boundary
from everalgo.boundary import BoundaryDetector, Cuts, to_memcells
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage


def test_dunder_all_lists_exact_public_surface() -> None:
    """__all__ exposes the three documented symbols — no more, no less."""
    assert sorted(everalgo.boundary.__all__) == ["BoundaryDetector", "Cuts", "to_memcells"]


@pytest.mark.parametrize(
    "name",
    [
        "DetectionResult",
        "BoundaryDecision",
        "adetect_boundaries",
        "detect_boundaries",
        "adetect_boundary_step",
    ],
)
def test_earlier_entry_points_are_gone(name: str) -> None:
    assert not hasattr(everalgo.boundary, name)


@pytest.mark.parametrize("name", ["adetect", "detect"])
def test_the_batch_method_is_named_for_its_path(name: str) -> None:
    """``adetect`` / ``detect`` became ``adetect_batch`` / ``detect_batch``, symmetric with the step pair."""
    assert not hasattr(BoundaryDetector, name)


def test_the_two_methods_are_native_async_with_blocking_bridges() -> None:
    """The ``a`` prefix means native async; the bridge without it is synchronous."""
    assert inspect.iscoroutinefunction(BoundaryDetector.adetect_batch)
    assert inspect.iscoroutinefunction(BoundaryDetector.adetect_step)
    assert not inspect.iscoroutinefunction(BoundaryDetector.detect_batch)
    assert not inspect.iscoroutinefunction(BoundaryDetector.detect_step)


def _messages(count: int) -> list[ChatMessage]:
    return [
        ChatMessage(id=f"m{i}", role="user", content=f"hi {i}", timestamp=1_700_000_000_000 + i, sender_id="u")
        for i in range(count)
    ]


def test_boundary_detector_sync_bridges_block_and_return_cuts() -> None:
    """Called outside an event loop, ``detect_batch`` / ``detect_step`` return the ``Cuts`` themselves."""
    batch_reply = json.dumps({"reasoning": "r", "boundaries": [1]})
    step_reply = json.dumps({"reasoning": "r", "should_end": True})
    detector = BoundaryDetector(llm=FakeLLMClient(responses=[batch_reply, step_reply]))

    assert detector.detect_batch(_messages(2)) == Cuts(ends=[1], tail_start=1)
    assert detector.detect_step(_messages(3)) == Cuts(ends=[2], tail_start=2)


async def test_sync_bridge_refuses_to_run_inside_an_event_loop() -> None:
    """The bridge would block the loop it runs on; asgiref raises instead, pointing at the async method."""
    with pytest.raises(RuntimeError):
        BoundaryDetector(llm=FakeLLMClient(responses=[])).detect_batch(_messages(2))


def test_cuts_is_a_two_field_named_tuple() -> None:
    """Field order is part of the public surface — positional unpacking is documented — so it is pinned here."""
    cuts = Cuts(ends=[2, 4], tail_start=4)

    ends, tail_start = cuts
    assert isinstance(cuts, tuple)
    assert cuts._fields == ("ends", "tail_start")
    assert (ends, tail_start) == ([2, 4], 4)
    assert cuts == Cuts([2, 4], 4)


def test_to_memcells_is_a_plain_function() -> None:
    """Pure compute: synchronous, no client, no bridge."""
    assert not inspect.iscoroutinefunction(to_memcells)
    assert list(inspect.signature(to_memcells).parameters) == ["items", "cuts"]


def test_workspace_extractor_is_not_in_public_api() -> None:
    """The unimplemented stub must not leak into the package's public surface (it raises on use)."""
    assert "WorkspaceMemCellExtractor" not in everalgo.boundary.__all__
    assert not hasattr(everalgo.boundary, "WorkspaceMemCellExtractor")


def test_workspace_extractor_stub_raises_when_called() -> None:
    """Reachable only via explicit submodule import; calling it fails fast with NotImplementedError."""
    from everalgo.boundary.workspace import WorkspaceMemCellExtractor

    with pytest.raises(NotImplementedError):
        WorkspaceMemCellExtractor().detect(raw_data=None)  # type: ignore[arg-type]
