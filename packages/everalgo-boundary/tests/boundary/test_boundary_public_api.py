"""Tests for everalgo.boundary package-level public API surface.

Verifies:
- __all__ contains exactly the documented exported symbols
- each symbol is importable and has the expected kind (async-coroutinefunction / class)
- adetect_boundaries is natively async; detect_boundaries is its blocking sync bridge
- BoundaryDetector carries a blocking sync bridge for each of its async methods
- WorkspaceMemCellExtractor is NOT part of the public API (unimplemented stub)
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest

import everalgo.boundary
from everalgo.boundary import BoundaryDetector, DetectionResult, adetect_boundaries, detect_boundaries
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage


def test_dunder_all_lists_exact_public_surface() -> None:
    """__all__ exposes the six documented symbols — no more, no less."""
    assert sorted(everalgo.boundary.__all__) == sorted(
        [
            "BoundaryDecision",
            "BoundaryDetector",
            "DetectionResult",
            "adetect_boundaries",
            "adetect_boundary_step",
            "detect_boundaries",
        ]
    )


def test_boundary_functions_are_importable_as_top_level_symbols() -> None:
    assert everalgo.boundary.adetect_boundaries is adetect_boundaries
    assert everalgo.boundary.detect_boundaries is detect_boundaries


def test_adetect_boundaries_is_async_coroutinefunction() -> None:
    """The ``a`` prefix means native async — callers must ``await`` it."""
    assert inspect.iscoroutinefunction(adetect_boundaries)


def _two_messages() -> list[ChatMessage]:
    return [
        ChatMessage(id=f"m{i}", role="user", content=f"hi {i}", timestamp=1_700_000_000_000 + i, sender_id="u")
        for i in range(2)
    ]


def test_detect_boundaries_is_a_blocking_sync_bridge() -> None:
    """Without the prefix the name is synchronous: called outside an event loop, it returns the result itself."""
    reply = json.dumps({"reasoning": "r", "boundaries": [1]})
    messages = _two_messages()

    result = detect_boundaries(messages, llm=FakeLLMClient(responses=[reply]))

    assert not inspect.iscoroutinefunction(detect_boundaries)
    assert result.tail == messages[1:]
    assert [cell.items for cell in result.cells] == [messages[:1]]


async def test_detect_boundaries_refuses_to_run_inside_an_event_loop() -> None:
    """The bridge would block the loop it runs on; asgiref raises instead, pointing at ``adetect_boundaries``."""
    with pytest.raises(RuntimeError):
        detect_boundaries(_two_messages(), llm=FakeLLMClient(responses=[]))


def test_boundary_detector_sync_bridges_block_and_return_the_result() -> None:
    """``detect`` / ``detect_step`` are the blocking twins of ``adetect`` / ``adetect_step``."""
    batch_reply = json.dumps({"reasoning": "r", "boundaries": [1]})
    step_reply = json.dumps({"reasoning": "r", "should_end": False, "confidence": 0.5, "topic_summary": ""})
    messages = _two_messages()
    detector = BoundaryDetector(llm=FakeLLMClient(responses=[batch_reply, step_reply]))

    batch = detector.detect(messages)
    step = detector.detect_step(messages[:1], messages[1])

    assert inspect.iscoroutinefunction(BoundaryDetector.adetect)
    assert inspect.iscoroutinefunction(BoundaryDetector.adetect_step)
    assert not inspect.iscoroutinefunction(BoundaryDetector.detect)
    assert not inspect.iscoroutinefunction(BoundaryDetector.detect_step)
    assert [cell.items for cell in batch.cells] == [messages[:1]]
    assert step == ([], messages)


def test_detection_result_is_importable_as_top_level_symbol() -> None:
    assert hasattr(everalgo.boundary, "DetectionResult")
    assert everalgo.boundary.DetectionResult is DetectionResult


def test_detection_result_is_named_tuple_subclass() -> None:
    """Still a NamedTuple, now two fields wide.

    The field order is part of the public surface — index and positional access are documented — so it is
    pinned here rather than left to whichever order a future edit happens to leave.
    """
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


def test_adetect_boundaries_coroutine_is_awaitable() -> None:
    """Calling adetect_boundaries() without await returns a coroutine object."""
    coro = adetect_boundaries([], llm=FakeLLMClient(responses=[]))
    assert asyncio.iscoroutine(coro)
    # Clean up: must close to avoid 'coroutine was never awaited' warning.
    coro.close()
