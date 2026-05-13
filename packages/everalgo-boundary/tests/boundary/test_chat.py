"""Tests for everalgo.boundary.chat — ChatMemCellExtractor interface contract (stub stage)."""

from __future__ import annotations

import pytest

from everalgo.boundary.chat import ChatMemCellExtractor, DetectionOutput
from everalgo.types import MemCell, Message, MessageRole


def _user(content: str, ts: int = 1700000000000) -> Message:
    return Message(role=MessageRole.USER, content=content, timestamp=ts)


def test_detection_output_supports_positional_unpacking() -> None:
    """``cells, tail = output`` works because DetectionOutput subclasses ``tuple``."""
    out = DetectionOutput(cells=[], tail=[_user("hi")])
    cells, tail = out
    assert cells == []
    assert tail == [_user("hi")]


def test_detection_output_supports_named_access() -> None:
    """``output.cells`` / ``output.tail`` work via NamedTuple field names."""
    out = DetectionOutput(cells=[], tail=[_user("hi")])
    assert out.cells == []
    assert out.tail == [_user("hi")]


def test_detection_output_supports_index_access() -> None:
    """``output[0]`` / ``output[1]`` work via tuple indexing."""
    out = DetectionOutput(cells=[MemCell(id="mc_1", messages=[], timestamp=0)], tail=[])
    assert out[0][0].id == "mc_1"
    assert out[1] == []


async def test_adetect_is_stub_raising_not_implemented() -> None:
    """Interface contract is defined; body is a stub pending implementation."""
    with pytest.raises(NotImplementedError):
        await ChatMemCellExtractor().adetect(
            [_user("hello")],
            is_final=True,
            hard_token_limit=1024,
            hard_msg_limit=10,
        )


def test_sync_detect_bridge_is_callable_and_stub() -> None:
    """Sync bridge exists, mirrors async stub semantics."""
    with pytest.raises(NotImplementedError):
        ChatMemCellExtractor().detect([_user("hello")])
