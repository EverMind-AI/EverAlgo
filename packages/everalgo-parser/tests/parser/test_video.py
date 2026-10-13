"""Unit tests for the deferred video parser (NotImplementedError stubs)."""

from __future__ import annotations

import pytest

from everalgo.parser import video
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import RawFile


def _fake() -> FakeLLMClient:
    return FakeLLMClient(responses=['{"text": "x"}'])


async def test_aparse_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="deferred"):
        await video.aparse(RawFile(content=b"x", extension="mp4"), llm=_fake())


def test_parse_sync_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="deferred"):
        video.parse(RawFile(content=b"x", extension="mp4"), llm=_fake())
