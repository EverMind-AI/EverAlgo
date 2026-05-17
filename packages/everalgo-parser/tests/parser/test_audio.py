"""Unit tests for everalgo.parser.audio — error paths and happy path."""

from __future__ import annotations

import pytest

from everalgo.llm.types import ChatResponse
from everalgo.parser import audio
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Modality, RawFile


def _fake(text: str = "transcribed") -> FakeLLMClient:
    return FakeLLMClient(responses=[ChatResponse(content=text, model="fake", finish_reason="stop")])


async def test_aparse_unsupported_extension_raises_value_error() -> None:
    """Extension not in _MIME_MAP → ValueError mentioning 'unsupported audio extension'."""
    with pytest.raises(ValueError, match="unsupported audio extension"):
        await audio.aparse(RawFile(content=b"x", extension="xyz"), llm=_fake())


async def test_aparse_empty_content_raises_value_error() -> None:
    """Empty bytes → ValueError."""
    with pytest.raises(ValueError, match="empty"):
        await audio.aparse(RawFile(content=b"", extension="mp3"), llm=_fake())


async def test_aparse_mp3_returns_parsed_content() -> None:
    fake = _fake("hello world")
    result = await audio.aparse(RawFile(content=b"\x00fake mp3", extension="mp3"), llm=fake)
    assert result.modality is Modality.AUDIO
    assert result.text == "hello world"
    assert fake.call_count == 1


async def test_aparse_wav_returns_parsed_content() -> None:
    fake = _fake("wav result")
    result = await audio.aparse(RawFile(content=b"\x00fake wav", extension="wav"), llm=fake)
    assert result.modality is Modality.AUDIO
    assert result.text == "wav result"


async def test_aparse_metadata_contains_model() -> None:
    fake = _fake("text")
    result = await audio.aparse(RawFile(content=b"\x00fake", extension="mp3"), llm=fake)
    assert result.metadata["model"] == "fake"
    assert result.metadata["finish_reason"] == "stop"
