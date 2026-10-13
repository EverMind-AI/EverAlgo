"""Unit tests for everalgo.parser.audio — error paths, happy path and the reply contract."""

from __future__ import annotations

import json

import pytest

from everalgo.llm import LLMOutputError
from everalgo.parser import audio
from everalgo.parser._replies import _TextReply
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Modality, RawFile

# Not JSON; a missing field; an extra field; a wrong-typed field.
BAD_REPLIES = ["transcribed", '{"txt": "x"}', '{"text": "x", "extra": 1}', '{"text": 1}']


def _fake(text: str = "transcribed") -> FakeLLMClient:
    return FakeLLMClient(responses=[json.dumps({"text": text})])


async def test_aparse_unsupported_extension_raises_value_error() -> None:
    """Extension not in _MIME_MAP → ValueError mentioning 'unsupported audio extension'."""
    with pytest.raises(ValueError, match="not a supported audio extension"):
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
    assert fake.calls[0].response_format is _TextReply


async def test_aparse_wav_returns_parsed_content() -> None:
    fake = _fake("wav result")
    result = await audio.aparse(RawFile(content=b"\x00fake wav", extension="wav"), llm=fake)
    assert result.modality is Modality.AUDIO
    assert result.text == "wav result"


async def test_aparse_unknown_sentinel_passes_through_verbatim() -> None:
    """The prompt's ``##UNKNOWN`` answer now arrives inside ``text`` and is returned as is."""
    result = await audio.aparse(RawFile(content=b"\x00", extension="mp3"), llm=_fake("##UNKNOWN"))
    assert result.text == "##UNKNOWN"


@pytest.mark.parametrize("bad_reply", BAD_REPLIES)
async def test_aparse_invalid_reply_raises_llm_output_error(bad_reply: str) -> None:
    fake = FakeLLMClient(responses=[bad_reply])
    with pytest.raises(LLMOutputError, match="parser audio reply is not a valid _TextReply"):
        await audio.aparse(RawFile(content=b"\x00fake", extension="mp3"), llm=fake)


async def test_aparse_metadata_names_no_model() -> None:
    """The served model is the host's to record (llm-contract §6.4); ``achat`` returns only the text."""
    result = await audio.aparse(RawFile(content=b"\x00fake", extension="mp3"), llm=_fake("text"))
    assert result.metadata == {}
