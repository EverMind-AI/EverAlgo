"""The caller-input checks raise ``InvalidInputError``, with the offender located and the remedy named."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

import everalgo.parser as parser_pkg
from everalgo.parser import audio, url
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import RawFile
from everalgo.utils import InvalidInputError

if TYPE_CHECKING:
    import respx


async def test_a_file_with_neither_content_nor_uri_is_invalid_input() -> None:
    with pytest.raises(InvalidInputError, match="neither content nor an http"):
        await parser_pkg.aparse(RawFile(extension="pdf"), llm=FakeLLMClient(responses=[]))


async def test_an_unsupported_audio_extension_lists_the_supported_ones() -> None:
    with pytest.raises(
        InvalidInputError, match=r"RawFile\.extension 'xyz' is not a supported audio extension"
    ) as raised:
        await audio.aparse(RawFile(content=b"x", extension="xyz"), llm=FakeLLMClient(responses=[]))

    assert "mp3" in str(raised.value)


async def test_the_uri_in_an_error_keeps_no_query_fragment_or_credentials(respx_mock: respx.MockRouter) -> None:
    signed = "https://user:pw@example.com:8443/empty?token=SECRET#frag"
    respx_mock.get(signed.split("?")[0].replace("user:pw@", "")).mock(
        return_value=httpx.Response(200, content=b"", headers={"content-type": "text/html"})
    )

    with pytest.raises(InvalidInputError, match=r"https://example\.com:8443/empty") as raised:
        await url.aparse(RawFile(uri=signed), llm=FakeLLMClient(responses=[]))

    assert all(secret not in str(raised.value) for secret in ("SECRET", "frag", "pw", "user"))
