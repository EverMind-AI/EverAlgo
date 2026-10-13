"""Tests for ``everalgo.utils.http``."""

import httpx
import pytest
import respx

from everalgo.utils import fetch_uri


@pytest.mark.parametrize("uri", ["file:///etc/passwd", "ftp://example.com/a", "data:text/plain,hi"])
async def test_a_scheme_other_than_http_is_refused_before_any_request(uri: str) -> None:
    with pytest.raises(ValueError, match="only http and https"):
        await fetch_uri(uri)


async def test_an_empty_uri_is_refused() -> None:
    with pytest.raises(ValueError, match="empty uri"):
        await fetch_uri("")


async def test_returns_the_body_and_the_bare_media_type(respx_mock: respx.MockRouter) -> None:
    respx_mock.get("https://example.com/foo").mock(
        return_value=httpx.Response(200, content=b"hello", headers={"content-type": "Text/Plain; charset=utf-8"})
    )

    assert await fetch_uri("https://example.com/foo") == (b"hello", "text/plain")


async def test_follows_redirects(respx_mock: respx.MockRouter) -> None:
    respx_mock.get("https://example.com/old").mock(
        return_value=httpx.Response(301, headers={"location": "https://example.com/new"})
    )
    respx_mock.get("https://example.com/new").mock(return_value=httpx.Response(200, content=b"moved"))

    assert await fetch_uri("https://example.com/old") == (b"moved", "")


async def test_an_error_status_raises(respx_mock: respx.MockRouter) -> None:
    respx_mock.get("https://example.com/missing").mock(return_value=httpx.Response(404))

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_uri("https://example.com/missing")
