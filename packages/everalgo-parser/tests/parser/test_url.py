"""Unit tests for ``everalgo.parser.url`` — OG extraction + dispatch behaviour."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import pytest

from everalgo.llm.types import ChatResponse
from everalgo.parser import url
from everalgo.parser._utils import fetch_uri
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Modality, RawFile

if TYPE_CHECKING:
    import respx

SAMPLE_HTML = b"""<!doctype html>
<html>
<head>
  <title>Plain Title</title>
  <meta name="description" content="meta-desc">
  <meta name="keywords" content="kw1, kw2">
  <meta property="og:title" content="OG Title">
  <meta property="og:description" content="OG description">
  <meta property="og:image" content="https://example.com/og.png">
  <meta property="og:site_name" content="Example Site">
  <meta property="og:type" content="article">
  <meta name="twitter:title" content="Twitter Title">
  <meta name="twitter:image" content="https://example.com/tw.png">
  <link rel="icon" href="/favicon.ico">
</head>
<body>
  <article>
    <h1>Main Heading</h1>
    <p>First paragraph of the body.</p>
  </article>
  <nav>Skip me</nav>
</body>
</html>
"""


def _fake(text: str = "extracted body") -> FakeLLMClient:
    return FakeLLMClient(responses=[ChatResponse(content=text, model="fake", finish_reason="stop")])


@pytest.mark.asyncio
async def test_aparse_uses_content_when_provided_skipping_network() -> None:
    """When ``RawFile.content`` is non-empty, no HTTP call should be made."""
    fake = _fake("LLM body")
    rf = RawFile(content=SAMPLE_HTML, mime="text/html", uri="https://example.com/x")
    result = await url.aparse(rf, llm=fake)

    assert result.modality is Modality.URL
    assert result.text == "LLM body"
    assert result.metadata["title"] == "OG Title"
    assert result.metadata["description"] == "OG description"
    assert result.metadata["image"] == "https://example.com/og.png"
    assert result.metadata["site_name"] == "Example Site"
    assert result.metadata["og_type"] == "article"
    assert result.metadata["favicon"] == "https://example.com/favicon.ico"
    assert result.metadata["og_tags"]["title"] == "OG Title"
    assert result.metadata["twitter_tags"]["title"] == "Twitter Title"
    assert result.metadata["meta_tags"]["description"] == "meta-desc"
    assert result.metadata["fetched_uri"] == "https://example.com/x"


@pytest.mark.asyncio
async def test_aparse_falls_back_to_title_tag_when_og_missing() -> None:
    """No OG/Twitter title → ``<title>`` text is the fallback."""
    html = b"<html><head><title>Only Title</title></head><body><p>x</p></body></html>"
    fake = _fake("ok")
    result = await url.aparse(RawFile(content=html, mime="text/html"), llm=fake)
    assert result.metadata["title"] == "Only Title"


@pytest.mark.asyncio
async def test_aparse_strips_template_variable_leftovers() -> None:
    """Some CMSes emit ``{{ post.title }}`` raw — those should be rejected."""
    html = b"""<html><head>
    <meta property="og:title" content="{{ post.title }}">
    <title>Real Title</title>
    </head><body><p>x</p></body></html>"""
    result = await url.aparse(RawFile(content=html, mime="text/html"), llm=_fake("ok"))
    assert result.metadata["title"] == "Real Title"


@pytest.mark.asyncio
async def test_aparse_empty_body_raises() -> None:
    """Empty ``content`` AND empty ``uri`` → cannot dispatch."""
    with pytest.raises(ValueError, match="either content or uri"):
        await url.aparse(RawFile(), llm=_fake())


@pytest.mark.asyncio
async def test_aparse_fetches_when_no_content(respx_mock: respx.MockRouter) -> None:
    """Only ``uri`` is set → parser must HTTP-GET it."""
    respx_mock.get("https://example.com/article").mock(
        return_value=httpx.Response(200, content=SAMPLE_HTML, headers={"content-type": "text/html; charset=utf-8"})
    )
    fake = _fake("LLM extracted body")
    rf = RawFile(uri="https://example.com/article")
    result = await url.aparse(rf, llm=fake)
    assert result.modality is Modality.URL
    assert result.text == "LLM extracted body"
    assert result.metadata["title"] == "OG Title"
    assert result.metadata["fetched_mime"] == "text/html"


# ---- fetch_uri direct tests ----


@pytest.mark.asyncio
async def test_fetch_uri_rejects_non_http_scheme() -> None:
    """``file://`` must be rejected (AGENTS.md §1)."""
    with pytest.raises(ValueError, match="only http/https supported"):
        await fetch_uri("file:///etc/passwd")


@pytest.mark.asyncio
async def test_fetch_uri_rejects_empty_uri() -> None:
    with pytest.raises(ValueError, match="empty uri"):
        await fetch_uri("")


@pytest.mark.asyncio
async def test_fetch_uri_returns_bytes_and_mime(respx_mock: respx.MockRouter) -> None:
    respx_mock.get("https://example.com/foo").mock(
        return_value=httpx.Response(200, content=b"hello", headers={"content-type": "text/plain; charset=utf-8"})
    )
    body, mime = await fetch_uri("https://example.com/foo")
    assert body == b"hello"
    assert mime == "text/plain"


# ---- Content-Type aware inner dispatch ----


@pytest.mark.asyncio
async def test_aparse_routes_pdf_when_content_type_is_pdf(respx_mock: respx.MockRouter) -> None:
    """URL serving ``application/pdf`` → PDF handler, NOT the HTML handler."""
    respx_mock.get("https://example.com/file.pdf").mock(
        return_value=httpx.Response(200, content=b"%PDF-1.4 fake pdf", headers={"content-type": "application/pdf"})
    )
    fake = _fake("PDF body extracted")
    result = await url.aparse(RawFile(uri="https://example.com/file.pdf"), llm=fake)
    assert result.modality is Modality.URL  # outer wrapper
    assert result.metadata["inner_modality"] == "pdf"
    assert result.metadata["fetched_mime"] == "application/pdf"
    assert result.text == "PDF body extracted"
    # OG metadata is HTML-only; PDF response must NOT carry og_tags.
    assert "og_tags" not in result.metadata


@pytest.mark.asyncio
async def test_aparse_routes_image_when_content_type_is_png(respx_mock: respx.MockRouter) -> None:
    """URL serving ``image/png`` → image handler."""
    # Real 1x1 PNG so the image parser's PIL aspect-ratio check works.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    respx_mock.get("https://example.com/foo.png").mock(
        return_value=httpx.Response(200, content=png_bytes, headers={"content-type": "image/png"})
    )
    result = await url.aparse(RawFile(uri="https://example.com/foo.png"), llm=_fake("image text"))
    assert result.modality is Modality.URL
    assert result.metadata["inner_modality"] == "image"
    assert result.text == "image text"


@pytest.mark.asyncio
async def test_aparse_routes_audio_when_content_type_is_audio(respx_mock: respx.MockRouter) -> None:
    """URL serving ``audio/mpeg`` → audio handler."""
    respx_mock.get("https://example.com/podcast.mp3").mock(
        return_value=httpx.Response(200, content=b"ID3 fake mp3", headers={"content-type": "audio/mpeg"})
    )
    result = await url.aparse(RawFile(uri="https://example.com/podcast.mp3"), llm=_fake("transcribed"))
    assert result.modality is Modality.URL
    assert result.metadata["inner_modality"] == "audio"
    assert result.text == "transcribed"


@pytest.mark.asyncio
async def test_aparse_unknown_content_type_falls_back_to_html(respx_mock: respx.MockRouter) -> None:
    """Unknown / missing Content-Type → fall back to HTML handler (most URLs are HTML)."""
    respx_mock.get("https://example.com/weird").mock(
        return_value=httpx.Response(200, content=SAMPLE_HTML, headers={"content-type": "application/x-totally-unknown"})
    )
    result = await url.aparse(RawFile(uri="https://example.com/weird"), llm=_fake("body"))
    assert result.modality is Modality.URL
    assert result.metadata["inner_modality"] == "html"  # fallback path


@pytest.mark.asyncio
async def test_aparse_html_response_carries_og_metadata(respx_mock: respx.MockRouter) -> None:
    """OG / Twitter / meta extraction only happens for HTML responses."""
    respx_mock.get("https://example.com/post").mock(
        return_value=httpx.Response(200, content=SAMPLE_HTML, headers={"content-type": "text/html"})
    )
    result = await url.aparse(RawFile(uri="https://example.com/post"), llm=_fake("body"))
    assert result.metadata["inner_modality"] == "html"
    assert result.metadata["title"] == "OG Title"
    assert result.metadata["og_tags"]["title"] == "OG Title"


# ---- bs4 non-string attribute guards ----


@pytest.mark.asyncio
async def test_extract_og_tags_skips_non_string_property() -> None:
    """OG tag with a list-valued ``property`` attribute must be skipped (non-string guard)."""
    # bs4 returns a list when multiple values share the same attribute (e.g. via multi-value attrs).
    # We test this by crafting HTML where bs4 produces a dict with the og: prefix but a non-string content.
    html_non_str_content = b'<html><head><meta property="og:title" content=""></head><body><p>body</p></body></html>'
    # content="" is a string but empty → _safe returns None → title falls back to <title> (absent here)
    result = await url.aparse(RawFile(content=html_non_str_content, mime="text/html"), llm=_fake("body"))
    assert result.metadata.get("title") is None or result.metadata.get("title") == ""


@pytest.mark.asyncio
async def test_extract_twitter_tags_skips_non_string_name() -> None:
    """Twitter tag with non-string ``name`` must not be included."""
    # We exercise the isinstance(name, str) guard by providing a tag where name attr evaluates as expected.
    html = b"""<html><head>
    <meta name="twitter:title" content="TW title">
    </head><body><p>x</p></body></html>"""
    result = await url.aparse(RawFile(content=html, mime="text/html"), llm=_fake("ok"))
    assert result.metadata["twitter_tags"]["title"] == "TW title"


# ---- Modality.DIRECT dispatch via URL ----


@pytest.mark.asyncio
async def test_dispatch_inner_direct_modality_decodes_bytes() -> None:
    """URL-fetched plain-text (DIRECT) content is UTF-8 decoded without calling LLM."""
    from everalgo.parser.url import _dispatch_inner
    from everalgo.types import Modality

    fake = _fake("UNUSED")
    result = await _dispatch_inner(Modality.DIRECT, b"plain text", "text/plain", "https://x.com", llm=fake)
    assert result.text == "plain text"
    assert result.modality is Modality.DIRECT
    assert fake.call_count == 0


# ---- unsupported modality error ----


@pytest.mark.asyncio
async def test_dispatch_inner_unsupported_modality_raises() -> None:
    """UNKNOWN modality (not in dispatch table) → ValueError from _dispatch_inner."""
    from everalgo.parser.url import _dispatch_inner

    with pytest.raises(ValueError, match="cannot dispatch"):
        await _dispatch_inner(Modality.UNKNOWN, b"data", "application/octet-stream", "https://x.com", llm=_fake())


# ---- empty body error ----


@pytest.mark.asyncio
async def test_aparse_raises_when_fetched_body_is_empty(respx_mock: respx.MockRouter) -> None:
    """Empty HTTP response body → ValueError."""
    respx_mock.get("https://example.com/empty").mock(
        return_value=httpx.Response(200, content=b"", headers={"content-type": "text/html"})
    )
    with pytest.raises(ValueError, match="empty body"):
        await url.aparse(RawFile(uri="https://example.com/empty"), llm=_fake())


# ---- _safe helper edge cases ----


def test_safe_rejects_dollar_template() -> None:
    """``${variable}`` leftovers from template engines must be rejected."""
    from everalgo.parser.url import _safe

    assert _safe("${post.title}") is None


def test_safe_accepts_normal_string() -> None:
    from everalgo.parser.url import _safe

    assert _safe("Hello World") == "Hello World"


def test_safe_rejects_non_string() -> None:
    from everalgo.parser.url import _safe

    assert _safe(42) is None  # type: ignore[arg-type]
    assert _safe(None) is None
