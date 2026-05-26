"""Tests for top-level ``everalgo.parser.aparse`` dispatch by modality."""

from __future__ import annotations

import pytest

import everalgo.parser as parser_pkg
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Modality, RawFile


def _fake(text: str = "extracted") -> FakeLLMClient:
    return FakeLLMClient(responses=[ChatResponse(content=text, model="fake", finish_reason="stop")])


@pytest.mark.asyncio
async def test_aparse_routes_pdf_to_document() -> None:
    rf = RawFile(content=b"%PDF-1.4 x", extension="pdf")
    result = await parser_pkg.aparse(rf, llm=_fake("ok"))
    assert result.modality is Modality.PDF
    assert result.text == "ok"


@pytest.mark.asyncio
async def test_aparse_direct_passes_through_utf8_text() -> None:
    """DIRECT modality decodes bytes inline; no LLM call."""
    rf = RawFile(content=b"hello world", mime="text/plain", extension="txt")
    # No LLM should be invoked, but supply one to prove it.
    fake = _fake("UNUSED")
    result = await parser_pkg.aparse(rf, llm=fake)
    assert result.text == "hello world"
    assert result.modality is Modality.DIRECT
    assert result.mime == "text/plain"
    assert fake.call_count == 0


@pytest.mark.asyncio
async def test_aparse_direct_replaces_invalid_utf8() -> None:
    rf = RawFile(content=b"good\xff\xffend", extension="md")
    result = await parser_pkg.aparse(rf, llm=_fake())
    assert "good" in result.text
    assert "end" in result.text


@pytest.mark.asyncio
async def test_aparse_unknown_extension_raises() -> None:
    rf = RawFile(content=b"x", extension="zzz")
    with pytest.raises(ValueError, match="not map"):
        await parser_pkg.aparse(rf, llm=_fake())


# ---- MIME-based dispatch ----


@pytest.mark.asyncio
async def test_aparse_dispatches_by_mime_when_extension_missing_pdf() -> None:
    """``RawFile(content=..., mime="application/pdf")`` with no extension routes to PDF."""
    rf = RawFile(content=b"%PDF-1.4 fake", mime="application/pdf")  # extension intentionally empty
    result = await parser_pkg.aparse(rf, llm=_fake("pdf body"))
    assert result.modality is Modality.PDF
    assert result.text == "pdf body"


@pytest.mark.asyncio
async def test_aparse_dispatches_by_mime_when_extension_missing_image() -> None:
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    rf = RawFile(content=png_bytes, mime="image/png")  # no extension
    result = await parser_pkg.aparse(rf, llm=_fake("image text"))
    assert result.modality is Modality.IMAGE
    assert result.text == "image text"


@pytest.mark.asyncio
async def test_aparse_dispatches_by_mime_with_parameters() -> None:
    """``Content-Type: text/html; charset=utf-8`` parameters get stripped."""
    rf = RawFile(content=b"<html><body>x</body></html>", mime="text/html; charset=utf-8")
    result = await parser_pkg.aparse(rf, llm=_fake("ok"))
    assert result.modality is Modality.HTML


@pytest.mark.asyncio
async def test_aparse_dispatches_by_mime_audio_no_extension() -> None:
    rf = RawFile(content=b"\x00\x01 fake", mime="audio/mpeg")
    result = await parser_pkg.aparse(rf, llm=_fake("transcribed"))
    assert result.modality is Modality.AUDIO


@pytest.mark.asyncio
async def test_aparse_dispatches_by_mime_email_no_extension() -> None:
    eml = b"From: a@b\r\nSubject: x\r\n\r\nbody"
    fake = _fake()
    result = await parser_pkg.aparse(RawFile(content=eml, mime="message/rfc822"), llm=fake)
    assert result.modality is Modality.EMAIL
    assert "body" in result.text
    assert fake.call_count == 0


@pytest.mark.asyncio
async def test_aparse_dispatches_by_mime_direct_no_extension() -> None:
    rf = RawFile(content=b"plain words", mime="text/plain")
    fake = _fake("UNUSED")
    result = await parser_pkg.aparse(rf, llm=fake)
    assert result.modality is Modality.DIRECT
    assert result.text == "plain words"
    assert fake.call_count == 0


@pytest.mark.asyncio
async def test_aparse_extension_overrides_mime_when_both_set() -> None:
    """If extension is set, mime is ignored even when they would route differently."""
    # extension=pdf wins, mime=image/png ignored
    rf = RawFile(content=b"%PDF-1.4", extension="pdf", mime="image/png")
    result = await parser_pkg.aparse(rf, llm=_fake("pdf body"))
    assert result.modality is Modality.PDF


@pytest.mark.asyncio
async def test_aparse_empty_content_no_uri_raises() -> None:
    """No content and no http uri → cannot dispatch."""
    with pytest.raises(ValueError, match="no content"):
        await parser_pkg.aparse(RawFile(), llm=_fake())


# ---- max_tokens budget caps preserved from multimodal source ----


@pytest.mark.asyncio
async def test_image_single_path_caps_at_8000_output_tokens() -> None:
    """``image.aparse`` single-shot path passes ``max_tokens=8000`` per multimodal source."""
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    fake = _fake("png text")
    await parser_pkg.aparse(RawFile(content=png_bytes, extension="png"), llm=fake)
    assert fake.calls[0].max_tokens == 8000


@pytest.mark.asyncio
async def test_email_inline_image_ocr_caps_at_8000_output_tokens() -> None:
    """EML inline-image OCR per multimodal source caps each LLM call at 8000."""
    eml_with_inline = (
        b"From: a@example.com\r\n"
        b"To: b@example.com\r\n"
        b"Subject: with inline image\r\n"
        b'Content-Type: multipart/related; boundary="BOUNDARY"\r\n'
        b"\r\n"
        b"--BOUNDARY\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n"
        b"\r\n"
        b'<html><body>before <img src="cid:img1"> after</body></html>\r\n'
        b"--BOUNDARY\r\n"
        b"Content-Type: image/png\r\n"
        b"Content-ID: <img1>\r\n"
        b"Content-Transfer-Encoding: base64\r\n"
        b"\r\n"
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==\r\n"
        b"--BOUNDARY--\r\n"
    )
    fake = _fake("OCR result for inline png")
    rf = RawFile(content=eml_with_inline, extension="eml")
    result = await parser_pkg.aparse(rf, llm=fake)
    assert result.modality is Modality.EMAIL
    # The single LLM call here is the inline-image OCR; it must carry the 8K cap.
    assert fake.call_count == 1
    assert fake.calls[0].max_tokens == 8000


@pytest.mark.asyncio
async def test_aparse_image_routes_to_image_submodule() -> None:
    """IMAGE routing reaches the image submodule which calls the LLM."""
    # Minimal valid 1x1 transparent PNG for PIL aspect-ratio check.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\xdac\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    rf = RawFile(content=png_bytes, extension="png")
    result = await parser_pkg.aparse(rf, llm=_fake("text from image"))
    assert result.modality is Modality.IMAGE
    assert result.text == "text from image"


@pytest.mark.asyncio
async def test_aparse_audio_routes_to_audio_submodule() -> None:
    """AUDIO routing reaches the audio submodule which calls the LLM."""
    rf = RawFile(content=b"\x00\x01\x02 fake audio", extension="mp3")
    result = await parser_pkg.aparse(rf, llm=_fake("transcribed"))
    assert result.modality is Modality.AUDIO
    assert result.text == "transcribed"


@pytest.mark.asyncio
async def test_aparse_html_routes_to_document() -> None:
    rf = RawFile(content=b"<html></html>", extension="html")
    result = await parser_pkg.aparse(rf, llm=_fake("clean"))
    assert result.modality is Modality.HTML


@pytest.mark.asyncio
async def test_aparse_eml_routes_to_document_no_llm() -> None:
    eml = b"From: a@b\r\nSubject: x\r\n\r\nbody"
    fake = _fake()
    result = await parser_pkg.aparse(RawFile(content=eml, extension="eml"), llm=fake)
    assert result.modality is Modality.EMAIL
    assert "body" in result.text
    assert fake.call_count == 0


# ---- URL dispatch branch (line 85 in __init__.py) ----


@pytest.mark.asyncio
async def test_aparse_url_dispatch_routes_to_url_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty content + http uri → dispatched to url.aparse."""
    from everalgo.parser import url
    from everalgo.types import ParsedContent

    captured: list[RawFile] = []

    async def fake_url_aparse(rf: RawFile, *, llm: object) -> ParsedContent:
        captured.append(rf)
        return ParsedContent(text="from url", modality=Modality.URL, mime="text/html")

    monkeypatch.setattr(url, "aparse", fake_url_aparse)
    rf = RawFile(uri="https://example.com/page", content=b"")
    result = await parser_pkg.aparse(rf, llm=_fake())
    assert result.text == "from url"
    assert len(captured) == 1
    assert captured[0].uri == "https://example.com/page"


# ---- unsupported modality error (line 119 in __init__.py) ----


@pytest.mark.asyncio
async def test_aparse_raises_for_unsupported_modality(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Modality that is not in the dispatch table → ValueError 'Unsupported modality'."""
    import everalgo.parser as pkg
    from everalgo.types import Modality

    # Modality.URL is valid but not handled by the dispatch switch (IMAGE/AUDIO/PDF/etc.)
    monkeypatch.setattr(pkg, "get_modality", lambda _ext: Modality.URL)  # type: ignore[misc]
    with pytest.raises(ValueError, match="Unsupported modality"):
        await parser_pkg.aparse(RawFile(content=b"x", extension="mp4"), llm=_fake())


# ---- _is_http_uri helper (line 133 in __init__.py) ----


def test_is_http_uri_accepts_http() -> None:
    from everalgo.parser import _is_http_uri  # type: ignore[attr-defined]

    assert _is_http_uri("http://example.com") is True


def test_is_http_uri_accepts_https() -> None:
    from everalgo.parser import _is_http_uri  # type: ignore[attr-defined]

    assert _is_http_uri("https://example.com/path") is True


def test_is_http_uri_rejects_file_scheme() -> None:
    from everalgo.parser import _is_http_uri  # type: ignore[attr-defined]

    assert _is_http_uri("file:///etc/passwd") is False


def test_is_http_uri_rejects_ftp_scheme() -> None:
    from everalgo.parser import _is_http_uri  # type: ignore[attr-defined]

    assert _is_http_uri("ftp://ftp.example.com/file") is False


# ---- mime fallback / _fill_extension_from_mime (line 147 in __init__.py) ----


@pytest.mark.asyncio
async def test_aparse_fills_extension_from_mime_when_missing() -> None:
    """RawFile(mime='application/pdf') with no extension: extension derived to 'pdf' before forwarding."""
    # Already covered by test_aparse_dispatches_by_mime_when_extension_missing_pdf but this
    # explicitly checks that the routed RawFile carries the derived extension.
    from everalgo.parser import document
    from everalgo.types import ParsedContent

    seen: list[RawFile] = []

    async def capture_rf(rf: RawFile, *, llm: object) -> ParsedContent:
        seen.append(rf)
        return ParsedContent(text="pdf text", modality=Modality.PDF, mime="application/pdf")

    import unittest.mock

    with unittest.mock.patch.object(document, "aparse", side_effect=capture_rf):
        rf = RawFile(content=b"%PDF-1.4 x", mime="application/pdf")
        await parser_pkg.aparse(rf, llm=_fake())

    assert len(seen) == 1
    assert seen[0].extension == "pdf"
