"""Tests for ``everalgo.parser.document`` — PDF spike + dispatch."""

from __future__ import annotations

import base64

import pytest

from everalgo.llm.types import ChatResponse
from everalgo.parser import document
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Modality, RawFile

PDF_BYTES = b"%PDF-1.4 fake pdf payload"


def _make_fake(text: str = "extracted") -> FakeLLMClient:
    return FakeLLMClient(responses=[ChatResponse(content=text, model="fake-gemini", finish_reason="stop")])


@pytest.mark.asyncio
async def test_aparse_pdf_invokes_llm_with_data_uri_part() -> None:
    fake = _make_fake("# Extracted markdown\nhello")
    rf = RawFile(content=PDF_BYTES, mime="application/pdf", extension="pdf")

    result = await document.aparse(rf, llm=fake)

    assert result.text == "# Extracted markdown\nhello"
    assert result.modality is Modality.PDF
    assert result.mime == "application/pdf"
    assert result.metadata["model"] == "fake-gemini"
    assert result.metadata["finish_reason"] == "stop"

    # Verify the wire payload: text prompt + image_url data URI for the PDF bytes.
    assert fake.call_count == 1
    call = fake.calls[0]
    assert len(call.messages) == 1
    message = call.messages[0]
    assert message.role == "user"
    assert isinstance(message.content, list)
    assert len(message.content) == 2
    text_part = message.content[0]
    pdf_part = message.content[1]
    assert text_part.type == "text"
    assert "Read this document" in text_part.text
    assert pdf_part.type == "image_url"
    expected_b64 = base64.b64encode(PDF_BYTES).decode("ascii")
    assert pdf_part.image_url.url == f"data:application/pdf;base64,{expected_b64}"


@pytest.mark.asyncio
async def test_aparse_pdf_falls_back_to_pdf_mime_when_missing() -> None:
    """When ``RawFile.mime`` is empty, ``ParsedContent.mime`` defaults to ``application/pdf``."""
    fake = _make_fake()
    rf = RawFile(content=PDF_BYTES, extension="pdf")
    result = await document.aparse(rf, llm=fake)
    assert result.mime == "application/pdf"


@pytest.mark.asyncio
async def test_aparse_pdf_rejects_empty_content() -> None:
    fake = _make_fake()
    rf = RawFile(content=b"", extension="pdf")
    with pytest.raises(ValueError, match="empty"):
        await document.aparse(rf, llm=fake)
    assert fake.call_count == 0


@pytest.mark.asyncio
async def test_aparse_html_invokes_llm_with_text_only_payload() -> None:
    """HTML handler decodes bytes and sends as TextPart (no image_url part)."""
    fake = _make_fake("# heading\nbody")
    html = b"<html><body><h1>heading</h1><p>body</p></body></html>"
    rf = RawFile(content=html, extension="html")
    result = await document.aparse(rf, llm=fake)
    assert result.modality is Modality.HTML
    assert result.text == "# heading\nbody"
    assert result.metadata["raw_chars"] == len(html.decode())
    assert "cleaned_chars" in result.metadata
    # Wire: single TextPart, no image_url.
    msg = fake.calls[0].messages[0]
    assert isinstance(msg.content, list)
    assert len(msg.content) == 1
    assert msg.content[0].type == "text"
    assert "Read this HTML" in msg.content[0].text


@pytest.mark.asyncio
async def test_aparse_email_extracts_headers_and_body_without_llm() -> None:
    """EML handler uses stdlib `email`, no LLM call."""
    fake = _make_fake()
    eml = (
        b"From: a@example.com\r\n"
        b"To: b@example.com\r\n"
        b"Subject: Hello\r\n"
        b"Date: Mon, 1 Jan 2026 00:00:00 +0000\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"\r\n"
        b"Body line 1\r\nBody line 2\r\n"
    )
    rf = RawFile(content=eml, extension="eml")
    result = await document.aparse(rf, llm=fake)
    assert result.modality is Modality.EMAIL
    assert "From: a@example.com" in result.text
    assert "Subject: Hello" in result.text
    assert "Body line 1" in result.text
    assert result.metadata["subject"] == "Hello"
    assert fake.call_count == 0


@pytest.mark.asyncio
async def test_aparse_rejects_modality_outside_document_scope() -> None:
    fake = _make_fake()
    rf = RawFile(content=b"x", extension="png")  # IMAGE — not document.aparse's job
    with pytest.raises(ValueError, match="modality"):
        await document.aparse(rf, llm=fake)


def test_sync_parse_bridge_exists_and_is_callable() -> None:
    """``parse`` is derived from ``aparse`` via ``asgiref.async_to_sync`` (ADR-010)."""
    assert callable(document.parse)


# ---- HTML path ----


@pytest.mark.asyncio
async def test_aparse_html_empty_after_clean_returns_empty_parsed_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTML that collapses to empty after bs4 cleanup returns a no-text ParsedContent."""
    import everalgo.parser.document as doc_mod

    # Monkeypatch clean_html_for_llm to return "" so the empty-after-cleanup branch fires
    monkeypatch.setattr(doc_mod, "clean_html_for_llm", lambda _: "")  # type: ignore[misc]

    fake = _make_fake()
    html = b"<html><body><p>some content</p></body></html>"
    rf = RawFile(content=html, extension="html")
    result = await document.aparse(rf, llm=fake)
    assert result.modality is Modality.HTML
    assert result.text == ""
    assert result.metadata.get("warning") == "empty after html cleanup"
    assert fake.call_count == 0


@pytest.mark.asyncio
async def test_aparse_html_truncates_overlong_input() -> None:
    """HTML longer than _HTML_MAX_INPUT_CHARS gets truncated before the LLM call."""
    fake = _make_fake("truncated")
    # Build HTML that is just slightly larger than the 1M-char cap
    content_size = 1_000_001
    big_html = b"<html><body><p>" + b"x" * content_size + b"</p></body></html>"
    rf = RawFile(content=big_html, extension="html")
    result = await document.aparse(rf, llm=fake)
    assert result.modality is Modality.HTML
    # LLM was still called (after truncation)
    assert fake.call_count == 1


# ---- EML without inline images ----


@pytest.mark.asyncio
async def test_aparse_email_without_inline_images_no_llm_call() -> None:
    """EML with only a plain-text body: no OCR needed, no LLM call."""
    fake = _make_fake()
    eml = (
        b"From: sender@example.com\r\n"
        b"To: receiver@example.com\r\n"
        b"Subject: No Images\r\n"
        b"Date: Mon, 1 Jan 2026 12:00:00 +0000\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"\r\n"
        b"Just plain text, no images here.\r\n"
    )
    rf = RawFile(content=eml, extension="eml")
    result = await document.aparse(rf, llm=fake)
    assert result.modality is Modality.EMAIL
    assert "Just plain text" in result.text
    assert result.metadata["inline_image_count"] == 0
    assert fake.call_count == 0


# ---- EML with inline images OCR ----

# Minimal 1x1 PNG (base64-encoded) used as an inline attachment
_INLINE_PNG_B64 = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


@pytest.mark.asyncio
async def test_aparse_email_with_inline_image_calls_ocr() -> None:
    """EML with one inline PNG: OCR branch fires, text substituted into body."""
    fake = _make_fake("OCR extracted text from image")
    eml = (
        b"From: a@example.com\r\n"
        b"To: b@example.com\r\n"
        b"Subject: With Inline\r\n"
        b'Content-Type: multipart/related; boundary="BOUND"\r\n'
        b"\r\n"
        b"--BOUND\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n"
        b"\r\n"
        b'<html><body>before <img src="cid:img1"> after</body></html>\r\n'
        b"--BOUND\r\n"
        b"Content-Type: image/png\r\n"
        b"Content-ID: <img1>\r\n"
        b"Content-Transfer-Encoding: base64\r\n"
        b"\r\n" + _INLINE_PNG_B64 + b"\r\n"
        b"--BOUND--\r\n"
    )
    rf = RawFile(content=eml, extension="eml")
    result = await document.aparse(rf, llm=fake)
    assert result.modality is Modality.EMAIL
    assert result.metadata["inline_image_count"] == 1
    assert fake.call_count == 1
    # OCR text must appear in the output (either inlined or appended)
    assert "OCR extracted text from image" in result.text
    assert result.metadata.get("inline_image_ocr") == "ok"


@pytest.mark.asyncio
async def test_aparse_email_inline_image_ocr_failure_is_graceful() -> None:
    """If the OCR LLM call fails for a single image, it should not crash the whole parse."""
    call_count = 0

    async def failing_handler(messages: list[object], **kwargs: object) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("simulated OCR failure")

    fake = FakeLLMClient(handler=failing_handler)
    eml = (
        b"From: a@example.com\r\n"
        b"To: b@example.com\r\n"
        b"Subject: OCR Fail\r\n"
        b'Content-Type: multipart/related; boundary="BOUND2"\r\n'
        b"\r\n"
        b"--BOUND2\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n"
        b"\r\n"
        b'<html><body><img src="cid:img2"> done</body></html>\r\n'
        b"--BOUND2\r\n"
        b"Content-Type: image/png\r\n"
        b"Content-ID: <img2>\r\n"
        b"Content-Transfer-Encoding: base64\r\n"
        b"\r\n" + _INLINE_PNG_B64 + b"\r\n"
        b"--BOUND2--\r\n"
    )
    rf = RawFile(content=eml, extension="eml")
    result = await document.aparse(rf, llm=fake)
    # Must not raise; result should still be EMAIL modality
    assert result.modality is Modality.EMAIL
    # Placeholder for failed OCR must appear in output
    assert "OCR failed" in result.text
