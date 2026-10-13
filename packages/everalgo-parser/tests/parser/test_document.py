"""Tests for ``everalgo.parser.document`` — PDF spike + dispatch."""

from __future__ import annotations

import base64
import json

import pytest

from everalgo.llm import LLMOutputError
from everalgo.parser import document
from everalgo.parser._replies import _TextReply
from everalgo.parser.prompts.en import PROMPT_FOR_FILE, PROMPT_FOR_HTML
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Modality, RawFile

PDF_BYTES = b"%PDF-1.4 fake pdf payload"

# Not JSON; a missing field; an extra field; a wrong-typed field.
BAD_REPLIES = ["# Extracted markdown", '{"txt": "x"}', '{"text": "x", "extra": 1}', '{"text": 1}']


def _reply(text: str) -> str:
    return json.dumps({"text": text})


def _make_fake(text: str = "extracted") -> FakeLLMClient:
    return FakeLLMClient(responses=[_reply(text)])


@pytest.mark.asyncio
async def test_aparse_pdf_invokes_llm_with_data_uri_part() -> None:
    fake = _make_fake("# Extracted markdown\nhello")
    rf = RawFile(content=PDF_BYTES, mime="application/pdf", extension="pdf")

    result = await document.aparse(rf, llm=fake)

    assert result.text == "# Extracted markdown\nhello"
    assert result.modality is Modality.PDF
    assert result.mime == "application/pdf"
    assert result.metadata == {}

    # Verify the wire payload: the rules as the system message, the PDF data URI alone as the user message.
    assert fake.call_count == 1
    call = fake.calls[0]
    assert call.response_format is _TextReply
    system, user = call.messages
    assert system.role == "system"
    assert system.content == PROMPT_FOR_FILE
    assert user.role == "user"
    assert isinstance(user.content, list)
    assert len(user.content) == 1
    pdf_part = user.content[0]
    assert pdf_part.type == "image_url"
    expected_b64 = base64.b64encode(PDF_BYTES).decode("ascii")
    assert pdf_part.image_url.url == f"data:application/pdf;base64,{expected_b64}"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_reply", BAD_REPLIES)
async def test_aparse_pdf_invalid_reply_raises_llm_output_error(bad_reply: str) -> None:
    fake = FakeLLMClient(responses=[bad_reply])
    with pytest.raises(LLMOutputError, match="parser pdf reply is not a valid _TextReply"):
        await document.aparse(RawFile(content=PDF_BYTES, extension="pdf"), llm=fake)


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
    """HTML handler decodes bytes and sends the cleaned HTML as tagged text (no image_url part)."""
    fake = _make_fake("# heading\nbody")
    html = b"<html><body><h1>heading</h1><p>body</p></body></html>"
    rf = RawFile(content=html, extension="html")
    result = await document.aparse(rf, llm=fake)
    assert result.modality is Modality.HTML
    assert result.text == "# heading\nbody"
    assert result.metadata["raw_chars"] == len(html.decode())
    assert "cleaned_chars" in result.metadata
    assert "model" not in result.metadata
    # Wire: the rules as the system message, the cleaned HTML in <html_content> as the user message.
    call = fake.calls[0]
    assert call.response_format is _TextReply
    system, user = call.messages
    assert system.role == "system"
    assert system.content == PROMPT_FOR_HTML
    assert user.role == "user"
    assert user.content == "<html_content>\n<body><h1>heading</h1><p>body</p></body>\n</html_content>"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_reply", BAD_REPLIES)
async def test_aparse_html_invalid_reply_raises_llm_output_error(bad_reply: str) -> None:
    fake = FakeLLMClient(responses=[bad_reply])
    with pytest.raises(LLMOutputError, match="parser html reply is not a valid _TextReply"):
        await document.aparse(RawFile(content=b"<p>body</p>", extension="html"), llm=fake)


@pytest.mark.asyncio
async def test_aparse_html_placeholder_like_text_reaches_the_model_verbatim() -> None:
    """The page is data: a ``{{html}}`` it contains is not re-substituted."""
    fake = _make_fake()
    await document.aparse(RawFile(content=b"<p>{{html}} and {{count}}</p>", extension="html"), llm=fake)
    assert fake.calls[0].messages[1].content == "<html_content>\n<p>{{html}} and {{count}}</p>\n</html_content>"


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

    async def failing_handler(messages: list[object], **kwargs: object) -> str:
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


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_reply", BAD_REPLIES)
async def test_aparse_email_inline_image_invalid_reply_is_graceful(bad_reply: str) -> None:
    """An invalid OCR reply is a failed OCR: that image gets the placeholder, the email still parses."""
    fake = FakeLLMClient(responses=[bad_reply])
    result = await document.aparse(RawFile(content=_inline_eml(), extension="eml"), llm=fake)
    assert result.modality is Modality.EMAIL
    assert "[image: image_1, OCR failed]" in result.text
    assert fake.calls[0].response_format is _TextReply


@pytest.mark.asyncio
async def test_aparse_email_inline_image_empty_reply_marks_no_content() -> None:
    fake = _make_fake("   ")
    result = await document.aparse(RawFile(content=_inline_eml(), extension="eml"), llm=fake)
    assert "[image: image_1, no content recognised]" in result.text


def _inline_eml() -> bytes:
    return (
        b"From: a@example.com\r\n"
        b'Content-Type: multipart/related; boundary="B"\r\n'
        b"\r\n"
        b"--B\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n"
        b"\r\n"
        b'<html><body><img src="cid:i1"> tail</body></html>\r\n'
        b"--B\r\n"
        b"Content-Type: image/png\r\n"
        b"Content-ID: <i1>\r\n"
        b"Content-Transfer-Encoding: base64\r\n"
        b"\r\n" + _INLINE_PNG_B64 + b"\r\n"
        b"--B--\r\n"
    )


# ---- Office path (LibreOffice stubbed out) ----


@pytest.fixture
def _stub_soffice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the LibreOffice binary lookup and conversion: the converted PDF is ``PDF_BYTES``."""

    async def convert(content: bytes, extension: str, soffice: str) -> bytes:
        return PDF_BYTES

    monkeypatch.setattr(document, "_find_soffice", lambda: "soffice")
    monkeypatch.setattr(document, "_convert_to_pdf_via_soffice", convert)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_soffice")
async def test_aparse_office_reads_the_converted_pdf() -> None:
    fake = _make_fake("office text")
    result = await document.aparse(RawFile(content=b"docx", extension="docx"), llm=fake)
    assert result.modality is Modality.DOCUMENT
    assert result.text == "office text"
    assert result.metadata == {"intermediate_pdf_bytes": len(PDF_BYTES)}
    call = fake.calls[0]
    assert call.response_format is _TextReply
    system, user = call.messages
    assert (system.role, system.content) == ("system", PROMPT_FOR_FILE)
    assert isinstance(user.content, list)
    assert [part.type for part in user.content] == ["image_url"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_stub_soffice")
@pytest.mark.parametrize("bad_reply", BAD_REPLIES)
async def test_aparse_office_invalid_reply_raises_llm_output_error(bad_reply: str) -> None:
    fake = FakeLLMClient(responses=[bad_reply])
    with pytest.raises(LLMOutputError, match="parser office reply is not a valid _TextReply"):
        await document.aparse(RawFile(content=b"docx", extension="docx"), llm=fake)
