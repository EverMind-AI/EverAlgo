"""End-to-end dispatch tests for everalgo.parser.aparse with FakeLLMClient.

Each test feeds a real fixture file through ``parser.aparse(...)`` and verifies
the resolved Modality, mime, and that the FakeLLMClient was (or wasn't) called
per the modality's documented behaviour.

No network, no API keys — these are the deterministic CI counterpart to the
real-OpenRouter ``packages/everalgo-parser/tests/parser/test_e2e_*.py`` suite.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

import everalgo.parser as parser_pkg
from everalgo.llm.types import ChatMessage, ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Modality, RawFile

FIXTURES = Path(__file__).resolve().parents[2] / "packages" / "everalgo-parser" / "tests" / "fixtures"


def _fake_handler(text: str) -> FakeLLMClient:
    """Multi-call deterministic LLM stub: every call returns ``text``."""

    def handler(messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        return ChatResponse(content=text, model="fake", finish_reason="stop")

    return FakeLLMClient(handler=handler)


def _raw_file(fixture: str) -> RawFile:
    """Load a fixture file as a RawFile with its extension set."""
    data = (FIXTURES / fixture).read_bytes()
    ext = fixture.rsplit(".", 1)[1]
    return RawFile(content=data, extension=ext)


async def test_pdf_dispatch() -> None:
    fake = _fake_handler("fake pdf text")
    result = await parser_pkg.aparse(_raw_file("sample.pdf"), llm=fake)
    assert result.modality is Modality.PDF
    assert result.text == "fake pdf text"
    assert fake.call_count >= 1


async def test_png_image_dispatch() -> None:
    fake = _fake_handler("ocr png")
    result = await parser_pkg.aparse(_raw_file("sample.png"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "ocr png"
    assert fake.call_count >= 1


async def test_jpg_image_dispatch() -> None:
    fake = _fake_handler("ocr jpg")
    result = await parser_pkg.aparse(_raw_file("sample.jpg"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "ocr jpg"
    assert fake.call_count >= 1


async def test_webp_image_dispatch() -> None:
    fake = _fake_handler("ocr webp")
    result = await parser_pkg.aparse(_raw_file("sample.webp"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "ocr webp"
    assert fake.call_count >= 1


async def test_svg_image_dispatch() -> None:
    pytest.importorskip("cairosvg")
    fake = _fake_handler("ocr svg")
    result = await parser_pkg.aparse(_raw_file("sample.svg"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "ocr svg"
    assert fake.call_count >= 1


async def test_audio_wav_dispatch() -> None:
    fake = _fake_handler("transcribed wav")
    result = await parser_pkg.aparse(_raw_file("sample.wav"), llm=fake)
    assert result.modality is Modality.AUDIO
    assert result.text == "transcribed wav"
    assert fake.call_count >= 1


def _soffice_available() -> bool:
    if shutil.which("soffice"):
        return True
    from pathlib import Path as _Path

    return _Path("/Applications/LibreOffice.app/Contents/MacOS/soffice").exists()


async def test_docx_dispatch() -> None:
    if not _soffice_available():
        pytest.skip("LibreOffice (soffice) not available; skipping DOCUMENT dispatch test")
    fake = _fake_handler("docx body")
    result = await parser_pkg.aparse(_raw_file("sample.docx"), llm=fake)
    assert result.modality is Modality.DOCUMENT
    assert result.text == "docx body"
    assert fake.call_count >= 1


async def test_xlsx_dispatch() -> None:
    if not _soffice_available():
        pytest.skip("LibreOffice (soffice) not available; skipping DOCUMENT dispatch test")
    fake = _fake_handler("xlsx body")
    result = await parser_pkg.aparse(_raw_file("sample.xlsx"), llm=fake)
    assert result.modality is Modality.DOCUMENT
    assert result.text == "xlsx body"
    assert fake.call_count >= 1


async def test_html_dispatch() -> None:
    fake = _fake_handler("html parsed")
    result = await parser_pkg.aparse(_raw_file("sample.html"), llm=fake)
    assert result.modality is Modality.HTML
    assert result.text == "html parsed"


async def test_eml_no_inline_dispatch() -> None:
    """Plain EML without inline images must not invoke the LLM."""
    # Construct a minimal EML with no inline images so the parser takes the
    # stdlib-only path (no image OCR, no LLM call).  The sample.eml fixture
    # contains an embedded PNG and therefore does trigger the LLM; we use an
    # explicit inline EML here to guarantee the no-LLM code path.
    eml_bytes = (
        b"From: sender@example.com\r\n"
        b"To: recipient@example.com\r\n"
        b"Subject: plain text email\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n"
        b"\r\n"
        b"This is the email body with no attachments."
    )
    fake = _fake_handler("should not be called")
    result = await parser_pkg.aparse(RawFile(content=eml_bytes, extension="eml"), llm=fake)
    assert result.modality is Modality.EMAIL
    assert result.text  # body must be non-empty
    assert fake.call_count == 0


async def test_txt_direct_dispatch() -> None:
    """TXT is DIRECT modality — bytes are UTF-8 decoded, LLM is never called."""
    fake = _fake_handler("should not be called")
    raw = _raw_file("sample.txt")
    result = await parser_pkg.aparse(raw, llm=fake)
    assert result.modality is Modality.DIRECT
    # The fixture is UTF-8 Chinese text; verify round-trip decode matches raw bytes.
    expected = raw.content.decode("utf-8", errors="replace")
    assert result.text == expected
    assert fake.call_count == 0
