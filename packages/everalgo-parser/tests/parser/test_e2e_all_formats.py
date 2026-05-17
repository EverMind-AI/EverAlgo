"""End-to-end multimodal parser test across every supported format.

Run:
    OPENROUTER_API_KEY=sk-or-v1-... \
    HTTPS_PROXY=http://... pytest test_e2e_all_formats.py -v -m integration -s

Gated by ``@pytest.mark.integration`` — default ``pytest`` skips this file.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from everalgo.llm.config import LLMConfig
from everalgo.llm.providers.openai_compat import OpenAICompatClient
from everalgo.parser import aparse
from everalgo.types import Modality, RawFile

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# Algorithm-side pin: parser was tuned against this model.
TEST_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@pytest.fixture(scope="module")
def llm_client() -> OpenAICompatClient:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set; skipping all real-LLM e2e tests")
    return OpenAICompatClient(
        LLMConfig(
            model=TEST_MODEL,
            api_key=api_key,  # type: ignore[arg-type]
            base_url=OPENROUTER_BASE_URL,
            max_tokens=8000,
            timeout=180.0,
        )
    )


# ============================================================
#  LLM-backed modalities (PDF / IMAGE / HTML / Office)
# ============================================================


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixture_name", "expected_modality"),
    [
        ("sample.pdf", Modality.PDF),
        ("sample.png", Modality.IMAGE),
        ("sample.jpg", Modality.IMAGE),
        ("sample.bmp", Modality.IMAGE),
        ("sample.tiff", Modality.IMAGE),
        ("sample.webp", Modality.IMAGE),
        ("sample.svg", Modality.IMAGE),  # rasterised via cairosvg in image.aparse
        ("sample.html", Modality.HTML),
        ("sample.htm", Modality.HTML),
    ],
)
async def test_e2e_llm_backed_modalities(
    llm_client: OpenAICompatClient,
    fixture_name: str,
    expected_modality: Modality,
) -> None:
    """Real fixture → real OpenRouter / Gemini → ParsedContent with non-empty text."""
    fixture = FIXTURES_DIR / fixture_name
    assert fixture.exists(), f"Fixture missing: {fixture}"
    extension = fixture.suffix.lstrip(".")
    rf = RawFile(content=fixture.read_bytes(), extension=extension)

    result = await aparse(rf, llm=llm_client)

    assert result.modality is expected_modality
    assert result.text, f"Empty text returned for {fixture_name}"
    assert len(result.text) >= 5, f"Suspiciously short extraction for {fixture_name}: {result.text!r}"

    print(f"\n=== {fixture_name} ({expected_modality.value}) ===")
    print(f"chars={len(result.text)} model={result.metadata.get('model')}")
    print(result.text[:300])


# ============================================================
#  No-LLM modalities (EMAIL / DIRECT)
# ============================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_url_fetch_and_parse(llm_client: OpenAICompatClient) -> None:
    """Real HTTP fetch → HTML cleanup → LLM extraction → ParsedContent."""
    rf = RawFile(uri="https://example.com/")
    result = await aparse(rf, llm=llm_client)
    assert result.modality is Modality.URL
    assert result.text, "url e2e: LLM-extracted text must not be empty"
    assert result.metadata.get("fetched_uri") == "https://example.com/"
    assert result.metadata.get("fetched_mime") == "text/html"
    assert result.metadata.get("inner_modality") == "html"
    # example.com always has the literal "Example Domain" title.
    title = result.metadata.get("title") or ""
    assert "example" in title.lower(), f"unexpected title for example.com: {title!r}"
    print(f"\n=== https://example.com/ (url, {len(result.text)} chars) ===")
    print(f"title={title!r}")
    print(result.text[:400])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_design_example_uri_pdf_returns_pdf(llm_client: OpenAICompatClient) -> None:
    """design.md §2.1: ``RawFile(uri=pdf_url, mime="application/pdf")`` returns PDF.

    Uses a small public PDF (W3C "dummy.pdf") so the inner modality is verified
    against a real Content-Type=``application/pdf`` response.
    """
    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    rf = RawFile(uri=pdf_url, mime="application/pdf")
    result = await aparse(rf, llm=llm_client)
    assert result.modality is Modality.URL
    assert result.metadata.get("inner_modality") == "pdf", (
        f"design example expected PDF inner_modality; got {result.metadata.get('inner_modality')!r}"
    )
    assert result.metadata.get("fetched_uri") == pdf_url
    assert result.metadata.get("fetched_mime") == "application/pdf"
    assert result.text, "design example: expected non-empty extraction from the PDF"
    print(f"\n=== {pdf_url} (url→pdf, {len(result.text)} chars) ===")
    print(result.text[:300])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_email_parses_locally(llm_client: OpenAICompatClient) -> None:
    """EML uses pure-python stdlib — no LLM call expected, but client is still passed."""
    rf = RawFile(content=(FIXTURES_DIR / "sample.eml").read_bytes(), extension="eml")
    result = await aparse(rf, llm=llm_client)
    assert result.modality is Modality.EMAIL
    assert result.text, "EML body must not be empty"

    print("\n=== sample.eml (email) ===")
    print(f"chars={len(result.text)} subject={result.metadata.get('subject')}")
    print(result.text[:400])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_audio(llm_client: OpenAICompatClient) -> None:
    """Real WAV → Gemini transcription. May return '##UNKNOWN' for unintelligible audio."""
    rf = RawFile(content=(FIXTURES_DIR / "sample.wav").read_bytes(), extension="wav")
    result = await aparse(rf, llm=llm_client)
    assert result.modality is Modality.AUDIO
    assert result.text, "audio transcription must produce some text (even '##UNKNOWN')"
    print(f"\n=== sample.wav (audio, {len(result.text)} chars) ===")
    print(result.text[:400])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_e2e_direct_txt_passthrough(llm_client: OpenAICompatClient) -> None:
    """DIRECT (txt) doesn't hit the LLM — just bytes → UTF-8 decode."""
    rf = RawFile(content=(FIXTURES_DIR / "sample.txt").read_bytes(), extension="txt")
    result = await aparse(rf, llm=llm_client)
    assert result.modality is Modality.DIRECT
    assert result.text

    print("\n=== sample.txt (direct) ===")
    print(result.text[:400])


# ============================================================
#  Office formats (LibreOffice required)
# ============================================================


def _soffice_available() -> bool:
    if shutil.which("soffice"):
        return True
    macos_path = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
    return macos_path.exists()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("fixture_name", ["sample.docx", "sample.xlsx", "sample.pptx"])
async def test_e2e_office_via_libreoffice(llm_client: OpenAICompatClient, fixture_name: str) -> None:
    """Office → LibreOffice → PDF → Gemini. Skipped when soffice is missing."""
    if not _soffice_available():
        pytest.skip("LibreOffice (soffice) not on PATH; skipping Office e2e")

    fixture = FIXTURES_DIR / fixture_name
    rf = RawFile(content=fixture.read_bytes(), extension=fixture.suffix.lstrip("."))
    result = await aparse(rf, llm=llm_client)
    assert result.modality is Modality.DOCUMENT
    assert result.text, f"Empty extraction for {fixture_name}"

    print(f"\n=== {fixture_name} (document) ===")
    print(f"chars={len(result.text)} intermediate_pdf={result.metadata.get('intermediate_pdf_bytes')}")
    print(result.text[:400])
