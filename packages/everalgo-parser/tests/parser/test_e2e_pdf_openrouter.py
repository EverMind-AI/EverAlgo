"""End-to-end PDF parser test against real OpenRouter / Gemini.

Run:
    OPENROUTER_API_KEY=sk-or-v1-... pytest test_e2e_pdf_openrouter.py -v -m integration

Gated by ``@pytest.mark.integration`` so the default ``pytest`` invocation skips
this; only runs when explicitly opted in via the marker AND a real key is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from everalgo.llm.config import LLMConfig
from everalgo.llm.providers.openai_compat import OpenAICompatClient
from everalgo.parser import document
from everalgo.types import Modality, RawFile

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

# Per algorithm-side decision: pin to the model the parser was originally
# tuned against, NOT whatever a deployment config.yaml might select.
TEST_MODEL = "google/gemini-3-flash-preview"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pdf_e2e_against_real_openrouter() -> None:
    """Real PDF bytes → OpenRouter → Gemini → ParsedContent.text non-empty."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set in env; skipping real-LLM e2e")

    fixture = FIXTURES_DIR / "sample.pdf"
    assert fixture.exists(), f"Fixture missing: {fixture}"
    pdf_bytes = fixture.read_bytes()
    assert len(pdf_bytes) > 0

    client = OpenAICompatClient(
        LLMConfig(
            model=TEST_MODEL,
            api_key=api_key,  # type: ignore[arg-type]  # pydantic coerces str → SecretStr
            base_url=OPENROUTER_BASE_URL,
            max_tokens=8000,
            timeout=120.0,
        )
    )

    raw = RawFile(content=pdf_bytes, mime="application/pdf", extension="pdf")
    result = await document.aparse(raw, llm=client)

    # Structural assertions on the contract.
    assert result.modality is Modality.PDF
    assert result.mime == "application/pdf"
    assert result.metadata.get("model"), "metadata should carry the model name"

    # Content assertion: Gemini must have extracted *something* from the PDF.
    assert result.text, "ParsedContent.text was empty — real LLM call produced no content"
    assert len(result.text) >= 10, f"Extracted text suspiciously short: {result.text!r}"

    # Surface the result for human inspection when -s is used.
    print(f"\n--- Extracted {len(result.text)} chars from sample.pdf ---")
    print(result.text[:800])
    print("--- metadata ---")
    print(result.metadata)
