"""everalgo.parser — multimodal file → ParsedContent dispatch.

Demonstrates how a ``RawFile`` (hydrated bytes + extension hint) flows
through ``parser.aparse(...)`` to one of the modality-specific submodules
(IMAGE / PDF / DOCUMENT / DIRECT / ...). Three illustrative inputs:
plain text (DIRECT, no LLM), inline-bytes PDF (PDF, LLM-backed),
and HTML (HTML, LLM-backed). Uses ``FakeLLMClient`` so no API key is needed.

Run:
    uv run python examples/07_parser_dispatch.py
"""

from __future__ import annotations

import asyncio
from typing import Any

import everalgo.parser as parser_pkg
from everalgo.llm.types import ChatMessage, ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import RawFile


def _make_fake(text: str) -> FakeLLMClient:
    """LLM stub: every call returns the same scripted text."""

    def handler(messages: list[ChatMessage], **_kwargs: Any) -> ChatResponse:
        return ChatResponse(content=text, model="fake", finish_reason="stop")

    return FakeLLMClient(handler=handler)


async def main() -> None:
    """Run aparse on three inputs and print resolved modality + extracted text."""
    # 1. DIRECT — utf-8 text bytes, no LLM call.
    txt_raw = RawFile(
        content=b"Hello, EverAlgo! This is a plain text snippet.",
        extension="txt",
    )
    txt_fake = _make_fake("UNUSED — DIRECT path skips LLM")
    parsed_txt = await parser_pkg.aparse(txt_raw, llm=txt_fake)
    print(f"[txt]  modality={parsed_txt.modality.value:8}  text={parsed_txt.text!r}")
    print(f"        llm calls = {txt_fake.call_count} (DIRECT path does not invoke LLM)")
    print()

    # 2. PDF — LLM-backed multimodal OCR (stubbed text below).
    pdf_raw = RawFile(content=b"%PDF-1.4 ...minimal stub...", extension="pdf")
    pdf_fake = _make_fake("Extracted PDF body: lorem ipsum dolor sit amet.")
    parsed_pdf = await parser_pkg.aparse(pdf_raw, llm=pdf_fake)
    print(f"[pdf]  modality={parsed_pdf.modality.value:8}  text={parsed_pdf.text!r}")
    print(f"        llm calls = {pdf_fake.call_count}")
    print()

    # 3. HTML — text extraction (LLM-backed for normalised output).
    html_raw = RawFile(
        content=b"<html><body><h1>Title</h1><p>Body text here.</p></body></html>",
        extension="html",
    )
    html_fake = _make_fake("Title\nBody text here.")
    parsed_html = await parser_pkg.aparse(html_raw, llm=html_fake)
    print(f"[html] modality={parsed_html.modality.value:8}  text={parsed_html.text!r}")
    print(f"        llm calls = {html_fake.call_count}")


if __name__ == "__main__":
    asyncio.run(main())
