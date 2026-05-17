"""End-to-end pipeline: text file → parser → ChatMessage wrapping → boundary → MemCell.

Simulates the upstream caller path where a hydrated RawFile is parsed into
ParsedContent, the resulting text is wrapped as a single user ChatMessage
batch, and boundary detection then produces MemCell segmentation.

Uses FakeLLMClient at both stages with distinct scripted responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import everalgo.parser as parser_pkg
from everalgo.boundary import detect_boundaries
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import ChatMessage, Modality, RawFile

# Scripted boundary response: no internal cut, all messages form one MemCell.
_BOUNDARY_NO_CUT_JSON = json.dumps({"reasoning": "single coherent topic", "boundaries": [], "should_wait": False})

_FIXTURES = Path(__file__).resolve().parents[2] / "packages" / "everalgo-parser" / "tests" / "fixtures"


def _multi_purpose_fake() -> FakeLLMClient:
    """Returns boundary-shaped JSON for every call (parser doesn't invoke LLM on .txt DIRECT path)."""

    def handler(messages: list[LLMChatMessage], **_kwargs: Any) -> ChatResponse:
        return ChatResponse(content=_BOUNDARY_NO_CUT_JSON, model="fake", finish_reason="stop")

    return FakeLLMClient(handler=handler)


async def test_txt_to_boundary_pipeline() -> None:
    """Plain .txt (DIRECT modality, no LLM in parser) → boundary → 1 MemCell."""
    raw = RawFile(
        content=b"Alice: I want to learn Python async.\nBob: Start with asyncio.run().",
        extension="txt",
    )
    fake = _multi_purpose_fake()

    parsed = await parser_pkg.aparse(raw, llm=fake)
    assert parsed.modality is Modality.DIRECT
    assert parsed.text == "Alice: I want to learn Python async.\nBob: Start with asyncio.run()."
    # DIRECT path bypasses the LLM in parser.
    assert fake.call_count == 0

    # Wrap parser output as a single user-turn ChatMessage batch — simulates the
    # naive ``parse-then-segment`` upstream pattern.
    msgs = [
        ChatMessage(
            id="m1",
            role="user",
            content=parsed.text,
            timestamp=1_700_000_000_000,
            sender_id="u_caller",
        ),
        ChatMessage(
            id="m2",
            role="user",
            content="(follow-up empty)",
            timestamp=1_700_000_001_000,
            sender_id="u_caller",
        ),
    ]

    result = await detect_boundaries(msgs, llm=fake, is_final=True)
    # Boundary stub said "no cut" + is_final=True → all msgs land in single cell.
    assert len(result.cells) == 1
    assert result.tail == []
    assert len(result.cells[0].items) == 2
    assert fake.call_count == 1  # only boundary called LLM


async def test_pdf_to_boundary_pipeline() -> None:
    """PDF (LLM-backed) → boundary → 1 MemCell. Verifies multi-stage LLM call accounting."""
    raw = RawFile(content=(_FIXTURES / "sample.pdf").read_bytes(), extension="pdf")

    # Pattern: first LLM call = parser OCR (returns PDF text); subsequent calls = boundary JSON.
    calls: list[str] = []

    def handler(messages: list[LLMChatMessage], **_kwargs: Any) -> ChatResponse:
        calls.append("call")
        if len(calls) == 1:
            return ChatResponse(
                content="Alice: learning async. Bob: try asyncio.",
                model="fake",
                finish_reason="stop",
            )
        return ChatResponse(content=_BOUNDARY_NO_CUT_JSON, model="fake", finish_reason="stop")

    fake = FakeLLMClient(handler=handler)

    parsed = await parser_pkg.aparse(raw, llm=fake)
    assert parsed.modality is Modality.PDF
    assert "Alice" in parsed.text

    msgs = [
        ChatMessage(
            id=f"m{i}",
            role="user",
            content=line,
            timestamp=1_700_000_000_000 + i * 1000,
            sender_id="u_caller",
        )
        for i, line in enumerate(parsed.text.split(". "))
        if line.strip()
    ]
    if len(msgs) < 2:
        msgs.append(
            ChatMessage(
                id="m_tail",
                role="user",
                content="(filler)",
                timestamp=1_700_000_100_000,
                sender_id="u_caller",
            )
        )

    result = await detect_boundaries(msgs, llm=fake, is_final=True)
    assert len(result.cells) >= 1
    assert fake.call_count >= 2  # parser OCR + boundary
