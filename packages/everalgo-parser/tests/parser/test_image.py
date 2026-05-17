"""Unit tests for everalgo.parser.image — covers PIL-based paths, tall-image split/merge.

No cairosvg / SVG paths are tested here (those are pragma'd as native CI gaps).
No real LLM calls: all tests use FakeLLMClient.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from everalgo.llm.types import ChatResponse
from everalgo.parser import image
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Modality, RawFile

# ---- helpers ----


def _png_bytes(width: int, height: int, color: str = "white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _bmp_bytes(width: int = 10, height: int = 10) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="blue").save(buf, format="BMP")
    return buf.getvalue()


def _tiff_bytes(width: int = 10, height: int = 10) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="green").save(buf, format="TIFF")
    return buf.getvalue()


def _fake(*texts: str) -> FakeLLMClient:
    return FakeLLMClient(responses=[ChatResponse(content=t, model="fake", finish_reason="stop") for t in texts])


# ---- error paths ----


async def test_aparse_empty_content_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        await image.aparse(RawFile(content=b"", extension="png"), llm=_fake("unused"))


async def test_aparse_unsupported_extension_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unsupported image extension"):
        await image.aparse(RawFile(content=b"x", extension="xyz"), llm=_fake("unused"))


# ---- single-shot normal path ----


async def test_aparse_png_single_shot_returns_parsed_content() -> None:
    png = _png_bytes(100, 50)  # ratio < 10, single shot
    fake = _fake("ocr result")
    result = await image.aparse(RawFile(content=png, extension="png"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "ocr result"
    assert fake.call_count == 1


async def test_aparse_jpeg_single_shot() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (100, 50)).save(buf, format="JPEG")
    fake = _fake("jpeg ocr")
    result = await image.aparse(RawFile(content=buf.getvalue(), extension="jpg"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "jpeg ocr"


# ---- BMP / TIFF transcode paths ----


async def test_aparse_bmp_transcoded_to_png() -> None:
    bmp = _bmp_bytes(50, 50)
    fake = _fake("bmp text")
    result = await image.aparse(RawFile(content=bmp, extension="bmp"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "bmp text"
    assert fake.call_count == 1


async def test_aparse_tiff_transcoded_to_png() -> None:
    tiff = _tiff_bytes(50, 50)
    fake = _fake("tiff text")
    result = await image.aparse(RawFile(content=tiff, extension="tiff"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "tiff text"
    assert fake.call_count == 1


async def test_aparse_tif_extension_transcoded_to_png() -> None:
    tiff = _tiff_bytes(50, 50)
    fake = _fake("tif text")
    result = await image.aparse(RawFile(content=tiff, extension="tif"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "tif text"


# ---- tall image split path ----


async def test_aparse_tall_image_uses_split_path_and_merge() -> None:
    # height/width = 100/3000 → ratio ~30; splits into 3 parts; 3 OCR + 1 merge = 4 calls
    tall_png = _png_bytes(100, 3000)
    # Provide enough responses for 3 OCR slices + 1 merge call
    fake = _fake("part1 text", "part2 text", "part3 text", "merged text")
    result = await image.aparse(RawFile(content=tall_png, extension="png"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.metadata["tall_image_parts"] >= 2
    # LLM called at least for each slice + 1 merge
    assert fake.call_count >= 3


async def test_aparse_tall_image_all_empty_slices_returns_empty_parsed_content() -> None:
    # Each OCR slice returns empty string
    tall_png = _png_bytes(100, 3000)
    # Need enough empty responses for all OCR slices (no merge call when all empty)
    fake = FakeLLMClient(responses=[ChatResponse(content="", model="fake", finish_reason="stop") for _ in range(10)])
    result = await image.aparse(RawFile(content=tall_png, extension="png"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == ""
    assert result.metadata.get("warning") == "all OCR slices returned empty"


async def test_aparse_tall_image_single_non_empty_slice_skips_merge() -> None:
    # 3 slices; only first returns text; remaining empty → no merge call
    tall_png = _png_bytes(100, 3000)
    # Provide: first slice non-empty, rest empty, extra to prevent exhaustion
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content="only slice text", model="fake", finish_reason="stop"),
            ChatResponse(content="", model="fake", finish_reason="stop"),
            ChatResponse(content="", model="fake", finish_reason="stop"),
            ChatResponse(content="", model="fake", finish_reason="stop"),
            ChatResponse(content="", model="fake", finish_reason="stop"),
        ]
    )
    result = await image.aparse(RawFile(content=tall_png, extension="png"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "only slice text"
    # No merge call — call count equals number of slices only
    assert fake.call_count <= 5


async def test_aparse_tall_image_merge_failure_falls_back_to_join() -> None:
    """When the merge LLM call throws, the result falls back to deterministic join."""
    tall_png = _png_bytes(100, 3000)

    call_count = 0

    async def handler(messages: list[object], **kwargs: object) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        # First N calls are OCR slices returning real text
        if call_count <= 3:
            return ChatResponse(content=f"slice{call_count}", model="fake", finish_reason="stop")
        # Merge call raises
        raise RuntimeError("network error")

    fake = FakeLLMClient(handler=handler)
    result = await image.aparse(RawFile(content=tall_png, extension="png"), llm=fake)
    assert result.modality is Modality.IMAGE
    # Fallback join must produce something from the slice texts
    assert "slice1" in result.text
    assert result.metadata.get("merge_fallback") == "deterministic_join"


# ---- metadata fields ----


async def test_aparse_single_metadata_contains_model_and_finish_reason() -> None:
    png = _png_bytes(100, 50)
    fake = _fake("result")
    result = await image.aparse(RawFile(content=png, extension="png"), llm=fake)
    assert result.metadata["model"] == "fake"
    assert result.metadata["finish_reason"] == "stop"
    assert result.metadata["tall_image_parts"] == 1
