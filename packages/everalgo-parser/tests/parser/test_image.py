"""Unit tests for everalgo.parser.image — covers PIL-based paths, tall-image split/merge.

No cairosvg / SVG paths are tested here (those are pragma'd as native CI gaps).
No real LLM calls: all tests use FakeLLMClient, scripted with ``{"text": ...}`` replies.
"""

from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from everalgo.llm import LLMOutputError
from everalgo.parser import image
from everalgo.parser._replies import _TextReply
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Modality, RawFile

# Not JSON; a missing field; an extra field; a wrong-typed field.
BAD_REPLIES = ["plain OCR text", '{"txt": "x"}', '{"text": "x", "extra": 1}', '{"text": 1}']

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


def _reply(text: str) -> str:
    return json.dumps({"text": text})


def _fake(*texts: str) -> FakeLLMClient:
    return FakeLLMClient(responses=[_reply(t) for t in texts])


# ---- error paths ----


async def test_aparse_empty_content_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        await image.aparse(RawFile(content=b"", extension="png"), llm=_fake("unused"))


async def test_aparse_unsupported_extension_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not a supported image extension"):
        await image.aparse(RawFile(content=b"x", extension="xyz"), llm=_fake("unused"))


# ---- single-shot normal path ----


async def test_aparse_png_single_shot_returns_parsed_content() -> None:
    png = _png_bytes(100, 50)  # ratio < 10, single shot
    fake = _fake("ocr result")
    result = await image.aparse(RawFile(content=png, extension="png"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "ocr result"
    assert fake.call_count == 1
    assert fake.calls[0].response_format is _TextReply


async def test_aparse_jpeg_single_shot() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (100, 50)).save(buf, format="JPEG")
    fake = _fake("jpeg ocr")
    result = await image.aparse(RawFile(content=buf.getvalue(), extension="jpg"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "jpeg ocr"


@pytest.mark.parametrize("bad_reply", BAD_REPLIES)
async def test_aparse_single_invalid_reply_raises_llm_output_error(bad_reply: str) -> None:
    fake = FakeLLMClient(responses=[bad_reply])
    with pytest.raises(LLMOutputError, match="parser image reply is not a valid _TextReply"):
        await image.aparse(RawFile(content=_png_bytes(100, 50), extension="png"), llm=fake)


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
    fake = _fake("part1 text", "part2 text", "part3 text", "merged text")
    result = await image.aparse(RawFile(content=tall_png, extension="png"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "merged text"
    assert result.metadata["tall_image_parts"] == 3
    assert "merge_fallback" not in result.metadata
    assert fake.call_count == 4
    assert all(call.response_format is _TextReply for call in fake.calls)
    merge_call = fake.calls[-1]
    assert merge_call.temperature == 0.0
    assert merge_call.max_tokens == 32_000


async def test_aparse_tall_image_all_empty_slices_returns_empty_parsed_content() -> None:
    tall_png = _png_bytes(100, 3000)
    # No merge call when every slice is empty; the extra replies are never consumed.
    fake = FakeLLMClient(responses=[_reply("") for _ in range(10)])
    result = await image.aparse(RawFile(content=tall_png, extension="png"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == ""
    assert result.metadata.get("warning") == "all OCR slices returned empty"
    assert fake.call_count == 3


async def test_aparse_tall_image_single_non_empty_slice_skips_merge() -> None:
    tall_png = _png_bytes(100, 3000)
    fake = FakeLLMClient(responses=[_reply("only slice text"), _reply(""), _reply("  "), _reply("")])
    result = await image.aparse(RawFile(content=tall_png, extension="png"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "only slice text"
    # No merge call — one call per slice only.
    assert fake.call_count == 3


@pytest.mark.parametrize("bad_reply", BAD_REPLIES)
async def test_aparse_tall_image_invalid_slice_reply_raises_llm_output_error(bad_reply: str) -> None:
    fake = FakeLLMClient(responses=[_reply("slice1"), bad_reply])
    with pytest.raises(LLMOutputError, match="parser image reply"):
        await image.aparse(RawFile(content=_png_bytes(100, 3000), extension="png"), llm=fake)


async def test_aparse_tall_image_merge_failure_falls_back_to_join() -> None:
    """When the merge LLM call throws, the result falls back to deterministic join."""
    tall_png = _png_bytes(100, 3000)
    fake = FakeLLMClient(responses=[_reply("slice1"), _reply("slice2"), _reply("slice3"), RuntimeError("network")])
    result = await image.aparse(RawFile(content=tall_png, extension="png"), llm=fake)
    assert result.modality is Modality.IMAGE
    assert result.text == "slice1\n\nslice2\n\nslice3"
    assert result.metadata.get("merge_fallback") == "deterministic_join"


@pytest.mark.parametrize("bad_reply", BAD_REPLIES)
async def test_aparse_tall_image_invalid_merge_reply_falls_back_to_join(bad_reply: str) -> None:
    """An invalid merge reply is a failed merge: the raw OCR is joined, not lost to an exception."""
    fake = FakeLLMClient(responses=[_reply("slice1"), _reply("slice2"), _reply("slice3"), bad_reply])
    result = await image.aparse(RawFile(content=_png_bytes(100, 3000), extension="png"), llm=fake)
    assert result.text == "slice1\n\nslice2\n\nslice3"
    assert result.metadata.get("merge_fallback") == "deterministic_join"


async def test_aparse_tall_image_empty_merge_reply_falls_back_to_join() -> None:
    fake = _fake("slice1", "slice2", "slice3", "")
    result = await image.aparse(RawFile(content=_png_bytes(100, 3000), extension="png"), llm=fake)
    assert result.text == "slice1\n\nslice2\n\nslice3"
    assert result.metadata.get("merge_fallback") == "deterministic_join"


# ---- metadata fields ----


async def test_aparse_single_metadata_counts_parts_and_names_no_model() -> None:
    """The served model is the host's to record (llm-contract §6.4); ``achat`` returns only the text."""
    png = _png_bytes(100, 50)
    result = await image.aparse(RawFile(content=png, extension="png"), llm=_fake("result"))
    assert result.metadata == {"tall_image_parts": 1}
