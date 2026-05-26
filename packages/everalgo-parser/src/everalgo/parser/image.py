"""Image parser — multimodal LLM OCR with tall-image splitting.

- Normal-ratio images go through a single LLM call.
- Tall screenshots (height/width > 10) are split into overlapping slices
  via ``_utils.split_image_with_overlap``; each slice is OCR'd independently
  and the per-slice results are merged with a second LLM call using
  ``PROMPT_FOR_MERGE`` to dedupe the overlap regions.
- BMP / TIFF are transcoded to PNG via Pillow (Gemini does not accept those
  MIMEs natively).
- SVG is rasterised to PNG via ``cairosvg`` (optional extra ``everalgo-parser[svg]``).
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING

from asgiref.sync import async_to_sync
from PIL import Image

from everalgo.llm import ChatMessage, LLMClient, TextPart, image_url_part_from_bytes
from everalgo.parser._utils import check_aspect_ratio, split_image_with_overlap
from everalgo.parser.prompts.en import PROMPT_FOR_MERGE, PROMPT_FOR_PICTURE
from everalgo.types import Modality, ParsedContent

if TYPE_CHECKING:
    from everalgo.types import RawFile

__all__ = ["aparse", "parse"]

logger = logging.getLogger(__name__)


_MIME_MAP: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "bmp": "image/bmp",
    "svg": "image/svg+xml",
    "tif": "image/tiff",
    "tiff": "image/tiff",
}

# Formats Gemini multimodal does not accept natively. We transcode to PNG via
# Pillow before sending.
_TRANSCODE_TO_PNG: frozenset[str] = frozenset({"bmp", "tif", "tiff"})

# SVG is vector; rasterise via cairosvg before sending.
_SVG_EXTENSIONS: frozenset[str] = frozenset({"svg"})

_TALL_IMAGE_RATIO_THRESHOLD = 10.0

# Token budgets. Per-slice OCR caps at 8K; the merge step gets 4x because it
# joins multiple OCR'd chunks and the merged text is longer than any individual slice.
_MAX_OCR_OUTPUT_TOKENS = 8_000
_MAX_MERGE_OUTPUT_TOKENS = 32_000


async def aparse(raw_file: RawFile, *, llm: LLMClient) -> ParsedContent:
    """Parse an image ``RawFile`` into ``ParsedContent`` via multimodal LLM.

    Tall images (height/width > 10) are split into overlapping vertical
    slices and OCR'd in parallel; per-slice results are then merged with a
    second LLM call. Normal-ratio images go through a single OCR call.
    """
    if not raw_file.content:
        raise ValueError("RawFile.content is empty; cannot parse image")

    extension = raw_file.extension.lower().lstrip(".")
    mime = _MIME_MAP.get(extension)
    if mime is None:
        raise ValueError(
            f"image.aparse: unsupported image extension {raw_file.extension!r}. Supported: {sorted(_MIME_MAP)}"
        )

    payload_bytes = raw_file.content
    payload_mime = mime
    if extension in _TRANSCODE_TO_PNG:
        payload_bytes = _transcode_to_png(raw_file.content)
        payload_mime = "image/png"
    elif extension in _SVG_EXTENSIONS:
        payload_bytes = _rasterise_svg_to_png(raw_file.content)
        payload_mime = "image/png"

    needs_split, ratio = check_aspect_ratio(payload_bytes, max_ratio=_TALL_IMAGE_RATIO_THRESHOLD)
    if needs_split:
        return await _aparse_tall(payload_bytes, payload_mime, raw_file, ratio, llm)
    return await _aparse_single(payload_bytes, payload_mime, raw_file, llm)


parse = async_to_sync(aparse)
"""Sync bridge — see ``document.parse`` for details."""


# ---- internals ----


async def _aparse_single(
    payload_bytes: bytes,
    payload_mime: str,
    raw_file: RawFile,
    client: LLMClient,
) -> ParsedContent:
    """Single-shot OCR call for normal-ratio images."""
    message = ChatMessage(
        role="user",
        content=[
            TextPart(text=PROMPT_FOR_PICTURE),
            image_url_part_from_bytes(payload_bytes, payload_mime),
        ],
    )
    logger.debug("parsing image (single) bytes=%d mime=%s", len(payload_bytes), payload_mime)
    response = await client.chat([message], max_tokens=_MAX_OCR_OUTPUT_TOKENS)
    return ParsedContent(
        text=response.content,
        modality=Modality.IMAGE,
        mime=raw_file.mime or payload_mime,
        metadata={
            "model": response.model,
            "finish_reason": response.finish_reason,
            "tall_image_parts": 1,
        },
    )


async def _aparse_tall(
    payload_bytes: bytes,
    payload_mime: str,
    raw_file: RawFile,
    ratio: float,
    client: LLMClient,
) -> ParsedContent:
    r"""Split-then-merge OCR for tall screenshots.

    1. Split into N overlapping vertical slices.
    2. OCR each slice; **drop empty results** so the merge step sees only
       real text (an all-empty slice from a blank PDF region would otherwise
       confuse the merge LLM).
    3. ≤ 1 non-empty result → return that single text (or empty string when
       all slices came back blank).
    4. Otherwise call the merge LLM with ``temperature=0.0`` and a 4x token
       cap; on merge failure fall back to a deterministic ``\n\n`` join so
       the caller at least gets the raw OCR rather than an exception.
    """
    parts = split_image_with_overlap(payload_bytes, max_ratio=_TALL_IMAGE_RATIO_THRESHOLD)
    logger.info("tall image detected: ratio=%.2f, splitting into %d parts", ratio, len(parts))

    ocr_results: list[str] = []
    last_response = None
    for i, part_bytes in enumerate(parts, 1):
        message = ChatMessage(
            role="user",
            content=[
                TextPart(text=PROMPT_FOR_PICTURE),
                image_url_part_from_bytes(part_bytes, payload_mime),
            ],
        )
        logger.debug("OCR tall image part %d/%d, bytes=%d", i, len(parts), len(part_bytes))
        resp = await client.chat([message], max_tokens=_MAX_OCR_OUTPUT_TOKENS)
        last_response = resp
        # Drop empty per-slice OCR so merge prompt isn't padded with blank parts.
        text = (resp.content or "").strip()
        if text:
            ocr_results.append(text)

    # All slices came back empty — return an empty ParsedContent rather than
    # call the merge LLM with no content (matches original ``parse()`` bailout).
    if not ocr_results:
        assert last_response is not None  # at least one chat() call was made above
        return ParsedContent(
            text="",
            modality=Modality.IMAGE,
            mime=raw_file.mime or payload_mime,
            metadata={
                "model": last_response.model,
                "finish_reason": last_response.finish_reason,
                "tall_image_parts": len(parts),
                "aspect_ratio": round(ratio, 2),
                "warning": "all OCR slices returned empty",
            },
        )

    # Single non-empty result — skip the merge LLM, return as-is (saves 1 call).
    if len(ocr_results) == 1:
        assert last_response is not None
        return ParsedContent(
            text=ocr_results[0],
            modality=Modality.IMAGE,
            mime=raw_file.mime or payload_mime,
            metadata={
                "model": last_response.model,
                "finish_reason": last_response.finish_reason,
                "tall_image_parts": len(parts),
                "aspect_ratio": round(ratio, 2),
            },
        )

    # Multi-slice merge via LLM. Algorithm-layer fallback (NOT engineering
    # retry/fallback per ADR-012): if the merge call itself throws, fall back
    # to a deterministic join so the caller never loses the raw OCR.
    parts_text = "\n\n".join(f"=== Part {i} ===\n{p}" for i, p in enumerate(ocr_results, 1))
    merge_prompt = PROMPT_FOR_MERGE.replace("{parts_text}", parts_text)
    merge_message = ChatMessage(role="user", content=merge_prompt)
    logger.debug("merging %d OCR parts", len(ocr_results))
    merge_fallback_used = False
    try:
        merge_resp = await client.chat(
            [merge_message],
            temperature=0.0,
            max_tokens=_MAX_MERGE_OUTPUT_TOKENS,
        )
        merged_text = merge_resp.content or "\n\n".join(ocr_results)
        if not merge_resp.content:
            merge_fallback_used = True
        last_response = merge_resp
    except Exception as exc:
        logger.warning(
            "LLM merge of tall-image OCR parts failed (%s); falling back to deterministic join", exc, exc_info=True
        )
        merged_text = "\n\n".join(ocr_results)
        merge_fallback_used = True

    assert last_response is not None, "tall-image OCR produced no responses"
    metadata: dict[str, object] = {
        "model": last_response.model,
        "finish_reason": last_response.finish_reason,
        "tall_image_parts": len(parts),
        "aspect_ratio": round(ratio, 2),
    }
    if merge_fallback_used:
        metadata["merge_fallback"] = "deterministic_join"
    return ParsedContent(
        text=merged_text,
        modality=Modality.IMAGE,
        mime=raw_file.mime or payload_mime,
        metadata=metadata,
    )


def _transcode_to_png(data: bytes) -> bytes:
    """Decode arbitrary Pillow-readable bytes and re-encode as PNG."""
    with Image.open(io.BytesIO(data)) as src:
        img: Image.Image = (
            src.convert("RGBA" if "A" in src.mode else "RGB") if src.mode not in ("RGB", "RGBA") else src.copy()
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


def _rasterise_svg_to_png(data: bytes) -> bytes:  # pragma: no cover  (cairosvg native, not in CI)
    """Rasterise SVG bytes to PNG via cairosvg (optional extra)."""
    try:
        import cairosvg  # type: ignore[import-not-found]  # pragma: no cover
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(  # pragma: no cover
            "SVG support requires the optional `cairosvg` dependency. Install via: pip install 'everalgo-parser[svg]'"
        ) from exc
    return cairosvg.svg2png(bytestring=data)  # type: ignore[no-any-return]  # pragma: no cover
