"""Image parser — OCR. Stub."""

from __future__ import annotations

from everalgo.types import ParsedContent, RawFile

__all__ = ["aparse", "parse"]


async def aparse(raw_file: RawFile) -> ParsedContent:
    """Async parse image → ParsedContent (含 OCR). Stub — TBD."""
    raise NotImplementedError("stub")


def parse(raw_file: RawFile) -> ParsedContent:
    """Sync parse image → ParsedContent. Stub — TBD."""
    raise NotImplementedError("stub")
