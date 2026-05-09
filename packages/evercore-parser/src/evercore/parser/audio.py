"""Audio parser — ASR. Stub."""

from __future__ import annotations

from evercore.types import ParsedContent, RawFile

__all__ = ["aparse", "parse"]


async def aparse(raw_file: RawFile) -> ParsedContent:
    """Async parse audio → ParsedContent (含 ASR). Stub — TBD."""
    raise NotImplementedError("stub")


def parse(raw_file: RawFile) -> ParsedContent:
    """Sync parse audio → ParsedContent. Stub — TBD."""
    raise NotImplementedError("stub")
