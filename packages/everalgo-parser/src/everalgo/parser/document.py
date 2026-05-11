"""Document parser — PDF / DOC layout. Stub."""

from __future__ import annotations

from everalgo.types import ParsedContent, RawFile

__all__ = ["aparse", "parse"]


async def aparse(raw_file: RawFile) -> ParsedContent:
    """Async parse document → ParsedContent (PDF / DOC 版面解析). Stub — TBD."""
    raise NotImplementedError("stub")


def parse(raw_file: RawFile) -> ParsedContent:
    """Sync parse document → ParsedContent. Stub — TBD."""
    raise NotImplementedError("stub")
