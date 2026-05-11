"""URL parser — fetch + extract. Stub."""

from __future__ import annotations

from everalgo.types import ParsedContent, RawFile

__all__ = ["aparse", "parse"]


async def aparse(raw_file: RawFile) -> ParsedContent:
    """Async parse url → ParsedContent (URL 抓取 + 解析). Stub — TBD."""
    raise NotImplementedError("stub")


def parse(raw_file: RawFile) -> ParsedContent:
    """Sync parse url → ParsedContent. Stub — TBD."""
    raise NotImplementedError("stub")
