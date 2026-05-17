"""URL parser — fetch + extract. Stub."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everalgo.types import ParsedContent, RawFile

__all__ = ["aparse", "parse"]


async def aparse(raw_file: RawFile) -> ParsedContent:
    """EXPERIMENTAL: NOT YET IMPLEMENTED — raises NotImplementedError.

    Async parse url → ParsedContent (URL 抓取 + 解析). Stub — TBD.
    """
    raise NotImplementedError("stub")


def parse(raw_file: RawFile) -> ParsedContent:
    """EXPERIMENTAL: NOT YET IMPLEMENTED — raises NotImplementedError.

    Sync parse url → ParsedContent. Stub — TBD.
    """
    raise NotImplementedError("stub")
