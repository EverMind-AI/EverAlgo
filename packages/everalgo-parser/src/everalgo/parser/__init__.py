"""Multimodal parser — top-level dispatch by RawFile.mime. Stubs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from everalgo.parser import audio, document, image, url, video

if TYPE_CHECKING:
    from everalgo.types import ParsedContent, RawFile

__all__ = [
    "aparse",
    "audio",
    "document",
    "image",
    "parse",
    "url",
    "video",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())


async def aparse(raw_file: RawFile) -> ParsedContent:
    """EXPERIMENTAL: NOT YET IMPLEMENTED — raises NotImplementedError.

    Async parse — dispatch by raw_file.mime. Stub — TBD.
    """
    raise NotImplementedError("stub")


def parse(raw_file: RawFile) -> ParsedContent:
    """EXPERIMENTAL: NOT YET IMPLEMENTED — raises NotImplementedError.

    Sync parse — dispatch by raw_file.mime. Stub — TBD.
    """
    raise NotImplementedError("stub")
