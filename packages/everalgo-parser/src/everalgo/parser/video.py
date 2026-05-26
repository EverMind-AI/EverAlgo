"""Video parser — deferred.

No implementation yet; selection between Gemini Video and Whisper + frame
extraction is a follow-up decision. Calling ``aparse`` / ``parse`` today
raises ``NotImplementedError`` so dispatch in ``everalgo.parser.aparse``
surfaces the deferred status cleanly to callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everalgo.llm import LLMClient
    from everalgo.types import ParsedContent, RawFile

__all__ = ["aparse", "parse"]


async def aparse(raw_file: RawFile, *, llm: LLMClient) -> ParsedContent:
    """Parse video → ``ParsedContent``. Deferred — see module docstring."""
    raise NotImplementedError(
        "video parser is deferred; selection between Gemini Video / Whisper + frame extraction pending an ADR."
    )


def parse(raw_file: RawFile, *, llm: LLMClient) -> ParsedContent:
    """Sync bridge. Deferred — see module docstring."""
    raise NotImplementedError(
        "video parser is deferred; selection between Gemini Video / Whisper + frame extraction pending an ADR."
    )
