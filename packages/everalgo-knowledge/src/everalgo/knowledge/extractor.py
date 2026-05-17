"""KnowledgeExtractor — ParsedContent → list[KnowledgeMemory]. Stub."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import KnowledgeMemory, ParsedContent


class KnowledgeExtractor:
    """Extract knowledge memories from parsed content. Stub."""

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        parsed: ParsedContent,
        *,
        prompt: str | None = None,
    ) -> list[KnowledgeMemory]:
        """EXPERIMENTAL: NOT YET IMPLEMENTED — raises NotImplementedError.

        Stub: returns placeholder.
        """
        raise NotImplementedError("stub")

    def extract(
        self,
        parsed: ParsedContent,
        *,
        prompt: str | None = None,
    ) -> list[KnowledgeMemory]:
        """EXPERIMENTAL: NOT YET IMPLEMENTED — raises NotImplementedError.

        Stub: returns placeholder.
        """
        raise NotImplementedError("stub")
