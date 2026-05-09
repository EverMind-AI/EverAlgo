"""KnowledgeExtractor — ParsedContent → list[KnowledgeMemory]. Stub."""

from __future__ import annotations

from evercore.llm.protocols import LLMClient
from evercore.types import KnowledgeMemory, ParsedContent


class KnowledgeExtractor:
    """Extract knowledge memories from parsed content. Stub."""

    async def aextract(
        self,
        parsed: ParsedContent,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[KnowledgeMemory]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")

    def extract(
        self,
        parsed: ParsedContent,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[KnowledgeMemory]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")
