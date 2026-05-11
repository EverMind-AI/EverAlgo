"""AtomicFactExtractor — single MemCell → list[AtomicFact]. Stub."""

from __future__ import annotations

from everalgo.llm.protocols import LLMClient
from everalgo.types import AtomicFact, MemCell


class AtomicFactExtractor:
    """Extract atomic facts from a single MemCell. Stub."""

    async def aextract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[AtomicFact]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")

    def extract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[AtomicFact]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")
