"""ForesightExtractor — single MemCell → list[Foresight]. Stub."""

from __future__ import annotations

from everalgo.llm.protocols import LLMClient
from everalgo.types import Foresight, MemCell


class ForesightExtractor:
    """Extract foresights (anticipated commitments) from a single MemCell. Stub."""

    async def aextract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[Foresight]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")

    def extract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[Foresight]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")
