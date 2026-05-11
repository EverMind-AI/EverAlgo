"""AgentCaseExtractor — single MemCell → list[AgentCase]. Stub."""

from __future__ import annotations

from everalgo.llm.protocols import LLMClient
from everalgo.types import AgentCase, MemCell


class AgentCaseExtractor:
    """Extract agent cases from a single MemCell. Stub."""

    async def aextract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[AgentCase]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")

    def extract(
        self,
        memcell: MemCell,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[AgentCase]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")
