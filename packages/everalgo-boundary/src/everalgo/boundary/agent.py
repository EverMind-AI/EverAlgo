"""Agent execution trace MemCell extractor — stub."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import MemCell, RawData


class AgentMemCellExtractor:
    """Slice agent execution trace into MemCells.

    Stub — real implementation TBD.
    """

    async def adetect(
        self,
        agent_trace: RawData,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[MemCell]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")

    def detect(
        self,
        agent_trace: RawData,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[MemCell]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")
