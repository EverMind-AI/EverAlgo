"""Workspace data MemCell extractor — Jira / Email / Confluence / ... stub."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import MemCell, RawData


class WorkspaceMemCellExtractor:
    """Slice workspace structured data (Jira / Email / Confluence) into MemCells.

    Stub — real implementation TBD.
    """

    async def adetect(
        self,
        raw_data: RawData,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[MemCell]:
        """EXPERIMENTAL: NOT YET IMPLEMENTED — raises NotImplementedError.

        Stub: returns placeholder.
        """
        raise NotImplementedError("stub")

    def detect(
        self,
        raw_data: RawData,
        *,
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[MemCell]:
        """EXPERIMENTAL: NOT YET IMPLEMENTED — raises NotImplementedError.

        Stub: returns placeholder.
        """
        raise NotImplementedError("stub")
