"""ProfileExtractor — incrementally edit Profile from cluster_episodes. Stub."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.types import MemCell, Profile


class ProfileExtractor:
    """Update user profile based on cluster of historical episodes. Stub."""

    async def aextract(
        self,
        memcell: MemCell,
        *,
        cluster_episodes: list[MemCell],
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> Profile:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")

    def extract(
        self,
        memcell: MemCell,
        *,
        cluster_episodes: list[MemCell],
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> Profile:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")
