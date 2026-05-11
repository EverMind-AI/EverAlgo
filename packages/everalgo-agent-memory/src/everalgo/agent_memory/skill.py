"""AgentSkillExtractor — same-cluster cases → list[AgentSkill]. Stub."""

from __future__ import annotations

from collections.abc import Sequence

from everalgo.llm.protocols import LLMClient
from everalgo.types import AgentCase, AgentSkill


class AgentSkillExtractor:
    """Aggregate AgentCases in same cluster into AgentSkills. Stub."""

    async def aextract(
        self,
        case: AgentCase,
        *,
        cluster_id: str,
        existing: Sequence[AgentSkill] = (),
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[AgentSkill]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")

    def extract(
        self,
        case: AgentCase,
        *,
        cluster_id: str,
        existing: Sequence[AgentSkill] = (),
        llm: LLMClient | None = None,
        prompt: str | None = None,
    ) -> list[AgentSkill]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")
