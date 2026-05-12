"""Agent-side memory extractors — 2 Extractors + boundary re-export."""

import logging

from everalgo.agent_memory.case import AgentCaseExtractor
from everalgo.agent_memory.skill import AgentSkillExtractor
from everalgo.boundary.agent import AgentMemCellExtractor

__all__ = [
    "AgentCaseExtractor",
    "AgentMemCellExtractor",
    "AgentSkillExtractor",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
