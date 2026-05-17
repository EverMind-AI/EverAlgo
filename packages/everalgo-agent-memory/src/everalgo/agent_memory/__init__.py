"""Agent-side memory extractors — 2 Extractors + boundary facade."""

import logging

from everalgo.agent_memory.boundary import AgentBoundaryDetector
from everalgo.agent_memory.case import AgentCaseExtractor
from everalgo.agent_memory.skill import AgentSkillExtractor

__all__ = [
    "AgentBoundaryDetector",
    "AgentCaseExtractor",
    "AgentSkillExtractor",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
