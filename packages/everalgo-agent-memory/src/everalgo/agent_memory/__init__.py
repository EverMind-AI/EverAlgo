"""Agent-side memory extractors — 3 Extractors + boundary facade."""

import logging

from everalgo.agent_memory.boundary import AgentBoundaryDetector
from everalgo.agent_memory.case import AgentCaseExtractor
from everalgo.agent_memory.profile import AgentProfileExtractor
from everalgo.agent_memory.reasons import (
    CaseExtractionResult,
    CaseSkipReason,
    OpOutcome,
    SkillExtractionResult,
    SkillSkipReason,
)
from everalgo.agent_memory.skill import AgentSkillExtractor

__all__ = [
    "AgentBoundaryDetector",
    "AgentCaseExtractor",
    "AgentProfileExtractor",
    "AgentSkillExtractor",
    "CaseExtractionResult",
    "CaseSkipReason",
    "OpOutcome",
    "SkillExtractionResult",
    "SkillSkipReason",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
