"""Agent-side memory extractors — 3 Extractors + the re-exported ``everalgo.boundary.BoundaryDetector``."""

import logging

from everalgo.agent_memory.boundary import BoundaryDetector
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
    "AgentCaseExtractor",
    "AgentProfileExtractor",
    "AgentSkillExtractor",
    "BoundaryDetector",
    "CaseExtractionResult",
    "CaseSkipReason",
    "OpOutcome",
    "SkillExtractionResult",
    "SkillSkipReason",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
