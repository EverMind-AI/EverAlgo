"""Agent-side memory extractors — 3 Extractors + boundary re-exports.

:class:`BoundaryDetector`, :class:`Cuts` and :func:`to_memcells` are ``everalgo.boundary``'s own, re-exported
unchanged; ``everalgo.user_memory`` re-exports the same three.
"""

import logging

from everalgo.agent_memory.boundary import BoundaryDetector, Cuts, to_memcells
from everalgo.agent_memory.case import AgentCaseExtractor
from everalgo.agent_memory.gate import (
    AgentCaseGate,
    CaseGateConfig,
    CaseGateDecision,
    CaseGateQuestion,
    CaseJudge,
    should_extract_skill,
)
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
    "AgentCaseGate",
    "AgentProfileExtractor",
    "AgentSkillExtractor",
    "BoundaryDetector",
    "CaseExtractionResult",
    "CaseGateConfig",
    "CaseGateDecision",
    "CaseGateQuestion",
    "CaseJudge",
    "CaseSkipReason",
    "Cuts",
    "OpOutcome",
    "SkillExtractionResult",
    "SkillSkipReason",
    "should_extract_skill",
    "to_memcells",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
