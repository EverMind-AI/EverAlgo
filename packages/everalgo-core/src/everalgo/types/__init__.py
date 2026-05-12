"""Public data contracts for EverAlgo — minimal EPISODE-path subset.

Sub-project 1 deliverable. Adding more memory types (AtomicFact,
Foresight, Profile, AgentCase, AgentSkill, ClusterState, ...) later
is a SemVer minor bump for users that import from this module.
"""

import logging

from everalgo.types.agent import AgentCase, AgentSkill
from everalgo.types.knowledge import KnowledgeMemory
from everalgo.types.memcell import MemCell, Message, MessageRole
from everalgo.types.memories import AtomicFact, Episode, Foresight, Profile
from everalgo.types.parsed import ParsedContent
from everalgo.types.rank import (
    Candidate,
    FactCandidate,
    RankInput,
    RankOutput,
    ScoredItem,
)
from everalgo.types.raw import RawData, RawFile

__all__ = [
    "AgentCase",
    "AgentSkill",
    "AtomicFact",
    "Candidate",
    "Episode",
    "FactCandidate",
    "Foresight",
    "KnowledgeMemory",
    "MemCell",
    "Message",
    "MessageRole",
    "ParsedContent",
    "Profile",
    "RankInput",
    "RankOutput",
    "RawData",
    "RawFile",
    "ScoredItem",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
