"""Public data contracts for EverAlgo.

Adding more memory types (AtomicFact, Foresight, Profile, AgentCase, AgentSkill, ClusterState, ...)
later is a SemVer minor bump for users that import from this module.
"""

import logging

from everalgo.types.agent import (
    AgentCase,
    AgentSkill,
    ToolCall,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)
from everalgo.types.chat import ChatMessage
from everalgo.types.content import ContentBlock, TextContent
from everalgo.types.conversation import ConversationItem, MemCell
from everalgo.types.knowledge import KnowledgeMemory
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
    "ChatMessage",
    "ContentBlock",
    "ConversationItem",
    "Episode",
    "FactCandidate",
    "Foresight",
    "KnowledgeMemory",
    "MemCell",
    "ParsedContent",
    "Profile",
    "RankInput",
    "RankOutput",
    "RawData",
    "RawFile",
    "ScoredItem",
    "TextContent",
    "ToolCall",
    "ToolCallFunction",
    "ToolCallRequest",
    "ToolCallResult",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
