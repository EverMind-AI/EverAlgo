"""Public data contracts for EverAlgo.

Adding more memory types (AtomicFact, Foresight, Profile, AgentCase, AgentSkill, ClusterState, ...)
later is a SemVer minor bump for users that import from this module.
"""

import logging

from everalgo.types.agent import (
    AgentCase,
    AgentProfilePatch,
    AgentProfileSignal,
    AgentProfileUpdate,
    AgentSkill,
    ToolCall,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)
from everalgo.types.chat import ChatMessage
from everalgo.types.content import ContentBlock, TextContent
from everalgo.types.conversation import ConversationItem, MemCell
from everalgo.types.knowledge import CategorySpec, KnowledgeMemory
from everalgo.types.memories import AtomicFact, Episode, Foresight, Profile
from everalgo.types.modality import (
    EXTENSION_TO_MODALITY,
    MIME_TO_EXTENSION,
    MIME_TO_MODALITY,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_MIMES,
    Modality,
    get_extension_from_mime,
    get_modality,
    get_modality_from_mime,
)
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
    "EXTENSION_TO_MODALITY",
    "MIME_TO_EXTENSION",
    "MIME_TO_MODALITY",
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_MIMES",
    "AgentCase",
    "AgentProfilePatch",
    "AgentProfileSignal",
    "AgentProfileUpdate",
    "AgentSkill",
    "AtomicFact",
    "Candidate",
    "CategorySpec",
    "ChatMessage",
    "ContentBlock",
    "ConversationItem",
    "Episode",
    "FactCandidate",
    "Foresight",
    "KnowledgeMemory",
    "MemCell",
    "Modality",
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
    "get_extension_from_mime",
    "get_modality",
    "get_modality_from_mime",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
