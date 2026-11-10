"""Boundary detection — where to cut a conversation into MemCells.

Public surface:
- BoundaryDetector — the only entry, binding an LLM client: ``adetect_batch`` / ``detect_batch`` (one window of
                     ``ConversationItem``; tool items are kept in place but never shown to the model) and
                     ``adetect_step`` / ``detect_step`` (the judged history plus one new ``ChatMessage``)
- Cuts             — ``(ends, tail_start)`` NamedTuple both methods return: indices into the caller's input
- to_memcells      — pure function: the caller's input and its ``Cuts`` → one ``MemCell`` per closed segment

``everalgo.user_memory`` and ``everalgo.agent_memory`` re-export all three unchanged.

``WorkspaceMemCellExtractor`` (Jira / Email / Confluence) is NOT YET IMPLEMENTED and is
deliberately omitted from the public API — its methods raise ``NotImplementedError``. Import it
directly from ``everalgo.boundary.workspace`` if you need the reserved placeholder.
"""

import logging

from everalgo.boundary._cuts import Cuts, to_memcells
from everalgo.boundary._detector import BoundaryDetector

__all__ = [
    "BoundaryDetector",
    "Cuts",
    "to_memcells",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
