"""Boundary detection — cut a conversation into MemCells.

Public surface:
- BoundaryDetector      — class binding an LLM client: ``adetect`` / ``detect`` (batch), ``adetect_step`` /
                          ``detect_step`` (incremental, one message at a time)
- adetect_boundaries    — async function: the batch path over ``ConversationItem`` (tool items are kept in the
                          cells but never shown to the model)
- detect_boundaries     — sync bridge over ``adetect_boundaries``; non-event-loop contexts only
- DetectionResult       — ``(cells, tail)`` NamedTuple returned by all of the above
- adetect_boundary_step — async function: the single LLM decision under ``adetect_step``
- BoundaryDecision      — NamedTuple returned by ``adetect_boundary_step``

``everalgo.user_memory`` and ``everalgo.agent_memory`` re-export ``BoundaryDetector`` unchanged.

``WorkspaceMemCellExtractor`` (Jira / Email / Confluence) is NOT YET IMPLEMENTED and is
deliberately omitted from the public API — its methods raise ``NotImplementedError``. Import it
directly from ``everalgo.boundary.workspace`` if you need the reserved placeholder.
"""

import logging

from everalgo.boundary._detector import (
    BoundaryDecision,
    BoundaryDetector,
    DetectionResult,
    adetect_boundaries,
    adetect_boundary_step,
    detect_boundaries,
)

__all__ = [
    "BoundaryDecision",
    "BoundaryDetector",
    "DetectionResult",
    "adetect_boundaries",
    "adetect_boundary_step",
    "detect_boundaries",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
