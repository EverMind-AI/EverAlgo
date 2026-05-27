"""Boundary detection — chat MemCell primitive.

Public surface:
- detect_boundaries — async function: split chat messages into MemCells
- DetectionResult    — ``(cells, tail)`` NamedTuple returned by detect_boundaries

``WorkspaceMemCellExtractor`` (Jira / Email / Confluence) is NOT YET IMPLEMENTED and is
deliberately omitted from the public API — its methods raise ``NotImplementedError``. Import it
directly from ``everalgo.boundary.workspace`` if you need the reserved placeholder.

Facade classes (BoundaryDetector / AgentBoundaryDetector) live in everalgo-user-memory and
everalgo-agent-memory respectively; see Stage 3-4 of the refactor.
"""

import logging

from everalgo.boundary.chat import BoundaryDecision, DetectionResult, adetect_boundary_step, detect_boundaries

__all__ = [
    "BoundaryDecision",
    "DetectionResult",
    "adetect_boundary_step",
    "detect_boundaries",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
