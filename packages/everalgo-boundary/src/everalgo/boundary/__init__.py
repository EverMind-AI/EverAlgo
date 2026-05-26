"""Boundary detection — chat MemCell primitive.

Public surface:
- detect_boundaries — async function: split chat messages into MemCells
- DetectionResult    — ``(cells, tail)`` NamedTuple returned by detect_boundaries
- WorkspaceMemCellExtractor — Jira / Email / Confluence stub (unchanged)

Facade classes (BoundaryDetector / AgentBoundaryDetector) live in everalgo-user-memory and
everalgo-agent-memory respectively; see Stage 3-4 of the refactor.
"""

import logging

from everalgo.boundary.chat import DetectionResult, detect_boundaries
from everalgo.boundary.workspace import WorkspaceMemCellExtractor

__all__ = [
    "DetectionResult",
    "WorkspaceMemCellExtractor",
    "detect_boundaries",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
