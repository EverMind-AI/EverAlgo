"""Boundary extractors — chat / workspace / agent.

Public surface:
- ChatMemCellExtractor — slice chat messages into MemCells (new-release batch multi-boundary algorithm)
- DetectionOutput — ``(cells, tail)`` NamedTuple returned by ChatMemCellExtractor.adetect
- WorkspaceMemCellExtractor — Jira / Email / Confluence stub
- AgentMemCellExtractor — agent trace stub
"""

import logging

from everalgo.boundary.agent import AgentMemCellExtractor
from everalgo.boundary.chat import ChatMemCellExtractor, DetectionOutput
from everalgo.boundary.workspace import WorkspaceMemCellExtractor

__all__ = [
    "AgentMemCellExtractor",
    "ChatMemCellExtractor",
    "DetectionOutput",
    "WorkspaceMemCellExtractor",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
