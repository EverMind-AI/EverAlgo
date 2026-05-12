"""Boundary extractors — chat / workspace / agent.

Public surface:
- ChatMemCellExtractor — slice chat messages into MemCells (real impl)
- WorkspaceMemCellExtractor — Jira / Email / Confluence stub
- AgentMemCellExtractor — agent trace stub
"""

import logging

from everalgo.boundary.agent import AgentMemCellExtractor
from everalgo.boundary.chat import ChatMemCellExtractor
from everalgo.boundary.workspace import WorkspaceMemCellExtractor

__all__ = [
    "AgentMemCellExtractor",
    "ChatMemCellExtractor",
    "WorkspaceMemCellExtractor",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
