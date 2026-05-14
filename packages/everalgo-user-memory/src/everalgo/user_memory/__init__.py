"""User-side memory extractors — 4 Extractors + boundary re-exports."""

import logging

from everalgo.boundary.chat import ChatMemCellExtractor, DetectionOutput
from everalgo.boundary.workspace import WorkspaceMemCellExtractor
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.episode import EpisodeExtractor
from everalgo.user_memory.foresight import ForesightExtractor
from everalgo.user_memory.profile import ProfileExtractor

__all__ = [
    "AtomicFactExtractor",
    "ChatMemCellExtractor",
    "DetectionOutput",
    "EpisodeExtractor",
    "ForesightExtractor",
    "ProfileExtractor",
    "WorkspaceMemCellExtractor",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
