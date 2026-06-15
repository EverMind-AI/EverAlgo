"""User-side memory extractors — 4 Extractors + EpisodeReflector + boundary facade + DetectionResult re-export.

The user-scenario boundary facade (:class:`BoundaryDetector`) lives here; agent-trajectory
boundary detection will be added to ``everalgo.agent_memory`` in Stage 4.
"""

import logging

from everalgo.boundary import DetectionResult
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.boundary import BoundaryDetector
from everalgo.user_memory.episode import EpisodeExtractor
from everalgo.user_memory.foresight import ForesightExtractor
from everalgo.user_memory.profile import ProfileExtractor
from everalgo.user_memory.reflect import EpisodeReflector

__all__ = [
    "AtomicFactExtractor",
    "BoundaryDetector",
    "DetectionResult",
    "EpisodeExtractor",
    "EpisodeReflector",
    "ForesightExtractor",
    "ProfileExtractor",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
