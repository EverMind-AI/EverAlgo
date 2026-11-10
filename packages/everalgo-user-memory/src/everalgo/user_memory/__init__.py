"""User-side memory extractors — 4 Extractors + EpisodeReflector + EpisodeSummarizer + boundary re-exports.

:class:`BoundaryDetector`, :class:`Cuts` and :func:`to_memcells` are ``everalgo.boundary``'s own, re-exported
unchanged; ``everalgo.agent_memory`` re-exports the same three.
"""

import logging

from everalgo.user_memory._language import OutputLanguage
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.boundary import BoundaryDetector, Cuts, to_memcells
from everalgo.user_memory.episode import EpisodeExtractor, EpisodeSummarizer
from everalgo.user_memory.foresight import ForesightExtractor
from everalgo.user_memory.profile import ProfileExtractor
from everalgo.user_memory.reflect import EpisodeReflector

__all__ = [
    "AtomicFactExtractor",
    "BoundaryDetector",
    "Cuts",
    "EpisodeExtractor",
    "EpisodeReflector",
    "EpisodeSummarizer",
    "ForesightExtractor",
    "OutputLanguage",
    "ProfileExtractor",
    "to_memcells",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
