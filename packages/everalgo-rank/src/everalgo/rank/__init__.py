"""Memory ranking — 4 business facades + 3 algorithm-tool modules. All stubs."""

import logging

from everalgo.rank import case, episodic, fusion, profile, rerank, skill, weight

__all__ = [
    "case",
    "episodic",
    "fusion",
    "profile",
    "rerank",
    "skill",
    "weight",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
