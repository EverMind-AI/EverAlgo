"""File-based knowledge extraction — KnowledgeExtractor stub."""

import logging

from everalgo.knowledge.extractor import KnowledgeExtractor

__all__ = ["KnowledgeExtractor"]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
