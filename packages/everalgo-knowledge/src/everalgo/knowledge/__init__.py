"""File-based knowledge extraction — parsed documents to topic-tree memories."""

from __future__ import annotations

import logging

from everalgo.knowledge._classify import aclassify_category, classify_category
from everalgo.knowledge.extractor import KnowledgeExtractor

__all__ = [
    "KnowledgeExtractor",
    "aclassify_category",
    "classify_category",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
