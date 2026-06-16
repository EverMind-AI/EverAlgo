"""Tests for everalgo.knowledge package-level public API."""

from __future__ import annotations

import everalgo.knowledge
from everalgo.testing.fake_llm import FakeLLMClient


def test_knowledge_extractor_exported() -> None:
    """Knowledge exposes KnowledgeExtractor at top level."""
    assert hasattr(everalgo.knowledge, "KnowledgeExtractor")


def test_dunder_all_exposes_extractor_and_classifier() -> None:
    """__all__ exposes KnowledgeExtractor plus the classify_category async + sync pair."""
    assert sorted(everalgo.knowledge.__all__) == [
        "KnowledgeExtractor",
        "aclassify_category",
        "classify_category",
    ]
    assert hasattr(everalgo.knowledge, "aclassify_category")
    assert hasattr(everalgo.knowledge, "classify_category")


def test_extractor_instantiable_with_llm() -> None:
    """KnowledgeExtractor accepts a required llm= keyword argument at construction."""
    from everalgo.knowledge import KnowledgeExtractor

    fake = FakeLLMClient(responses=[])
    assert KnowledgeExtractor(llm=fake).__class__.__name__ == "KnowledgeExtractor"
