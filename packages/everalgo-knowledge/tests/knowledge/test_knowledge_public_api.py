"""Tests for everalgo.knowledge package-level public API."""

import everalgo.knowledge
from everalgo.testing.fake_llm import FakeLLMClient


def test_knowledge_extractor_exported() -> None:
    """Knowledge exposes KnowledgeExtractor at top level."""
    assert hasattr(everalgo.knowledge, "KnowledgeExtractor")


def test_dunder_all_lists_one_symbol() -> None:
    """__all__ exposes exactly KnowledgeExtractor."""
    assert everalgo.knowledge.__all__ == ["KnowledgeExtractor"]


def test_extractor_instantiable_with_llm() -> None:
    """KnowledgeExtractor accepts a required llm= keyword argument at construction."""
    from everalgo.knowledge import KnowledgeExtractor

    fake = FakeLLMClient(responses=[])
    assert KnowledgeExtractor(llm=fake).__class__.__name__ == "KnowledgeExtractor"
