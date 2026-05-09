"""Tests for evercore.knowledge package-level public API."""

import evercore.knowledge


def test_knowledge_extractor_exported() -> None:
    """Knowledge exposes KnowledgeExtractor at top level."""
    assert hasattr(evercore.knowledge, "KnowledgeExtractor")


def test_dunder_all_lists_one_symbol() -> None:
    """__all__ exposes exactly KnowledgeExtractor."""
    assert evercore.knowledge.__all__ == ["KnowledgeExtractor"]


def test_extractor_instantiable() -> None:
    """KnowledgeExtractor can be instantiated without args."""
    from evercore.knowledge import KnowledgeExtractor

    assert KnowledgeExtractor().__class__.__name__ == "KnowledgeExtractor"
