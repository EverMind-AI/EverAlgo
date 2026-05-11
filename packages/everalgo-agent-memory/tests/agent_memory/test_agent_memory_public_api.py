"""Tests for everalgo.agent_memory package-level public API."""

import everalgo.agent_memory


def test_three_symbols_exported() -> None:
    """agent_memory exposes 2 Extractors + 1 boundary re-export."""
    for name in ("AgentCaseExtractor", "AgentMemCellExtractor", "AgentSkillExtractor"):
        assert hasattr(everalgo.agent_memory, name)


def test_dunder_all_lists_three_symbols() -> None:
    """__all__ exposes exactly 3 symbols."""
    assert sorted(everalgo.agent_memory.__all__) == sorted(
        ["AgentCaseExtractor", "AgentMemCellExtractor", "AgentSkillExtractor"]
    )


def test_two_stub_extractors_instantiable() -> None:
    """2 stub Extractors can be instantiated without args."""
    from everalgo.agent_memory import AgentCaseExtractor, AgentSkillExtractor

    assert AgentCaseExtractor().__class__.__name__ == "AgentCaseExtractor"
    assert AgentSkillExtractor().__class__.__name__ == "AgentSkillExtractor"
