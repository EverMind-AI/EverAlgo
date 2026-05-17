"""Tests for everalgo.agent_memory package-level public API."""

import everalgo.agent_memory
from everalgo.testing.fake_llm import FakeLLMClient


def test_three_symbols_exported() -> None:
    """agent_memory exposes 2 Extractors + 1 boundary facade (SkillConfig removed from public API)."""
    for name in ("AgentBoundaryDetector", "AgentCaseExtractor", "AgentSkillExtractor"):
        assert hasattr(everalgo.agent_memory, name)
    assert not hasattr(everalgo.agent_memory, "SkillConfig")


def test_dunder_all_lists_three_symbols() -> None:
    """__all__ exposes exactly 3 symbols."""
    assert sorted(everalgo.agent_memory.__all__) == sorted(
        ["AgentBoundaryDetector", "AgentCaseExtractor", "AgentSkillExtractor"]
    )


def test_extractors_instantiable_with_llm() -> None:
    """Both Extractors accept a required llm= keyword argument at construction."""
    from everalgo.agent_memory import AgentCaseExtractor, AgentSkillExtractor

    fake = FakeLLMClient(responses=[])
    assert AgentCaseExtractor(llm=fake).__class__.__name__ == "AgentCaseExtractor"
    assert AgentSkillExtractor(llm=fake).__class__.__name__ == "AgentSkillExtractor"


def test_skill_config_not_in_public_api() -> None:
    """_SkillCfg is private; SkillConfig is no longer exported from the public API."""
    import everalgo.agent_memory as pkg

    assert not hasattr(pkg, "SkillConfig")
