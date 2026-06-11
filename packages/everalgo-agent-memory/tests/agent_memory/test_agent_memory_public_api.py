"""Tests for everalgo.agent_memory package-level public API."""

import everalgo.agent_memory
from everalgo.testing.fake_llm import FakeLLMClient


def test_four_symbols_exported() -> None:
    """agent_memory exposes 3 Extractors + 1 boundary facade (SkillConfig removed from public API)."""
    for name in ("AgentBoundaryDetector", "AgentCaseExtractor", "AgentProfileExtractor", "AgentSkillExtractor"):
        assert hasattr(everalgo.agent_memory, name)
    assert not hasattr(everalgo.agent_memory, "SkillConfig")


def test_dunder_all_lists_four_symbols() -> None:
    """__all__ exposes exactly 4 symbols."""
    assert sorted(everalgo.agent_memory.__all__) == sorted(
        ["AgentBoundaryDetector", "AgentCaseExtractor", "AgentProfileExtractor", "AgentSkillExtractor"]
    )


def test_extractors_instantiable_with_llm() -> None:
    """All Extractors accept a required llm= keyword argument at construction."""
    from everalgo.agent_memory import AgentCaseExtractor, AgentProfileExtractor, AgentSkillExtractor

    fake = FakeLLMClient(responses=[])
    assert AgentCaseExtractor(llm=fake).__class__.__name__ == "AgentCaseExtractor"
    assert AgentSkillExtractor(llm=fake).__class__.__name__ == "AgentSkillExtractor"
    assert AgentProfileExtractor(llm=fake).__class__.__name__ == "AgentProfileExtractor"


def test_skill_config_not_in_public_api() -> None:
    """_SkillCfg is private; SkillConfig is no longer exported from the public API."""
    import everalgo.agent_memory as pkg

    assert not hasattr(pkg, "SkillConfig")
