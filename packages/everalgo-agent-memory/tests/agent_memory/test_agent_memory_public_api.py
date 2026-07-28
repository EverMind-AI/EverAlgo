"""Tests for everalgo.agent_memory package-level public API."""

import everalgo.agent_memory
from everalgo.testing.fake_llm import FakeLLMClient

_EXPECTED_EXPORTS = [
    # 3 Extractors + 1 boundary facade
    "AgentBoundaryDetector",
    "AgentCaseExtractor",
    "AgentProfileExtractor",
    "AgentSkillExtractor",
    # Diagnostic surface of the *_with_reason methods
    "CaseExtractionResult",
    "CaseSkipReason",
    "OpOutcome",
    "SkillExtractionResult",
    "SkillSkipReason",
]


def test_expected_symbols_exported() -> None:
    """agent_memory exposes the extractors plus the skip-reason types (SkillConfig removed)."""
    for name in _EXPECTED_EXPORTS:
        assert hasattr(everalgo.agent_memory, name)
    assert not hasattr(everalgo.agent_memory, "SkillConfig")


def test_dunder_all_matches_expected_exports() -> None:
    """__all__ exposes exactly the expected symbols — no more, no less."""
    assert sorted(everalgo.agent_memory.__all__) == sorted(_EXPECTED_EXPORTS)


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
