"""Tests for everalgo.agent_memory package-level public API."""

import everalgo.agent_memory


def test_four_symbols_exported() -> None:
    """agent_memory exposes 2 Extractors + SkillConfig + 1 boundary re-export."""
    for name in ("AgentCaseExtractor", "AgentMemCellExtractor", "AgentSkillExtractor", "SkillConfig"):
        assert hasattr(everalgo.agent_memory, name)


def test_dunder_all_lists_four_symbols() -> None:
    """__all__ exposes exactly 4 symbols."""
    assert sorted(everalgo.agent_memory.__all__) == sorted(
        ["AgentCaseExtractor", "AgentMemCellExtractor", "AgentSkillExtractor", "SkillConfig"]
    )


def test_extractors_instantiable_without_args() -> None:
    """Both Extractors are stateless callable classes — constructible with no args."""
    from everalgo.agent_memory import AgentCaseExtractor, AgentSkillExtractor

    assert AgentCaseExtractor().__class__.__name__ == "AgentCaseExtractor"
    assert AgentSkillExtractor().__class__.__name__ == "AgentSkillExtractor"


def test_skill_config_defaults() -> None:
    """SkillConfig is a frozen dataclass with the documented defaults."""
    from everalgo.agent_memory import SkillConfig

    cfg = SkillConfig()
    assert cfg.maturity_threshold == 0.6
    assert cfg.retire_confidence == 0.1
    assert cfg.failure_quality_threshold == 0.5
    assert cfg.skip_maturity_scoring is True  # default skips LLM maturity scoring; returns 1.0
    assert cfg.max_case_history == 9
    assert cfg.maturity_trivial_change_ratio == 0.2
    assert cfg.maturity_reeval_change_ratio == 0.4
    assert not hasattr(cfg, "max_skills_in_prompt")  # caller does external top-K filtering
