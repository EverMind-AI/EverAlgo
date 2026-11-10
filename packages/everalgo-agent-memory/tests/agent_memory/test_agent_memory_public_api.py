"""Tests for everalgo.agent_memory package-level public API."""

import ast
from pathlib import Path

import everalgo.agent_memory
from everalgo.testing.fake_llm import FakeLLMClient

_EXPECTED_EXPORTS = [
    # 3 Extractors
    "AgentCaseExtractor",
    "AgentProfileExtractor",
    "AgentSkillExtractor",
    # everalgo.boundary's public surface, re-exported unchanged
    "BoundaryDetector",
    "Cuts",
    "to_memcells",
    # Optional correctness admission; the host supplies the CaseJudge
    "AgentCaseGate",
    "CaseGateConfig",
    "CaseGateDecision",
    "CaseGateQuestion",
    "CaseJudge",
    "should_extract_skill",
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


def test_boundary_names_are_the_boundary_package_objects() -> None:
    """The product package re-exports ``everalgo.boundary``'s objects themselves, not wrappers or subclasses."""
    import everalgo.boundary

    assert everalgo.agent_memory.BoundaryDetector is everalgo.boundary.BoundaryDetector
    assert everalgo.agent_memory.Cuts is everalgo.boundary.Cuts
    assert everalgo.agent_memory.to_memcells is everalgo.boundary.to_memcells


def test_boundary_module_holds_nothing_but_the_re_export() -> None:
    """``boundary.py`` is a docstring, ``__all__`` and one import: no logic of its own to drift from the class."""
    tree = ast.parse((Path(everalgo.agent_memory.__file__).parent / "boundary.py").read_text(encoding="utf-8"))
    statements = [ast.unparse(node) for node in tree.body[1:]]

    assert statements == [
        "from __future__ import annotations",
        "__all__ = ['BoundaryDetector', 'Cuts', 'to_memcells']",
        "from everalgo.boundary import BoundaryDetector, Cuts, to_memcells",
    ]


def test_agent_boundary_detector_is_gone() -> None:
    assert "AgentBoundaryDetector" not in everalgo.agent_memory.__all__
    assert not hasattr(everalgo.agent_memory, "AgentBoundaryDetector")
