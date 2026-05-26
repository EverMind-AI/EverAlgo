"""Public-API surface tests for ``everalgo.rank``."""

from __future__ import annotations

import inspect

import pytest


def test_top_level_exports_full_surface() -> None:
    from everalgo.rank import __all__

    assert sorted(__all__) == sorted(
        [
            "CaseRanker",
            "DEFAULT_RANK_CONFIG",
            "EpisodicRanker",
            "FusionMode",
            "RankConfig",
            "SkillRanker",
            "arank",
            "case",
            "episodic",
            "fusion",
            "profile",
            "rank",
            "rerank",
            "skill",
            "weight",
        ]
    )


def test_top_level_dispatch_has_dual_interface() -> None:
    """``arank`` is async; ``rank`` is the sync bridge."""
    import inspect

    from everalgo.rank import arank, rank

    assert inspect.iscoroutinefunction(arank)
    assert callable(rank)
    assert not inspect.iscoroutinefunction(rank)


def test_facade_modules_have_arank_or_rank() -> None:
    from everalgo.rank import case, episodic, profile, skill

    assert inspect.iscoroutinefunction(episodic.arank)
    assert inspect.iscoroutinefunction(case.arank)
    assert inspect.iscoroutinefunction(skill.arank)
    assert callable(profile.rank)
    # episodic / case / skill must also expose sync bridges
    assert callable(episodic.rank)
    assert callable(case.rank)
    assert callable(skill.rank)
    # profile has no async variant
    assert not hasattr(profile, "arank")


def test_fusion_tools_callable() -> None:
    from everalgo.rank import fusion

    assert callable(fusion.rrf)
    assert callable(fusion.lr)
    assert callable(fusion.cosine_to_lr_score)
    assert callable(fusion.score_propagation)


def test_weight_tools_callable_and_export_lr_coefs() -> None:
    from everalgo.rank import weight

    assert callable(weight.weighted_score)
    assert callable(weight.multi_field_weighting)
    assert callable(weight.default_lr_coefs)
    # LRCoefs is a NamedTuple class
    assert hasattr(weight.LRCoefs, "_fields")


def test_rerank_dual_interface() -> None:
    from everalgo.rank import rerank

    assert inspect.iscoroutinefunction(rerank.arerank)
    assert callable(rerank.rerank)


def test_rank_io_schema_importable() -> None:
    from everalgo.types import (
        Candidate,
        FactCandidate,
        RankInput,
        RankOutput,
        ScoredItem,
    )

    rank_input = RankInput(
        query="hi",
        memory_type="episodic",
        dense_candidates=[Candidate(id="x", score=0.5)],
        episode_to_facts={"x": [FactCandidate(id="f", parent_episode_id="x", score=0.9)]},
    )
    assert rank_input.query == "hi"
    assert ScoredItem(id="y", score=0.3, item_type="case").item_type == "case"
    assert RankOutput().items == []


def test_prompts_modules_export_required_constants() -> None:
    from everalgo.rank.prompts.en.case import CASE_RERANK_PROMPT_EN
    from everalgo.rank.prompts.en.episodic import EPISODIC_RERANK_PROMPT_EN
    from everalgo.rank.prompts.en.skill import SKILL_RERANK_PROMPT_EN
    from everalgo.rank.prompts.zh.case import CASE_RERANK_PROMPT_ZH
    from everalgo.rank.prompts.zh.episodic import EPISODIC_RERANK_PROMPT_ZH
    from everalgo.rank.prompts.zh.skill import SKILL_RERANK_PROMPT_ZH

    for p in (
        EPISODIC_RERANK_PROMPT_EN,
        CASE_RERANK_PROMPT_EN,
        SKILL_RERANK_PROMPT_EN,
        EPISODIC_RERANK_PROMPT_ZH,
        CASE_RERANK_PROMPT_ZH,
        SKILL_RERANK_PROMPT_ZH,
    ):
        assert "{query}" in p
        assert "{candidates_json}" in p
        assert "{top_k}" in p


def test_rank_config_importable_and_frozen() -> None:
    from everalgo.rank import DEFAULT_RANK_CONFIG, RankConfig

    cfg = RankConfig()
    # Default is "rrf" — expand (mrag) is opt-in, parallel to lr/rrf rather
    # than the default behaviour. Matches enterprise's method=rrf default.
    assert cfg.fusion_mode == "rrf"
    assert DEFAULT_RANK_CONFIG.fusion_mode == "rrf"
    # frozen → assignment raises
    try:
        cfg.fusion_mode = "rrf"
    except Exception:
        pass
    else:
        raise AssertionError("RankConfig should be frozen")


# ─── Static registry / dispatch ─────────────────────────────────────────────


def test_four_builtin_facades_are_pre_registered() -> None:
    """``_ALGO_REGISTRY`` is a static dict in ``rerank`` populated at import time."""
    from everalgo.rank.rerank import _ALGO_REGISTRY

    assert set(_ALGO_REGISTRY) == {"episodic", "case", "skill", "profile"}


def test_registry_modes_table_matches_per_facade_capabilities() -> None:
    """Each registry entry declares which fusion modes the facade supports."""
    from everalgo.rank.rerank import _ALGO_REGISTRY

    assert _ALGO_REGISTRY["case"].modes == ("rrf", "lr", "vector_anchored")
    assert _ALGO_REGISTRY["skill"].modes == ("rrf", "lr")
    assert _ALGO_REGISTRY["episodic"].modes == ("rrf", "lr", "mrag")
    # Profile does not use fusion_mode at all.
    assert _ALGO_REGISTRY["profile"].modes == ()


async def test_invalid_fusion_mode_for_facade_raises() -> None:
    """``mrag`` is only allowed for episodic; case/skill raise ValueError."""
    import pytest as _pytest  # local alias to avoid clashing with outer pytest import

    from everalgo.rank import RankConfig, case
    from everalgo.types import RankInput

    with _pytest.raises(ValueError, match="not supported"):
        await case.arank(
            RankInput(query="q", memory_type="case"),
            config=RankConfig(fusion_mode="mrag"),
        )


async def test_vector_anchored_rejected_by_non_case_facades() -> None:
    """``vector_anchored`` is registered only for case; episodic/skill must refuse it."""
    import pytest as _pytest

    from everalgo.rank import RankConfig, episodic, skill
    from everalgo.types import RankInput

    with _pytest.raises(ValueError, match="not supported"):
        await episodic.arank(
            RankInput(query="q", memory_type="episodic"),
            config=RankConfig(fusion_mode="vector_anchored"),
        )

    with _pytest.raises(ValueError, match="not supported"):
        await skill.arank(
            RankInput(query="q", memory_type="skill"),
            config=RankConfig(fusion_mode="vector_anchored"),
        )


async def test_arank_dispatches_by_memory_type() -> None:
    from everalgo.rank import arank
    from everalgo.types import Candidate, RankInput

    rank_input = RankInput(
        query="q",
        memory_type="profile",
        dense_candidates=[Candidate(id="p1", score=0.9, source="vector")],
        top_k=1,
    )
    out = await arank(rank_input)

    assert len(out.items) == 1
    assert out.items[0].item_type == "profile"


async def test_arank_unknown_memory_type_raises_keyerror() -> None:
    """memory_type is a Literal so we bypass pydantic via model_copy to test the dispatcher."""
    from everalgo.rank import arank
    from everalgo.types import RankInput

    rank_input = RankInput(query="q", memory_type="case")
    bogus = rank_input.model_copy(update={"memory_type": "no_such_type"})

    with pytest.raises(KeyError):
        await arank(bogus)


def test_sync_rank_dispatch_bridges_to_async() -> None:
    """``rank`` (sync) drives the same dispatch from a non-event-loop context."""
    from everalgo.rank import rank
    from everalgo.types import Candidate, RankInput

    rank_input = RankInput(
        query="q",
        memory_type="profile",
        dense_candidates=[Candidate(id="p1", score=0.9, source="vector")],
        top_k=1,
    )
    out = rank(rank_input)

    assert len(out.items) == 1
    assert out.items[0].item_type == "profile"
