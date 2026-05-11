"""Stub existence tests for everalgo.rank."""

import inspect


def test_top_level_exports_seven_modules() -> None:
    from everalgo.rank import __all__

    assert sorted(__all__) == sorted(
        [
            "case",
            "episodic",
            "fusion",
            "profile",
            "rerank",
            "skill",
            "weight",
        ]
    )


def test_facade_modules_have_arank_or_rank() -> None:
    from everalgo.rank import case, episodic, profile, skill

    assert inspect.iscoroutinefunction(episodic.arank)
    assert inspect.iscoroutinefunction(case.arank)
    assert inspect.iscoroutinefunction(skill.arank)
    assert callable(profile.rank)


def test_fusion_tools_callable() -> None:
    from everalgo.rank import fusion

    assert callable(fusion.rrf)
    assert callable(fusion.lr)
    assert callable(fusion.cosine_to_lr_score)
    assert callable(fusion.score_propagation)


def test_weight_tools_callable() -> None:
    from everalgo.rank import weight

    assert callable(weight.weighted_score)
    assert callable(weight.multi_field_weighting)


def test_rerank_dual_interface() -> None:
    from everalgo.rank import rerank

    assert inspect.iscoroutinefunction(rerank.arerank)
    assert callable(rerank.rerank)
