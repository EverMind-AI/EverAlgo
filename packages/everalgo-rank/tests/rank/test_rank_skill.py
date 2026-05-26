"""Unit tests for ``everalgo.rank.skill``."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from everalgo.llm.errors import LLMError
from everalgo.rank import RankConfig, skill
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate, RankInput, RankOutput, ScoredItem

if TYPE_CHECKING:
    from everalgo.llm.types import ChatMessage, ChatResponse


def _mk(sparse: list[Candidate], dense: list[Candidate], top_k: int = 5) -> RankInput:
    return RankInput(
        query="implement structured logging",
        memory_type="skill",
        sparse_candidates=sparse,
        dense_candidates=dense,
        top_k=top_k,
    )


async def test_arank_returns_skill_items_sorted_by_fusion(
    skill_candidates: list[Candidate],
) -> None:
    """No business-field weighting — order driven entirely by fusion."""
    out = await skill.arank(
        _mk(sparse=[], dense=skill_candidates, top_k=3),
        config=RankConfig(fusion_mode="rrf"),
    )

    assert all(it.item_type == "skill" for it in out.items)
    scores = [it.score for it in out.items]
    assert scores == sorted(scores, reverse=True)


async def test_arank_passes_business_fields_through_metadata(
    skill_candidates: list[Candidate],
) -> None:
    """maturity_score / confidence stay in metadata for downstream use."""
    out = await skill.arank(_mk(sparse=[], dense=skill_candidates, top_k=3))

    for item in out.items:
        # Each skill_candidates entry has both business fields in metadata.
        assert "maturity_score" in item.metadata
        assert "confidence" in item.metadata


async def test_arank_empty_input_short_circuits() -> None:
    out = await skill.arank(_mk([], []))
    assert out.items == []
    assert out.metadata.get("stop_reason") == "no_candidates"


def test_sync_bridge_is_callable(skill_candidates: list[Candidate]) -> None:
    out = skill.rank(_mk(sparse=[], dense=skill_candidates, top_k=2))
    assert len(out.items) <= 2


# ─── averify (post-rerank LLM relevance verification) ───────────────────────


def _verify_response(scores: dict[int, float]) -> str:
    """Build a JSON-encoded ``_VerifyResponse`` payload for FakeLLMClient scripted mode."""
    return json.dumps({"results": [{"index": idx, "score": score, "reason": "test"} for idx, score in scores.items()]})


def _ranked(*items: tuple[str, float]) -> RankOutput:
    """Build a RankOutput of skill ScoredItems from ``(id, score)`` pairs."""
    return RankOutput(
        items=[
            ScoredItem(
                id=item_id,
                score=score,
                item_type="skill",
                metadata={"name": f"name-{item_id}", "description": f"desc-{item_id}", "content": "x"},
            )
            for item_id, score in items
        ]
    )


async def test_averify_filters_items_below_threshold() -> None:
    """Items with LLM-assigned score < threshold are dropped — matches enterprise hard-cut at 0.4."""
    ranked = _ranked(("s1", 0.9), ("s2", 0.8), ("s3", 0.7))
    fake = FakeLLMClient(responses=[_verify_response({0: 0.85, 1: 0.30, 2: 0.55})])

    out = await skill.averify(ranked, query="q", llm=fake, threshold=0.4)

    ids = [it.id for it in out.items]
    assert ids == ["s1", "s3"]  # s2 (0.30) dropped, sorted desc by new score
    assert out.metadata["verified"] is True
    assert out.metadata["verify_threshold"] == 0.4
    assert out.metadata["verify_dropped"] == 1


async def test_averify_overwrites_score_and_preserves_pre_verify_score() -> None:
    ranked = _ranked(("s1", 0.9), ("s2", 0.5))
    fake = FakeLLMClient(responses=[_verify_response({0: 0.85, 1: 0.95})])

    out = await skill.averify(ranked, query="q", llm=fake, threshold=0.0)

    by_id = {it.id: it for it in out.items}
    # Scores overwritten with LLM verdicts; original score stashed in metadata.
    assert by_id["s1"].score == 0.85
    assert by_id["s1"].metadata["pre_verify_score"] == 0.9
    assert by_id["s2"].score == 0.95
    assert by_id["s2"].metadata["pre_verify_score"] == 0.5


async def test_averify_sorts_descending_by_llm_score() -> None:
    """Output is sorted by the new LLM score, not the input ordering."""
    ranked = _ranked(("s1", 0.9), ("s2", 0.8), ("s3", 0.7))
    fake = FakeLLMClient(responses=[_verify_response({0: 0.50, 1: 0.95, 2: 0.70})])

    out = await skill.averify(ranked, query="q", llm=fake, threshold=0.0)

    assert [it.id for it in out.items] == ["s2", "s3", "s1"]


async def test_averify_graceful_degradation_on_llm_exception() -> None:
    """LLM failure → return input items unchanged (matches enterprise ``_verify_skill_relevance``)."""

    def boom(messages: list[ChatMessage], **_: object) -> ChatResponse:
        raise LLMError("simulated LLM outage")

    ranked = _ranked(("s1", 0.9), ("s2", 0.5))
    fake = FakeLLMClient(handler=boom)

    out = await skill.averify(ranked, query="q", llm=fake, threshold=0.4)

    # Scores and ids passed through unchanged; metadata still records the (failed) verify attempt.
    assert [(it.id, it.score) for it in out.items] == [("s1", 0.9), ("s2", 0.5)]
    assert out.metadata["verified"] is True
    assert out.metadata["verify_dropped"] == 0


async def test_averify_empty_input_returns_empty() -> None:
    fake = FakeLLMClient(responses=[_verify_response({})])

    out = await skill.averify(RankOutput(), query="q", llm=fake)

    assert out.items == []
    # LLM should not have been called on empty input.
    assert fake.call_count == 0


async def test_averify_missing_index_in_llm_response_treated_as_zero() -> None:
    """When the LLM omits a candidate index entirely, treat it as 0.0 and drop under any positive threshold."""
    ranked = _ranked(("s1", 0.9), ("s2", 0.8))
    # Only index 0 is scored; index 1 omitted.
    fake = FakeLLMClient(responses=[_verify_response({0: 0.9})])

    out = await skill.averify(ranked, query="q", llm=fake, threshold=0.4)

    assert [it.id for it in out.items] == ["s1"]
    assert out.metadata["verify_dropped"] == 1


# ─── enable_verify integration ──────────────────────────────────────────────


async def test_arank_enable_verify_runs_verify_stage(
    skill_candidates: list[Candidate],
) -> None:
    """``skill.arank(enable_verify=True, ...)`` runs the verify stage after fusion."""
    # 3 skill_candidates after rrf → 3 ScoredItems; verify keeps first two.
    fake = FakeLLMClient(responses=[_verify_response({0: 0.9, 1: 0.6, 2: 0.1})])

    out = await skill.arank(
        _mk(sparse=[], dense=skill_candidates, top_k=3),
        config=RankConfig(fusion_mode="rrf"),
        llm=fake,
        enable_verify=True,
    )

    assert fake.call_count == 1
    assert out.metadata.get("verified") is True
    assert out.metadata.get("verify_dropped") == 1
    assert len(out.items) == 2


async def test_arank_enable_verify_without_llm_raises(
    skill_candidates: list[Candidate],
) -> None:
    """Module-level ``arank`` needs an LLM client when ``enable_verify=True``."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="enable_verify=True requires llm"):
        await skill.arank(
            _mk(sparse=[], dense=skill_candidates, top_k=3),
            config=RankConfig(fusion_mode="rrf"),
            enable_verify=True,
        )


async def test_arank_disable_verify_skips_stage(
    skill_candidates: list[Candidate],
) -> None:
    """``enable_verify=False`` (default) leaves output untouched by verify."""
    fake = FakeLLMClient(responses=[])  # would error if invoked

    out = await skill.arank(
        _mk(sparse=[], dense=skill_candidates, top_k=3),
        config=RankConfig(fusion_mode="rrf"),
        llm=fake,
    )

    assert fake.call_count == 0
    assert "verified" not in out.metadata


async def test_skill_ranker_class_enable_verify(
    skill_candidates: list[Candidate],
) -> None:
    """``SkillRanker.arank`` exposes the same ``enable_verify`` flag, using the bound LLM."""
    fake = FakeLLMClient(responses=[_verify_response({0: 0.9, 1: 0.85, 2: 0.85})])
    ranker = skill.SkillRanker(llm=fake)

    out = await ranker.arank(
        _mk(sparse=[], dense=skill_candidates, top_k=3),
        config=RankConfig(fusion_mode="rrf"),
        enable_verify=True,
    )

    assert fake.call_count == 1
    assert out.metadata.get("verified") is True
    assert len(out.items) == 3


def test_verify_not_exposed_on_non_skill_facades() -> None:
    """``enable_verify`` is a skill-only kwarg; case / episodic facades must refuse it at signature level."""
    import inspect

    from everalgo.rank import case, episodic

    case_params = inspect.signature(case.arank).parameters
    episodic_params = inspect.signature(episodic.arank).parameters
    skill_params = inspect.signature(skill.arank).parameters

    assert "enable_verify" not in case_params
    assert "enable_verify" not in episodic_params
    assert "enable_verify" in skill_params  # sanity
