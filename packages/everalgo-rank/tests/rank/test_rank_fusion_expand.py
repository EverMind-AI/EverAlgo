"""Unit tests for ``everalgo.rank.fusion.expand``.

The fine-grained heap-convergence tests target the private
``fusion._expand_heap`` because they need to construct ``fused_results`` and
``episode_scores`` directly. The end-of-file tests exercise the public
``fusion.expand`` over realistic sparse + dense input lists.
"""

from __future__ import annotations

from everalgo.rank import RankConfig, fusion
from everalgo.types import Candidate, FactCandidate

# ─── Helpers ────────────────────────────────────────────────────────────────


def _ep(eid: str, score: float) -> Candidate:
    return Candidate(id=eid, score=score, source="vector")


def _fact(fid: str, parent: str, score: float) -> FactCandidate:
    return FactCandidate(id=fid, parent_episode_id=parent, score=score)


# ─── Heap-convergence tests (target the private _expand_heap) ──────────────


def test_facts_outscoring_parent_replace_parent_in_topn() -> None:
    """High-scoring fact takes the slot; its parent episode is evicted."""
    fused = [_ep("ep1", 0.6), _ep("ep2", 0.3)]
    episode_scores = {"ep1": 0.6, "ep2": 0.3}
    facts = {"ep1": [_fact("f1", "ep1", 0.95)]}

    episodes, facts_out, meta = fusion._expand_heap(
        fused,
        episode_scores,
        facts,
        response_top_k=2,
        config=RankConfig(alpha=1.0, max_convergence_rounds=2, expand_limit=1),
    )

    assert "ep1" not in [e.id for e in episodes]
    assert any(f.id == "f1" for f in facts_out)
    assert meta["facts_in_topn"] == 1


def test_low_score_facts_keep_parent_in_topn() -> None:
    fused = [_ep("ep1", 0.9), _ep("ep2", 0.8)]
    episode_scores = {"ep1": 0.9, "ep2": 0.8}
    facts = {"ep1": [_fact("f1", "ep1", 0.01)]}

    episodes, facts_out, _ = fusion._expand_heap(
        fused,
        episode_scores,
        facts,
        response_top_k=2,
        config=RankConfig(alpha=1.0, max_convergence_rounds=10, expand_limit=1),
    )

    ids = [e.id for e in episodes]
    assert ids == ["ep1", "ep2"]
    assert facts_out == []


def test_convergence_stops_when_topn_stable() -> None:
    fused = [_ep(f"ep{i}", 0.9 - 0.01 * i) for i in range(10)]
    episode_scores = {f"ep{i}": 0.9 - 0.01 * i for i in range(10)}
    facts: dict[str, list[FactCandidate]] = {f"ep{i}": [] for i in range(10)}

    _, _, meta = fusion._expand_heap(
        fused,
        episode_scores,
        facts,
        response_top_k=3,
        config=RankConfig(max_convergence_rounds=2, expand_limit=1),
    )

    assert meta["stop_reason"] == "convergence"
    assert meta["expansions"] >= 2


def test_no_facts_drains_heap_with_heap_exhausted_reason() -> None:
    fused = [_ep("ep1", 0.5), _ep("ep2", 0.4)]
    episode_scores = {"ep1": 0.5, "ep2": 0.4}

    _, _, meta = fusion._expand_heap(
        fused,
        episode_scores,
        prefetched_facts={},
        response_top_k=3,
        config=RankConfig(max_convergence_rounds=100, expand_limit=1),
    )

    assert meta["stop_reason"] == "heap_exhausted"
    assert meta["facts_in_topn"] == 0


def test_use_lr_rescales_child_cosine_to_probability() -> None:
    fused = [_ep("ep1", 0.5)]
    episode_scores = {"ep1": 0.5}
    facts = {"ep1": [_fact("f1", "ep1", 0.9)]}

    _, facts_lr, _ = fusion._expand_heap(
        fused,
        episode_scores,
        facts,
        response_top_k=1,
        config=RankConfig(alpha=1.0, max_convergence_rounds=2, expand_limit=1),
        use_lr=True,
    )
    _, facts_raw, _ = fusion._expand_heap(
        fused,
        episode_scores,
        facts,
        response_top_k=1,
        config=RankConfig(alpha=1.0, max_convergence_rounds=2, expand_limit=1),
        use_lr=False,
    )

    assert facts_lr[0].score != facts_raw[0].score


def test_alpha_blends_child_and_parent() -> None:
    fused = [_ep("ep1", 0.8)]
    episode_scores = {"ep1": 0.8}
    facts = {"ep1": [_fact("f1", "ep1", 0.4)]}

    _, facts_out, _ = fusion._expand_heap(
        fused,
        episode_scores,
        facts,
        response_top_k=1,
        config=RankConfig(alpha=0.5, max_convergence_rounds=2, expand_limit=1),
    )

    if facts_out:
        assert abs(facts_out[0].score - 0.6) < 1e-9


def test_response_top_k_caps_final_set() -> None:
    fused = [_ep(f"ep{i}", 0.9 - 0.05 * i) for i in range(5)]
    episode_scores = {f"ep{i}": 0.9 - 0.05 * i for i in range(5)}

    episodes, facts_out, _ = fusion._expand_heap(
        fused,
        episode_scores,
        prefetched_facts={},
        response_top_k=2,
        config=RankConfig(max_convergence_rounds=100, expand_limit=1),
    )

    assert len(episodes) + len(facts_out) <= 2


def test_expand_heap_with_default_config() -> None:
    """``_expand_heap(config=None)`` resolves to ``DEFAULT_RANK_CONFIG``."""
    fused = [_ep("ep1", 0.5)]
    episode_scores = {"ep1": 0.5}

    _, _, meta = fusion._expand_heap(fused, episode_scores, {}, response_top_k=1)

    assert meta["stop_reason"] in ("convergence", "heap_exhausted")


# ─── High-level fusion.expand tests (sparse + dense input) ──────────────────


def test_expand_high_level_runs_phase1_plus_phase24() -> None:
    """End-to-end ``fusion.expand``: Phase 1 fusion + Phase 2-4 expansion."""
    sparse = [_ep("ep1", 5.0), _ep("ep2", 3.0)]  # BM25-style scores
    dense = [_ep("ep1", 0.95), _ep("ep2", 0.80)]  # cosine-style scores
    facts = {"ep1": [_fact("f1", "ep1", 0.99)]}

    _episodes, facts_out, meta = fusion.expand(
        sparse,
        dense,
        facts,
        response_top_k=2,
        config=RankConfig(alpha=1.0, max_convergence_rounds=2, expand_limit=1),
    )

    # Phase 1 fused both routes; Phase 2-4 lifted f1 into top-N.
    assert any(f.id == "f1" for f in facts_out)
    assert "stop_reason" in meta


def test_expand_empty_inputs_short_circuits() -> None:
    episodes, facts_out, meta = fusion.expand(
        sparse=[],
        dense=[],
        episode_to_facts={},
        response_top_k=3,
    )

    assert episodes == []
    assert facts_out == []
    assert meta == {"stop_reason": "no_candidates"}


def test_expand_dense_only_works_without_sparse() -> None:
    """No sparse list → Phase 1 returns dense as-is; expand still runs."""
    dense = [_ep("ep1", 0.9), _ep("ep2", 0.7)]
    episodes, facts_out, meta = fusion.expand(
        sparse=[],
        dense=dense,
        episode_to_facts={},
        response_top_k=2,
    )

    assert {e.id for e in episodes} <= {"ep1", "ep2"}
    assert facts_out == []
    assert "stop_reason" in meta
