"""Shared fixtures for ``everalgo-rank`` tests."""

from __future__ import annotations

import pytest

from everalgo.types import Candidate, FactCandidate


@pytest.fixture
def dense_candidates() -> list[Candidate]:
    """5 vector-source candidates, descending cosine scores."""
    return [
        Candidate(id="d1", score=0.95, source="vector", metadata={"quality_score": 0.9}),
        Candidate(id="d2", score=0.80, source="vector", metadata={"quality_score": 0.6}),
        Candidate(id="d3", score=0.72, source="vector", metadata={"quality_score": 0.7}),
        Candidate(id="d4", score=0.55, source="vector", metadata={"quality_score": 0.4}),
        Candidate(id="d5", score=0.40, source="vector", metadata={"quality_score": 0.3}),
    ]


@pytest.fixture
def sparse_candidates() -> list[Candidate]:
    """4 keyword-source candidates, descending BM25 scores. Overlaps with dense on d1/d3."""
    return [
        Candidate(id="d1", score=12.5, source="keyword", metadata={"quality_score": 0.9}),
        Candidate(id="d6", score=10.0, source="keyword", metadata={"quality_score": 0.5}),
        Candidate(id="d3", score=8.2, source="keyword", metadata={"quality_score": 0.7}),
        Candidate(id="d7", score=4.1, source="keyword", metadata={"quality_score": 0.2}),
    ]


@pytest.fixture
def skill_candidates() -> list[Candidate]:
    """5 skill candidates with ``maturity_score`` + ``confidence`` metadata."""
    return [
        Candidate(
            id="s1",
            score=0.0,
            source="vector",
            metadata={"maturity_score": 0.9, "confidence": 0.85},
        ),
        Candidate(
            id="s2",
            score=0.0,
            source="vector",
            metadata={"maturity_score": 0.6, "confidence": 0.95},
        ),
        Candidate(
            id="s3",
            score=0.0,
            source="vector",
            metadata={"maturity_score": 0.4, "confidence": 0.3},
        ),
    ]


@pytest.fixture
def episode_to_facts() -> dict[str, list[FactCandidate]]:
    """Pre-fetched ep→fact linkage matching dense_candidates ids d1/d2/d3."""
    return {
        "d1": [
            FactCandidate(id="f11", parent_episode_id="d1", score=0.92),
            FactCandidate(id="f12", parent_episode_id="d1", score=0.88),
        ],
        "d2": [
            FactCandidate(id="f21", parent_episode_id="d2", score=0.50),
        ],
        "d3": [
            FactCandidate(id="f31", parent_episode_id="d3", score=0.78),
            FactCandidate(id="f32", parent_episode_id="d3", score=0.65),
        ],
    }
