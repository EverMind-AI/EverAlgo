"""Rank I/O contracts — Candidate / FactCandidate / RankInput / ScoredItem / RankOutput.

Spec: ``docs/superpowers/specs/2026-05-11-everalgo-rank-design.md`` §4.1.

Design contract: ``Rank`` performs **no storage I/O**. The caller's upstream
recall layer pre-fetches multi-route candidates plus cross-memory linkage
(e.g. Episode → AtomicFact) and passes them in via ``RankInput``. The ranker
reads only the dataclass; it never talks to a database.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Candidate(BaseModel):
    """A single recall candidate produced by the upstream recall layer and consumed by the ranker.

    Business fields (``quality_score`` / ``maturity_score`` / ``confidence`` /
    raw text / ``parent_episode_id`` / ...) live in ``metadata``. Each ranker
    facade reads only the keys it cares about.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    score: float
    source: Literal["keyword", "vector", "other"] = "other"
    metadata: dict[str, Any] = Field(default_factory=dict)


class FactCandidate(BaseModel):
    """AtomicFact candidate used by the episodic facade for ep→fact expansion.

    ``score`` is cosine similarity from a vector store, pre-fetched by the caller's recall layer.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    parent_episode_id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RankInput(BaseModel):
    """Recall-stage input passed to a ranker facade.

    Three groups of fields:

    1. Multi-route recall candidates (sparse + dense). Profile uses dense only;
       the other three facades use both.
    2. Pre-fetched cross-memory linkage (currently only Episode → AtomicFact
       for episodic). Other facades leave it empty.
    3. Business parameters (``top_k`` / ``radius``). The ranker does not read
       environment variables; the caller passes everything explicitly.
    """

    model_config = ConfigDict(extra="forbid")

    query: str
    memory_type: Literal["episodic", "profile", "case", "skill"]

    sparse_candidates: list[Candidate] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    dense_candidates: list[Candidate] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]

    episode_to_facts: dict[str, list[FactCandidate]] = Field(default_factory=dict)

    top_k: int = 10
    radius: float | None = None


class ScoredItem(BaseModel):
    """A single ranked output item.

    Episodic may produce a mix of episode and atomic_fact items (expansion
    decides); other facades produce a single ``item_type``. The caller reshapes
    the list back into the response DTO it serves to clients.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    score: float
    item_type: Literal["episode", "atomic_fact", "case", "skill", "profile"]
    metadata: dict[str, Any] = Field(default_factory=dict)
    parent_episode_id: str | None = None


class RankOutput(BaseModel):
    """Ranker output — a sorted list of ScoredItem plus optional debug metadata."""

    model_config = ConfigDict(extra="forbid")

    items: list[ScoredItem] = Field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    metadata: dict[str, Any] = Field(default_factory=dict)
