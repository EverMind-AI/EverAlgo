"""Rank I/O contracts — schema TBD."""

from typing import Any

from pydantic import BaseModel, Field


class RankInput(BaseModel):
    """Recall-stage input passed to Ranker.

    Stub — schema fields TBD (T1). Will include: sparse/dense candidates, pre-fetched cross-memory linkage
    (e.g. Episode → AtomicFact).
    """

    memory_type: str = Field(default="", description="episodic / case / skill / profile")
    candidates: list[dict[str, Any]] = Field(default_factory=list, description="TBD (T1 review)")


class RankOutput(BaseModel):
    """Ranked memory list — Ranker output.

    Stub — schema fields TBD (T1).
    """

    items: list[dict[str, Any]] = Field(default_factory=list, description="TBD (T1 review)")
