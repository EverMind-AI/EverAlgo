"""Cluster state value objects — frozen dataclasses, caller persists."""

from __future__ import annotations

from typing import Any

import numpy as np  # noqa: TC002  # pydantic resolves field annotations at runtime
from pydantic import BaseModel, ConfigDict, Field

ClusterId = str


class ClusterState(BaseModel):
    """Online incremental K-means accumulated state (centroid / count / last_ts).

    Frozen value object — assign() returns new instance, never mutates. Stub — assign() body TBD.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    centroids: dict[ClusterId, np.ndarray] = Field(default_factory=dict)
    counts: dict[ClusterId, int] = Field(default_factory=dict)
    last_ts: dict[ClusterId, float] = Field(default_factory=dict)

    @classmethod
    def empty(cls) -> ClusterState:
        return cls()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClusterState:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")

    def to_dict(self) -> dict[str, Any]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")

    def assign(
        self,
        cluster_id: ClusterId | None,
        vector: np.ndarray,
        timestamp: float,
    ) -> tuple[ClusterId, ClusterState]:
        """Stub: returns placeholder."""
        raise NotImplementedError("stub")


class ClusterConfig(BaseModel):
    """Cluster threshold bundle (caller configures once at startup)."""

    model_config = ConfigDict(frozen=True)

    threshold: float = 0.65
    time_window_days: float = 7.0
    k_candidates: int = 30
    llm_skip_threshold: float = 0.85


class Candidate(BaseModel):
    """Candidate cluster (output of _find_candidates)."""

    model_config = ConfigDict(frozen=True)

    cluster_id: ClusterId
    similarity: float
