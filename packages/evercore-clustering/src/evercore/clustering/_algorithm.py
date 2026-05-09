"""Cluster operators — cluster_by_geometry / cluster_by_llm stubs."""

from __future__ import annotations

import numpy as np

from evercore.clustering._state import (
    ClusterConfig,
    ClusterId,
    ClusterState,
)
from evercore.llm.protocols import LLMClient

__all__ = ["cluster_by_geometry", "cluster_by_llm"]


async def cluster_by_geometry(
    vector: np.ndarray,
    timestamp: float,
    state: ClusterState,
    *,
    config: ClusterConfig,
) -> tuple[ClusterId, ClusterState]:
    """Pure geometry clustering (cosine + time window + threshold). Stub — TBD."""
    raise NotImplementedError("stub")


async def cluster_by_llm(
    vector: np.ndarray,
    timestamp: float,
    query_text: str,
    state: ClusterState,
    *,
    config: ClusterConfig,
    llm: LLMClient,
    cluster_previews: dict[ClusterId, list[str]],
) -> tuple[ClusterId, ClusterState]:
    """LLM-refined clustering (embedding recall → fast path → LLM decision). Stub — TBD."""
    raise NotImplementedError("stub")
