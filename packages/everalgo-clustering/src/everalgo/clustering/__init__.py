"""Online incremental clustering operators — pure-function state-in / state-out.

Public API
----------
* :func:`cluster_by_geometry` — cosine similarity + time-window filter + threshold; no LLM.
* :func:`cluster_by_llm` — embedding top-K recall + fast path + LLM ranking with geometric fallback.
* :class:`ClusterState` — frozen value object the caller threads through and persists.
* :class:`ClusterConfig` — threshold bundle (caller configures once at startup).

Caller responsibilities (per ``docs/design.md §6.3``): embedding (EverAlgo accepts any np.ndarray
dimension as long as it is consistent across calls), persistence (``ClusterState.to_dict`` /
``from_dict``), pre-fetching ``cluster_previews`` for :func:`cluster_by_llm`, and read-modify-write
serialisation if multiple writers share a state.
"""

import logging

from everalgo.clustering._algorithm import cluster_by_geometry, cluster_by_llm
from everalgo.clustering._state import ClusterConfig, ClusterState

__all__ = [
    "ClusterConfig",
    "ClusterState",
    "cluster_by_geometry",
    "cluster_by_llm",
]

# ADR-013: per-subpackage logger gets a NullHandler so library use never emits to stderr by default.
logging.getLogger(__name__).addHandler(logging.NullHandler())
