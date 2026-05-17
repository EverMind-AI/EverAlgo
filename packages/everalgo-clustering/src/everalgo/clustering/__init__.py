"""Online incremental clustering operators — pure-function state-in / state-out.

Public API:

* :func:`cluster_by_geometry` — cosine similarity + time-window filter + threshold; no LLM.
* :func:`cluster_by_llm` — embedding top-K recall + fast-path skip + LLM ranking.
* :class:`Cluster` — frozen value object the caller threads through and persists as list[Cluster].

Caller responsibilities: embedding (EverAlgo accepts any np.ndarray dimension as long as it is
consistent across calls), persistence (Cluster.model_dump() / model_validate()), and
read-modify-write serialisation if multiple writers share a list.
"""

import logging

from everalgo.clustering.algorithm import cluster_by_geometry, cluster_by_llm
from everalgo.clustering.state import Cluster

__all__ = [
    "Cluster",
    "cluster_by_geometry",
    "cluster_by_llm",
]

# ADR-013: per-subpackage logger gets a NullHandler so library use never emits to stderr by default.
logging.getLogger(__name__).addHandler(logging.NullHandler())
