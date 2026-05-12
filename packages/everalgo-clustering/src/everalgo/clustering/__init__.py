"""Online incremental clustering operators — stubs."""

import logging

from everalgo.clustering._algorithm import cluster_by_geometry, cluster_by_llm
from everalgo.clustering._state import Candidate, ClusterConfig, ClusterState

__all__ = [
    "Candidate",
    "ClusterConfig",
    "ClusterState",
    "cluster_by_geometry",
    "cluster_by_llm",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
