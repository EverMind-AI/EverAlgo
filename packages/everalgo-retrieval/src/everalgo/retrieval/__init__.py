"""EverAlgo retrieval facade — hybrid / agentic-wrapper / hierarchical / maxsim / cluster strategies.

Five operators, all taking one or more ``RetrieveFn`` callables and returning ``list[Candidate]``,
so any operator can serve as ``base_retrieve`` to any other — the call graph is caller-owned.

Quick reference:

- ``ahybrid_retrieve(dense_retrieve=, sparse_retrieve=, ...)`` — combiner, dual-route RRF / LR fusion.
- ``amaxsim_retrieve(child_retrieve=, parent_fetch=, ...)`` — combiner, child-first MaxSim aggregation.
- ``ahierarchical_retrieve(parent_dense_retrieve=, parent_sparse_retrieve=, child_retrieve_for_parent=, ...)``
  — combiner, parent-first expand-children (MRAG-style).
- ``acluster_retrieve(base_retrieve=, clusters=, all_docs=, ...)`` — decorator, narrow scope to top-K clusters.
- ``aagentic_retrieve(base_retrieve=, llm=, rerank_fn=, round2_retrieve=, ...)`` — decorator,
  multi-round LLM-guided sufficiency loop.

A ``base_retrieve=`` parameter signals a decorator (wraps an existing ``RetrieveFn``); role-named
parameters (``dense_/sparse_/child_/parent_*``) signal a combiner.

Every async operator has a sync bridge via ``asgiref.async_to_sync`` (e.g., ``maxsim_retrieve``).
"""

import logging

from everalgo.retrieval.agentic import aagentic_retrieve, agentic_retrieve
from everalgo.retrieval.cluster import acluster_retrieve, cluster_retrieve
from everalgo.retrieval.hierarchical import ahierarchical_retrieve, hierarchical_retrieve
from everalgo.retrieval.hybrid import ahybrid_retrieve, hybrid_retrieve
from everalgo.retrieval.maxsim import ParentFetchFn, amaxsim_retrieve, maxsim_retrieve
from everalgo.retrieval.protocols import AgenticDecision, RerankFn, RetrieveFn

__all__ = [
    "AgenticDecision",
    "ParentFetchFn",
    "RerankFn",
    "RetrieveFn",
    "aagentic_retrieve",
    "acluster_retrieve",
    "agentic_retrieve",
    "ahierarchical_retrieve",
    "ahybrid_retrieve",
    "amaxsim_retrieve",
    "cluster_retrieve",
    "hierarchical_retrieve",
    "hybrid_retrieve",
    "maxsim_retrieve",
]

# Library logging setup (ADR-013): NullHandler on each subpackage logger.
logging.getLogger(__name__).addHandler(logging.NullHandler())
