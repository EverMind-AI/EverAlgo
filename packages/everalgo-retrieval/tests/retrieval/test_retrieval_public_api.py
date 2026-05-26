"""Lock the public retrieval surface — `__all__` ordering and re-export completeness."""

from __future__ import annotations

import everalgo.retrieval


def test_all_lists_expected_names_alphabetical() -> None:
    assert list(everalgo.retrieval.__all__) == [
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


def test_each_all_name_is_resolvable() -> None:
    for name in everalgo.retrieval.__all__:
        assert hasattr(everalgo.retrieval, name), f"{name} in __all__ but not bound on module"
