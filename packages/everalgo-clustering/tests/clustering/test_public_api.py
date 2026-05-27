"""Surface tests for ``everalgo.clustering`` public API."""

from __future__ import annotations

import inspect


def test_top_level_exports() -> None:
    from everalgo.clustering import __all__

    assert sorted(__all__) == sorted(
        [
            "Cluster",
            "cluster_by_geometry",
            "cluster_by_llm",
        ]
    )


def test_cluster_instantiates_with_required_fields() -> None:
    import numpy as np

    from everalgo.clustering import Cluster

    c = Cluster(centroid=np.array([1.0, 0.0], dtype=np.float32), last_ts=0)
    assert c.count == 1
    assert c.preview == []


def test_cluster_function_async_contract() -> None:
    from everalgo.clustering import cluster_by_geometry, cluster_by_llm

    # cluster_by_geometry is sync pure-compute (no I/O); cluster_by_llm is async (calls an LLM).
    assert not inspect.iscoroutinefunction(cluster_by_geometry)
    assert inspect.iscoroutinefunction(cluster_by_llm)
