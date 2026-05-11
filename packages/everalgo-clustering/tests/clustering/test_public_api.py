"""Stub existence tests for everalgo.clustering."""

import inspect


def test_top_level_exports() -> None:
    from everalgo.clustering import __all__

    assert sorted(__all__) == sorted(
        [
            "Candidate",
            "ClusterConfig",
            "ClusterState",
            "cluster_by_geometry",
            "cluster_by_llm",
        ]
    )


def test_dataclasses_instantiable() -> None:
    from everalgo.clustering import Candidate, ClusterConfig, ClusterState

    assert ClusterState.empty().centroids == {}
    assert ClusterConfig().threshold == 0.65
    assert Candidate(cluster_id="c1", similarity=0.9).cluster_id == "c1"


def test_cluster_functions_async() -> None:
    from everalgo.clustering import cluster_by_geometry, cluster_by_llm

    assert inspect.iscoroutinefunction(cluster_by_geometry)
    assert inspect.iscoroutinefunction(cluster_by_llm)
