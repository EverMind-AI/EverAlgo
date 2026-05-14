"""Surface tests for ``everalgo.clustering`` public API."""

from __future__ import annotations

import inspect


def test_top_level_exports() -> None:
    from everalgo.clustering import __all__

    assert sorted(__all__) == sorted(
        [
            "ClusterConfig",
            "ClusterState",
            "cluster_by_geometry",
            "cluster_by_llm",
        ]
    )


def test_value_objects_instantiate_with_defaults() -> None:
    from everalgo.clustering import ClusterConfig, ClusterState

    state = ClusterState.empty()
    assert state.centroids == {}
    assert state.counts == {}
    assert state.last_ts == {}
    assert state.next_idx == 0

    config = ClusterConfig()
    assert config.threshold == 0.65
    assert config.time_window_days == 7.0
    assert config.k_candidates == 30
    assert config.llm_skip_threshold == 0.85


def test_cluster_functions_are_async() -> None:
    from everalgo.clustering import cluster_by_geometry, cluster_by_llm

    assert inspect.iscoroutinefunction(cluster_by_geometry)
    assert inspect.iscoroutinefunction(cluster_by_llm)
