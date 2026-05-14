"""Behaviour tests for :func:`everalgo.clustering.cluster_by_geometry`."""

from __future__ import annotations

import numpy as np

from everalgo.clustering import ClusterConfig, ClusterState, cluster_by_geometry

_BASE_TS_MS = 1_700_000_000_000  # 2023-11-14, far enough from epoch to be realistic.
_DAY_MS = 86_400_000


async def test_empty_state_creates_first_cluster() -> None:
    state = ClusterState.empty()
    vector = np.array([1.0, 0.0], dtype=np.float32)

    cid, new_state = await cluster_by_geometry(vector, _BASE_TS_MS, state, config=ClusterConfig())

    assert cid == "cluster_000"
    assert new_state.next_idx == 1
    assert new_state.counts == {"cluster_000": 1}
    assert new_state.last_ts == {"cluster_000": _BASE_TS_MS}
    np.testing.assert_array_equal(new_state.centroids["cluster_000"], vector)


async def test_above_threshold_assigns_to_existing_cluster() -> None:
    seed = ClusterState(
        centroids={"cluster_000": np.array([1.0, 0.0], dtype=np.float32)},
        counts={"cluster_000": 1},
        last_ts={"cluster_000": _BASE_TS_MS},
        next_idx=1,
    )
    # cosine([0.99, 0.01], [1,0]) ≈ 0.9999 > 0.65 threshold.
    vector = np.array([0.99, 0.01], dtype=np.float32)

    cid, new_state = await cluster_by_geometry(vector, _BASE_TS_MS + 1000, seed, config=ClusterConfig())

    assert cid == "cluster_000"
    assert new_state.counts["cluster_000"] == 2
    assert new_state.next_idx == 1


async def test_below_threshold_creates_new_cluster() -> None:
    seed = ClusterState(
        centroids={"cluster_000": np.array([1.0, 0.0], dtype=np.float32)},
        counts={"cluster_000": 1},
        last_ts={"cluster_000": _BASE_TS_MS},
        next_idx=1,
    )
    # cosine([1,0], [0,1]) = 0 < 0.65 threshold → new cluster.
    vector = np.array([0.0, 1.0], dtype=np.float32)

    cid, new_state = await cluster_by_geometry(vector, _BASE_TS_MS + 1000, seed, config=ClusterConfig())

    assert cid == "cluster_001"
    assert new_state.next_idx == 2
    assert set(new_state.counts) == {"cluster_000", "cluster_001"}


async def test_time_window_excludes_old_clusters() -> None:
    old_ts = _BASE_TS_MS - 10 * _DAY_MS  # 10 days ago > 7-day window.
    seed = ClusterState(
        centroids={"cluster_000": np.array([1.0, 0.0], dtype=np.float32)},
        counts={"cluster_000": 1},
        last_ts={"cluster_000": old_ts},
        next_idx=1,
    )
    # Cosine match would be perfect, but time window kicks the candidate out → new cluster.
    vector = np.array([1.0, 0.0], dtype=np.float32)

    cid, new_state = await cluster_by_geometry(vector, _BASE_TS_MS, seed, config=ClusterConfig())

    assert cid == "cluster_001"
    assert new_state.next_idx == 2


async def test_centroid_increment_matches_online_kmeans_formula() -> None:
    # Seed: count=2, centroid=[1.0, 0.0]. New vector [0.0, 1.0] should average to ([2,0]+[0,1])/3 = [0.666, 0.333].
    seed = ClusterState(
        centroids={"cluster_000": np.array([1.0, 0.0], dtype=np.float32)},
        counts={"cluster_000": 2},
        last_ts={"cluster_000": _BASE_TS_MS},
        next_idx=1,
    )
    # Use a high-similarity vector to ensure assignment to cluster_000, but verify the *update* math.
    # Pick a vector close enough to the centroid that we land above threshold but still nudges it.
    vector = np.array([0.95, 0.05], dtype=np.float32)

    _cid, new_state = await cluster_by_geometry(vector, _BASE_TS_MS, seed, config=ClusterConfig())

    expected = (np.array([1.0, 0.0], dtype=np.float32) * 2 + vector) / 3
    np.testing.assert_allclose(new_state.centroids["cluster_000"], expected, rtol=1e-5)
    assert new_state.counts["cluster_000"] == 3


async def test_old_state_unchanged_after_call() -> None:
    seed = ClusterState.empty()
    vector = np.array([1.0, 0.0], dtype=np.float32)

    _cid, _new = await cluster_by_geometry(vector, _BASE_TS_MS, seed, config=ClusterConfig())

    assert seed.centroids == {}
    assert seed.counts == {}
    assert seed.last_ts == {}
    assert seed.next_idx == 0


async def test_last_ts_uses_max_not_clobber() -> None:
    # Existing ts is newer than the incoming event; assign should keep the larger of the two.
    later_ts = _BASE_TS_MS + 5_000
    seed = ClusterState(
        centroids={"cluster_000": np.array([1.0, 0.0], dtype=np.float32)},
        counts={"cluster_000": 1},
        last_ts={"cluster_000": later_ts},
        next_idx=1,
    )
    vector = np.array([0.99, 0.01], dtype=np.float32)

    _cid, new_state = await cluster_by_geometry(vector, _BASE_TS_MS, seed, config=ClusterConfig())

    assert new_state.last_ts["cluster_000"] == later_ts
