"""Behaviour tests for :func:`everalgo.clustering.cluster_by_geometry`."""

from __future__ import annotations

import numpy as np

from everalgo.clustering import Cluster, cluster_by_geometry

_BASE_TS_MS = 1_700_000_000_000  # 2023-11-14, far enough from epoch to be realistic.
_DAY_MS = 86_400_000


def _c(vec: list[float], ts: int = _BASE_TS_MS, count: int = 1, cid: str | None = None) -> Cluster:
    return Cluster(centroid=np.array(vec, dtype=np.float32), last_ts=ts, count=count, id=cid)


async def test_empty_existing_returns_none() -> None:
    new_c = _c([1.0, 0.0])

    result = await cluster_by_geometry(new_c, [])

    assert result is None


async def test_above_threshold_returns_merged_cluster() -> None:
    existing = [_c([1.0, 0.0], cid="cid_0")]
    # cosine([0.99, 0.01], [1,0]) ≈ 0.9999 > 0.65 threshold.
    new_c = _c([0.99, 0.01], ts=_BASE_TS_MS + 1000)

    result = await cluster_by_geometry(new_c, existing)

    assert result is not None
    assert result.id == "cid_0"
    assert result.count == 2
    assert result.last_ts == _BASE_TS_MS + 1000


async def test_below_threshold_returns_none() -> None:
    existing = [_c([1.0, 0.0])]
    # cosine([1,0], [0,1]) = 0 < 0.65 threshold → new cluster.
    new_c = _c([0.0, 1.0], ts=_BASE_TS_MS + 1000)

    result = await cluster_by_geometry(new_c, existing)

    assert result is None


async def test_time_window_excludes_old_clusters() -> None:
    old_ts = _BASE_TS_MS - 10 * _DAY_MS  # 10 days ago > 7-day window.
    existing = [_c([1.0, 0.0], ts=old_ts)]
    # Cosine match would be perfect, but time window kicks the candidate out → new cluster.
    new_c = _c([1.0, 0.0])

    result = await cluster_by_geometry(new_c, existing)

    assert result is None


async def test_centroid_merges_via_weighted_average() -> None:
    # count=2, centroid=[1.0, 0.0]. New vector [0.95, 0.05] → weighted avg.
    existing = [_c([1.0, 0.0], count=2)]
    new_vec = np.array([0.95, 0.05], dtype=np.float32)
    new_c = Cluster(centroid=new_vec, last_ts=_BASE_TS_MS, count=1)

    result = await cluster_by_geometry(new_c, existing)

    assert result is not None
    expected = (np.array([1.0, 0.0], dtype=np.float32) * 2 + new_vec) / 3
    np.testing.assert_allclose(result.centroid, expected, rtol=1e-5)
    assert result.count == 3


async def test_existing_unchanged_after_call() -> None:
    existing = [_c([1.0, 0.0])]
    original_centroid = existing[0].centroid.copy()
    new_c = _c([0.99, 0.01], ts=_BASE_TS_MS + 1)

    await cluster_by_geometry(new_c, existing)

    # existing list itself and Cluster inside must be unchanged (frozen).
    np.testing.assert_array_equal(existing[0].centroid, original_centroid)
    assert existing[0].count == 1


async def test_last_ts_uses_max() -> None:
    # Existing ts is newer; merge should keep the larger of the two.
    later_ts = _BASE_TS_MS + 5_000
    existing = [_c([1.0, 0.0], ts=later_ts)]
    new_c = _c([0.99, 0.01], ts=_BASE_TS_MS)  # older timestamp

    result = await cluster_by_geometry(new_c, existing)

    assert result is not None
    assert result.last_ts == later_ts


async def test_preview_is_merged_and_capped() -> None:
    existing = [Cluster(centroid=np.array([1.0, 0.0], dtype=np.float32), last_ts=_BASE_TS_MS, count=1, preview=["old"])]
    new_c = Cluster(
        centroid=np.array([0.99, 0.01], dtype=np.float32), last_ts=_BASE_TS_MS + 1, count=1, preview=["new"]
    )

    result = await cluster_by_geometry(new_c, existing, preview_cap=5)

    assert result is not None
    assert result.preview == ["old", "new"]


async def test_pick_best_among_multiple_existing() -> None:
    existing = [
        _c([1.0, 0.0], cid="cid_0"),  # cosine([0.9, 0.1], ...) ≈ 0.994
        _c([0.0, 1.0], cid="cid_1"),  # cosine([0.9, 0.1], ...) ≈ 0.099
    ]
    new_c = _c([0.9, 0.1], ts=_BASE_TS_MS + 1)

    result = await cluster_by_geometry(new_c, existing)

    assert result is not None
    assert result.id == "cid_0"  # best match
    assert result.count == 2


async def test_merge_appends_members() -> None:
    existing = [
        Cluster(
            centroid=np.array([1.0, 0.0], dtype=np.float32),
            last_ts=_BASE_TS_MS,
            members=["a", "b"],
        )
    ]
    new_c = Cluster(
        centroid=np.array([0.99, 0.01], dtype=np.float32),
        last_ts=_BASE_TS_MS + 1,
        members=["c"],
    )

    result = await cluster_by_geometry(new_c, existing)

    assert result is not None
    assert result.members == ["a", "b", "c"]


async def test_none_path_preserves_new_members() -> None:
    # No existing clusters → None returned; caller uses new_c as-is with its members intact.
    new_c = Cluster(
        centroid=np.array([1.0, 0.0], dtype=np.float32),
        last_ts=_BASE_TS_MS,
        members=["entity_x"],
    )

    result = await cluster_by_geometry(new_c, [])

    assert result is None
    assert new_c.members == ["entity_x"]
