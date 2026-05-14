"""``ClusterState`` value-object tests — serialisation roundtrip + frozen invariants."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from everalgo.clustering import ClusterState


def _seeded_state() -> ClusterState:
    return ClusterState(
        centroids={
            "cluster_000": np.array([1.0, 0.0], dtype=np.float32),
            "cluster_001": np.array([0.0, 1.0], dtype=np.float32),
        },
        counts={"cluster_000": 3, "cluster_001": 1},
        last_ts={"cluster_000": 1_700_000_000_000, "cluster_001": 1_700_000_100_000},
        next_idx=2,
    )


def test_to_from_dict_roundtrip_preserves_all_fields() -> None:
    state = _seeded_state()

    rebuilt = ClusterState.from_dict(state.to_dict())

    assert rebuilt.next_idx == state.next_idx
    assert rebuilt.counts == state.counts
    assert rebuilt.last_ts == state.last_ts
    assert set(rebuilt.centroids) == set(state.centroids)
    for cid in state.centroids:
        np.testing.assert_array_equal(rebuilt.centroids[cid], state.centroids[cid])


def test_from_dict_materialises_centroids_as_float32() -> None:
    rebuilt = ClusterState.from_dict(
        {
            "centroids": {"cluster_000": [0.5, 0.5, 0.5]},
            "counts": {"cluster_000": 1},
            "last_ts": {"cluster_000": 1_700_000_000_000},
            "next_idx": 1,
        }
    )

    assert rebuilt.centroids["cluster_000"].dtype == np.float32


def test_from_dict_tolerates_missing_fields() -> None:
    rebuilt = ClusterState.from_dict({})

    assert rebuilt.centroids == {}
    assert rebuilt.counts == {}
    assert rebuilt.last_ts == {}
    assert rebuilt.next_idx == 0


def test_state_is_frozen_attribute_reassignment_raises() -> None:
    state = ClusterState.empty()

    with pytest.raises(ValidationError):
        state.next_idx = 99  # type: ignore[misc]
