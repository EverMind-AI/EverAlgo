"""Cluster value-object tests — frozen invariants + basic field access."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from everalgo.clustering import Cluster


def test_cluster_default_count_is_one() -> None:
    c = Cluster(centroid=np.array([1.0, 0.0], dtype=np.float32), last_ts=1_700_000_000_000)
    assert c.count == 1


def test_cluster_preview_defaults_to_empty() -> None:
    c = Cluster(centroid=np.array([1.0, 0.0], dtype=np.float32), last_ts=1_700_000_000_000)
    assert c.preview == []


def test_cluster_is_frozen_raises_on_assignment() -> None:
    c = Cluster(centroid=np.array([1.0, 0.0], dtype=np.float32), last_ts=1_700_000_000_000)
    with pytest.raises(ValidationError):
        c.count = 99  # type: ignore[misc]


def test_cluster_arbitrary_ndarray_allowed() -> None:
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    c = Cluster(centroid=vec, last_ts=0)
    np.testing.assert_array_equal(c.centroid, vec)


def test_cluster_with_preview() -> None:
    c = Cluster(centroid=np.array([1.0], dtype=np.float32), last_ts=0, preview=["a", "b"])
    assert c.preview == ["a", "b"]


def test_cluster_members_defaults_to_empty() -> None:
    c = Cluster(centroid=np.array([1.0, 0.0], dtype=np.float32), last_ts=1_700_000_000_000)
    assert c.members == []


def test_cluster_members_roundtrip() -> None:
    c = Cluster(centroid=np.array([1.0, 0.0], dtype=np.float32), last_ts=0, members=["id_a", "id_b"])
    assert c.members == ["id_a", "id_b"]
