"""cluster_by_geometry — online incremental cosine-similarity clustering.

Demonstrates the pure-function stateless API:

* Two vectors with cosine similarity ≈ 0.9998 (nearly identical direction) land
  in the **same** cluster (index 0).
* A third vector orthogonal to the first two mints a **new** cluster (index 1).

No LLM is involved — geometry-only path.

Run:
    uv run python examples/02_clustering_geometry.py
"""

from __future__ import annotations

import numpy as np

from everalgo.clustering import Cluster, cluster_by_geometry

# ---------------------------------------------------------------------------
# Three deterministic vectors
# ---------------------------------------------------------------------------

_VEC_A: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float32)
_VEC_B: np.ndarray = np.array([0.98, 0.02, 0.0], dtype=np.float32)  # cosine ≈ 0.9998 → same cluster
_VEC_C: np.ndarray = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # orthogonal → new cluster

# Timestamps (Unix-ms).  All within the default 7-day time window.
_TS_A: int = 1_700_000_000_000
_TS_B: int = 1_700_000_001_000
_TS_C: int = 1_700_000_002_000


def _mint_cluster(vec: np.ndarray, ts: int, cid: str, entity_id: str) -> Cluster:
    """Build a size-1 Cluster with a caller-assigned id and entity id in members."""
    return Cluster(id=cid, centroid=vec, last_ts=ts, members=[entity_id])


def main() -> None:
    """Cluster three vectors geometrically and print cluster assignments and snapshots."""
    clusters: list[Cluster] = []
    next_id = 0

    new_a = Cluster(centroid=_VEC_A, last_ts=_TS_A, members=["entity_0"])
    result_a = cluster_by_geometry(new_a, clusters)
    assert result_a is None
    cid_a = f"cid_{next_id}"
    next_id += 1
    clusters.append(_mint_cluster(_VEC_A, _TS_A, cid_a, "entity_0"))

    new_b = Cluster(centroid=_VEC_B, last_ts=_TS_B, members=["entity_1"])
    result_b = cluster_by_geometry(new_b, clusters)
    if result_b is None:
        cid_b = f"cid_{next_id}"
        next_id += 1
        clusters.append(_mint_cluster(_VEC_B, _TS_B, cid_b, "entity_1"))
    else:
        assert result_b.id is not None
        cid_b = result_b.id
        idx_b = next(i for i, c in enumerate(clusters) if c.id == cid_b)
        clusters[idx_b] = result_b

    new_c = Cluster(centroid=_VEC_C, last_ts=_TS_C, members=["entity_2"])
    result_c = cluster_by_geometry(new_c, clusters)
    if result_c is None:
        cid_c = f"cid_{next_id}"
        next_id += 1
        clusters.append(_mint_cluster(_VEC_C, _TS_C, cid_c, "entity_2"))
    else:
        assert result_c.id is not None
        cid_c = result_c.id
        idx_c = next(i for i, c in enumerate(clusters) if c.id == cid_c)
        clusters[idx_c] = result_c

    print(f"vector A → cluster id {cid_a!r}")
    print(f"vector B → cluster id {cid_b!r}  (same direction as A; expected cid_0)")
    print(f"vector C → cluster id {cid_c!r}  (orthogonal to A; expected cid_1)")
    print()
    print(f"total clusters : {len(clusters)}  (2 distinct clusters minted)")
    print(f"cluster counts : {[c.count for c in clusters]}")
    print(f"cluster members: {[c.members for c in clusters]}  (entity ids appended on merge)")

    assert cid_a == cid_b, "A and B should share a cluster"
    assert cid_a != cid_c, "C should be in a different cluster"
    assert len(clusters) == 2


if __name__ == "__main__":
    main()
