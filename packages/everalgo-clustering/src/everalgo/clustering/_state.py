"""Cluster state value objects — frozen pydantic models, caller persists.

Algorithm shape
---------------
* :class:`ClusterState` is the accumulated state of an online incremental K-means run:
  ``centroids`` (each cluster's mean vector), ``counts`` (per-cluster event count, needed for the
  ``(C*n + v)/(n+1)`` centroid update), ``last_ts`` (per-cluster last update time in **ms epoch int** —
  aligned with ``MemCell.timestamp`` / ``Episode.timestamp``), plus a monotonic ``next_idx`` used to
  mint new cluster ids ``cluster_NNN``.
* Frozen value object — :meth:`_assign` never mutates ``self``; it returns ``(cluster_id, new_state)``.
  Algorithm callers (``cluster_by_geometry`` / ``cluster_by_llm``) thread state through; on exception the
  caller still holds the un-mutated original (transactional safety, per design.md §6.4).
* Caller owns persistence — load via :meth:`from_dict`, save via :meth:`to_dict`, choose any backing
  store (MongoDB / Redis / file). EverAlgo never does I/O.

Deviations from docs.md §6.2
----------------------------
* ``last_ts`` is **int ms**, not ``float`` seconds — aligns with the rest of EverAlgo's time convention
  (``MemCell.timestamp`` / ``Episode.timestamp`` are ms int).
* Adds a fourth field ``next_idx: int`` instead of reverse-deriving it from ``max(centroids.keys())+1``.
  Reverse derivation is a footgun when callers ever drop a middle cluster: the derived value falls
  below the historical maximum and the next ``_assign`` reuses an existing id. opensource (cluster_manager
  /manager.py:46) keeps ``next_cluster_idx`` as a tracked field for the same reason.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

ClusterId = str
"""Cluster identifier. Auto-minted ids follow ``cluster_NNN`` (3-digit zero-padded)."""


class ClusterState(BaseModel):
    """Online incremental K-means accumulated state.

    Four fields are needed for the algorithm:

    - ``centroids`` — mean vector per cluster, compared against new vectors via cosine similarity.
    - ``counts`` — event count per cluster, drives the centroid increment ``(C*n + v)/(n+1)``.
    - ``last_ts`` — last update time (ms epoch int) per cluster, drives ``time_window_days`` filtering
      in :func:`cluster_by_geometry`.
    - ``next_idx`` — monotonic counter for minting new ids; never decreases across :meth:`_assign` calls.

    Frozen — every :meth:`_assign` returns a new instance; pre-existing references remain valid
    snapshots of the prior state.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    centroids: dict[ClusterId, np.ndarray] = Field(default_factory=dict)
    counts: dict[ClusterId, int] = Field(default_factory=dict)
    last_ts: dict[ClusterId, int] = Field(default_factory=dict)
    next_idx: int = 0

    @classmethod
    def empty(cls) -> ClusterState:
        """Return an empty state — no clusters yet, ``next_idx`` = 0."""
        return cls()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ClusterState:
        """Rebuild a :class:`ClusterState` from its :meth:`to_dict` payload.

        Vectors are re-materialised as ``np.float32`` arrays. Missing fields default to empty / zero.
        """
        centroids_raw = cast("Mapping[str, Sequence[float]]", d.get("centroids") or {})
        centroids: dict[ClusterId, np.ndarray] = {
            cid: np.asarray(vec, dtype=np.float32) for cid, vec in centroids_raw.items()
        }
        counts_raw = cast("Mapping[str, int]", d.get("counts") or {})
        counts: dict[ClusterId, int] = {cid: int(c) for cid, c in counts_raw.items()}
        last_ts_raw = cast("Mapping[str, int]", d.get("last_ts") or {})
        last_ts: dict[ClusterId, int] = {cid: int(ts) for cid, ts in last_ts_raw.items()}
        next_idx = int(d.get("next_idx", 0))
        return cls(centroids=centroids, counts=counts, last_ts=last_ts, next_idx=next_idx)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-able dict — numpy arrays become plain ``list[float]``.

        Round-trip with :meth:`from_dict` reconstructs an equivalent state.
        """
        return {
            "centroids": {cid: vec.tolist() for cid, vec in self.centroids.items()},
            "counts": dict(self.counts),
            "last_ts": dict(self.last_ts),
            "next_idx": self.next_idx,
        }

    def _assign(
        self,
        cluster_id: ClusterId | None,
        vector: np.ndarray,
        timestamp_ms: int,
    ) -> tuple[ClusterId, ClusterState]:
        """Assign ``vector`` to ``cluster_id`` (or mint a new id when ``None``); return ``(id, new_state)``.

        Pure: ``self`` is never mutated. Centroid update uses the standard online K-means formula
        ``C' = (C*n + v) / (n+1)``; ``last_ts'`` = ``max(prev, timestamp_ms)``.

        ``cluster_id`` semantics:
            - ``None`` — mint ``cluster_{next_idx:03d}`` and bump ``next_idx``.
            - existing id — incremental update; centroid / count / last_ts shift forward.
            - unknown id (not in ``self.counts``) — treated as a new cluster with the caller-supplied id;
              ``next_idx`` is **not** bumped. Caller is responsible for avoiding collisions with future
              auto-minted ids (eg. don't pass ``cluster_999`` when ``next_idx == 500``).

        Private — only the algorithm functions in :mod:`everalgo.clustering._algorithm` call this; callers
        of the public API never touch it directly.
        """
        if cluster_id is None:
            new_cid = f"cluster_{self.next_idx:03d}"
            new_next_idx = self.next_idx + 1
        else:
            new_cid = cluster_id
            new_next_idx = self.next_idx

        existing_count = self.counts.get(new_cid, 0)
        vec_f32 = vector.astype(np.float32, copy=False)
        if existing_count == 0:
            new_centroid = vec_f32.copy()
            new_count = 1
        else:
            n = existing_count
            existing_centroid = self.centroids[new_cid].astype(np.float32, copy=False)
            new_centroid = ((existing_centroid * n + vec_f32) / (n + 1)).astype(np.float32, copy=False)
            new_count = n + 1

        prev_ts = self.last_ts.get(new_cid, timestamp_ms)
        new_ts = max(prev_ts, timestamp_ms)

        return new_cid, self.model_copy(
            update={
                "centroids": {**self.centroids, new_cid: new_centroid},
                "counts": {**self.counts, new_cid: new_count},
                "last_ts": {**self.last_ts, new_cid: new_ts},
                "next_idx": new_next_idx,
            }
        )


class ClusterConfig(BaseModel):
    """Cluster threshold bundle — caller configures once at startup, EverAlgo never mutates.

    Attributes
    ----------
    threshold
        Geometry decision threshold; cosine similarity ``>=`` this value assigns to the candidate top-1
        cluster, otherwise a new cluster is minted. Default ``0.65`` matches opensource
        ``similarity_threshold``.
    time_window_days
        :func:`cluster_by_geometry` only — clusters whose ``last_ts`` is older than this gap are excluded
        from candidacy. Default ``7.0`` matches opensource ``max_time_gap_days``.
    k_candidates
        :func:`cluster_by_llm` only — top-K nearest clusters retrieved from the embedding recall stage
        before LLM ranking. Default ``30`` matches opensource ``llm_top_k_clusters``.
    llm_skip_threshold
        :func:`cluster_by_llm` only — if the top-1 embedding similarity ``>=`` this value, skip the LLM
        call and assign to top-1 directly (fast path). Default ``0.85`` matches opensource.
    """

    model_config = ConfigDict(frozen=True)

    threshold: float = 0.65
    time_window_days: float = 7.0
    k_candidates: int = 30
    llm_skip_threshold: float = 0.85
