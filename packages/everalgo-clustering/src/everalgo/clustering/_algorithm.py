"""Cluster operators — geometry and LLM-refined paths over an online incremental K-means state.

Algorithm shape
---------------
Both are async, stateless, and obey EverAlgo's state-in / state-out contract: take a frozen
:class:`ClusterState`, return ``(cluster_id, new_state)`` without mutating the input. The caller owns
embedding, persistence, and locking — EverAlgo never does I/O.

Inherited from opensource ``cluster_manager/manager.py`` (line 263-539):
    - Centroid increment ``(C*n + v)/(n+1)`` and ``last_ts = max(prev, ts)`` updates (delegated to
      :meth:`ClusterState._assign`).
    - Time window filter on the geometry path: candidates whose ``last_ts`` is more than
      ``time_window_days`` ago are skipped.
    - Threshold decision: top-1 cosine ``>=`` ``config.threshold`` assigns; otherwise a new cluster is
      minted.
    - LLM path's three-stage flow: embedding top-K recall → fast-path skip (top-1
      ``>=`` ``llm_skip_threshold``) → LLM ranking with 3 retries → fallback to top-1 / new on failure.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from everalgo.clustering.prompts.en.cluster import CLUSTER_LLM_ASSIGN_PROMPT
from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.prompts import render_prompt

if TYPE_CHECKING:
    from everalgo.clustering._state import ClusterConfig, ClusterId, ClusterState
    from everalgo.llm.protocols import LLMClient

__all__ = ["cluster_by_geometry", "cluster_by_llm"]

logger = logging.getLogger(__name__)

_MS_PER_DAY = 86_400_000
"""Milliseconds per day; ``time_window_days * _MS_PER_DAY`` is the geometry path's cutoff."""

_MAX_LLM_RETRIES = 3
"""Retries for LLM JSON / schema failures (opensource cluster_manager line 656 uses 3).

Lower than EpisodeExtractor's 5 because clustering has a deterministic geometric fallback — exhausting
LLM retries lands on top-1 / new-cluster, never raises.
"""

_NORM_EPSILON = 1e-9
"""Numerical guard added to vector norms to avoid divide-by-zero (matches opensource)."""


async def cluster_by_geometry(
    vector: np.ndarray,
    timestamp_ms: int,
    state: ClusterState,
    *,
    config: ClusterConfig,
) -> tuple[ClusterId, ClusterState]:
    """Pure geometry clustering: cosine similarity + time-window filter + threshold decision.

    Parameters
    ----------
    vector
        Pre-computed embedding for the new event. EverAlgo does not embed — callers pick the model
        (Qwen3-embedding-4B, OpenAI ``text-embedding-3-small``, etc.). Dimension is arbitrary but must
        be consistent across all calls against the same state.
    timestamp_ms
        Event time in milliseconds (epoch). Used to filter out clusters whose ``last_ts`` is more than
        ``config.time_window_days`` old.
    state
        Accumulated :class:`ClusterState`. Returned unchanged on the call path (frozen).
    config
        :class:`ClusterConfig`. ``threshold`` and ``time_window_days`` are read; the LLM-specific knobs
        (``k_candidates`` / ``llm_skip_threshold``) are ignored here.

    Returns
    -------
    tuple[ClusterId, ClusterState]
        Either an existing cluster id (top-1 cosine cleared the threshold and the candidate was inside
        the time window), or a freshly minted ``cluster_NNN`` if no candidate qualified or ``state`` was
        empty.
    """
    best_cid = _find_best_within_window(state, vector, timestamp_ms, config)
    return state._assign(best_cid, vector, timestamp_ms)


async def cluster_by_llm(
    vector: np.ndarray,
    timestamp_ms: int,
    query_text: str,
    state: ClusterState,
    *,
    config: ClusterConfig,
    llm: LLMClient,
    cluster_previews: dict[ClusterId, list[str]],
    prompt: str | None = None,
) -> tuple[ClusterId, ClusterState]:
    """LLM-refined clustering: embedding recall → fast path → LLM decision with geometric fallback.

    Flow (opensource ``_cluster_memcell_llm`` line 404-539):

    1. **Empty state** — short-circuit to a freshly minted cluster; no LLM call.
    2. **Stage 1 (recall)** — top-K nearest clusters by cosine (no time-window filter).
    3. **Fast path** — if top-1 cosine ``>=`` ``config.llm_skip_threshold``, assign without calling the LLM.
    4. **Stage 2 (rank)** — render :data:`CLUSTER_LLM_ASSIGN_PROMPT` with ``query_text``,
       ``clusters_json`` (built from ``cluster_previews``), and ``next_new_id``; call the LLM with
       ``json_object`` response format; retry up to :data:`_MAX_LLM_RETRIES` on parse / schema failure.
    5. **Apply** — if the LLM picks an id already in ``state.counts``, assign there. If it picks
       anything else (eg. ``cluster_<new_idx>`` or an unknown id), mint a new cluster.
    6. **Fallback** — on LLM exhaustion or infrastructure error, mirror the geometry path's "top-1 if
       above ``config.threshold`` else new" rule. Clustering never raises on LLM failure (unlike
       :class:`everalgo.user_memory.EpisodeExtractor`); the geometric fallback is the hard contract.

    Parameters
    ----------
    vector
        Pre-computed embedding for the new event.
    timestamp_ms
        Event time in milliseconds (epoch).
    query_text
        Text representation of the new event (typically the ``task_intent`` for ``AgentCase`` clustering,
        or the episode body for episode clustering). Substituted into the prompt's ``{memcell_text}``.
    state
        Accumulated :class:`ClusterState`; returned unchanged on the call path.
    config
        :class:`ClusterConfig`. ``k_candidates`` / ``llm_skip_threshold`` / ``threshold`` are read.
    llm
        :class:`LLMClient` instance; caller routes the model (e.g. via scene-based router).
    cluster_previews
        Map of ``cluster_id -> [recent text, ...]`` for the candidate clusters returned by Stage 1.
        Caller is responsible for pre-fetching from their event store (EverAlgo never queries). Missing
        keys are tolerated — the corresponding cluster simply renders with empty ``recent_task_intents``.
    prompt
        Optional caller-supplied override for :data:`CLUSTER_LLM_ASSIGN_PROMPT`. Must remain compatible
        with :py:meth:`str.format` and reference at most the placeholders ``{memcell_text}`` /
        ``{clusters_json}`` / ``{next_new_id}``.
    """
    # 1. Empty state — nothing to compare against.
    if not state.centroids:
        return state._assign(None, vector, timestamp_ms)

    # 2. Stage 1: top-K embedding recall (no time-window filter here; the LLM ranks by semantics).
    candidates = _find_top_k_clusters(state, vector, config.k_candidates)
    if not candidates:
        return state._assign(None, vector, timestamp_ms)

    top_cid, top_sim = candidates[0]

    # 3. Fast path — top-1 confident enough to skip the LLM.
    if top_sim >= config.llm_skip_threshold:
        return state._assign(top_cid, vector, timestamp_ms)

    # 4. Stage 2: LLM rank.
    candidate_ids = [cid for cid, _ in candidates]
    clusters_json = _build_clusters_json(state, candidate_ids, cluster_previews)
    next_new_id = f"{state.next_idx:03d}"
    rendered = render_prompt(
        CLUSTER_LLM_ASSIGN_PROMPT,
        prompt,
        memcell_text=query_text,
        clusters_json=clusters_json,
        next_new_id=next_new_id,
    )
    llm_result = await _call_llm_with_retry(llm, rendered)

    # 5. Apply LLM choice — unknown / missing cluster_id means "create new".
    if llm_result is not None:
        chosen_id = llm_result.get("cluster_id", "")
        if isinstance(chosen_id, str) and chosen_id in state.counts:
            return state._assign(chosen_id, vector, timestamp_ms)
        return state._assign(None, vector, timestamp_ms)

    # 6. Fallback — LLM exhausted; mirror the geometry path's top-1-or-new rule (no time window).
    if top_sim >= config.threshold:
        return state._assign(top_cid, vector, timestamp_ms)
    return state._assign(None, vector, timestamp_ms)


# --- private helpers --------------------------------------------------------


def _cosine(v: np.ndarray, c: np.ndarray) -> float:
    """Cosine similarity with ``_NORM_EPSILON`` divide-by-zero guard."""
    v_norm = float(np.linalg.norm(v)) + _NORM_EPSILON
    c_norm = float(np.linalg.norm(c)) + _NORM_EPSILON
    return float(np.dot(v, c) / (v_norm * c_norm))


def _find_best_within_window(
    state: ClusterState,
    vector: np.ndarray,
    timestamp_ms: int,
    config: ClusterConfig,
) -> ClusterId | None:
    """Top-1 candidate honouring the time-window filter; ``None`` if no cluster cleared the threshold.

    Used by :func:`cluster_by_geometry`. Mirrors opensource ``_find_best_cluster`` (manager.py:673-714).
    """
    if not state.centroids:
        return None

    max_gap_ms = int(config.time_window_days * _MS_PER_DAY)
    best_cid: ClusterId | None = None
    best_sim = -1.0

    for cid, centroid in state.centroids.items():
        if centroid.size == 0:
            continue
        last_ts = state.last_ts.get(cid)
        if last_ts is not None and abs(timestamp_ms - last_ts) > max_gap_ms:
            continue

        sim = _cosine(vector, centroid)
        if sim > best_sim:
            best_sim = sim
            best_cid = cid

    if best_sim >= config.threshold:
        return best_cid
    return None


def _find_top_k_clusters(
    state: ClusterState,
    vector: np.ndarray,
    k: int,
) -> list[tuple[ClusterId, float]]:
    """Top-K candidates by cosine similarity, sorted descending. No time-window filter.

    Used by :func:`cluster_by_llm`. Mirrors opensource ``_find_top_k_clusters`` (manager.py:541-580).
    Clusters with empty / zero-sized centroids are skipped silently.
    """
    scored: list[tuple[ClusterId, float]] = []
    for cid, centroid in state.centroids.items():
        if centroid.size == 0:
            continue
        scored.append((cid, _cosine(vector, centroid)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def _build_clusters_json(
    state: ClusterState,
    candidate_ids: list[ClusterId],
    previews: dict[ClusterId, list[str]],
) -> str:
    """Serialise candidate clusters to the JSON payload expected by ``{clusters_json}``.

    Mirrors opensource ``_build_clusters_json`` (manager.py:629-650). Missing entries in ``previews``
    render as empty ``recent_task_intents`` lists rather than raising — caller may legitimately not have
    fetched previews for newly created singleton clusters.
    """
    if not candidate_ids:
        return "(No existing clusters)"

    clusters = [
        {
            "cluster_id": cid,
            "item_count": state.counts.get(cid, 0),
            "recent_task_intents": previews.get(cid, []),
        }
        for cid in candidate_ids
    ]
    return json.dumps(clusters, ensure_ascii=False, indent=2)


async def _call_llm_with_retry(client: LLMClient, rendered: str) -> dict[str, Any] | None:
    """Call the LLM up to :data:`_MAX_LLM_RETRIES` times; return parsed dict or ``None`` on exhaustion.

    Returns ``None`` (never raises) so that callers can apply the geometric fallback. Any exception
    raised by the LLM (auth / network / JSON / schema) is logged at WARNING on the final attempt and
    treated as a soft failure — clustering's contract is "always returns a cluster".
    """
    last_error: Exception | None = None
    for _attempt in range(_MAX_LLM_RETRIES):
        try:
            response = await client.chat(
                messages=[LLMChatMessage(role="user", content=rendered)],
                response_format={"type": "json_object"},
            )
            parsed = _parse_llm_json(response.content)
            if parsed is None:
                last_error = ValueError("LLM response is not valid JSON")
                continue
            if "cluster_id" not in parsed:
                last_error = ValueError("LLM response missing cluster_id field")
                continue
        except Exception as e:
            last_error = e
            continue
        else:
            return parsed
    logger.warning(
        "cluster_by_llm: all %d LLM attempts failed; falling back to geometry top-1. Last error: %s",
        _MAX_LLM_RETRIES,
        last_error,
    )
    return None


def _parse_llm_json(raw: str) -> dict[str, Any] | None:
    """Two-tier JSON parse: direct ``json.loads`` then triple-backtick json fence stripping.

    Returns ``None`` if neither strategy succeeds. The prompt explicitly forbids markdown fences but
    some models still emit them; the fence fallback is purely defensive.
    """
    try:
        return _coerce_dict(json.loads(raw))
    except json.JSONDecodeError:
        pass
    if "```json" in raw:
        start = raw.find("```json") + len("```json")
        end = raw.find("```", start)
        if end > start:
            try:
                return _coerce_dict(json.loads(raw[start:end].strip()))
            except json.JSONDecodeError:
                pass
    return None


def _coerce_dict(value: Any) -> dict[str, Any] | None:
    """Normalise a ``json.loads`` result to ``dict[str, Any]`` or ``None``."""
    if isinstance(value, dict):
        return cast("dict[str, Any]", value)
    return None
