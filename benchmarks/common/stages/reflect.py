"""Stage 2 — Reflect: merge episodes within multi-member clusters.

For each cluster with 2+ episodes, calls ``EpisodeReflector.areflect()`` to produce
a single reflected episode.  Single-member clusters pass through unchanged.  Old
episodes in merged clusters are replaced; cluster state (``episode_ids``, ``centroid``,
``preview``) is updated to reference the reflected episode.

Input files (from Extract Base):
- ``episodes_conv_<i>.json``
- ``clusters_conv_<i>.json``
- ``memcells_conv_<i>.json``  (copied through unchanged for downstream stages)

Output: updated versions of all three files in ``ctx.output_dir``.
"""

from __future__ import annotations

import logging
import shutil
import time
import traceback
from typing import TYPE_CHECKING, Any

from benchmarks.common.services import build_llm_client
from benchmarks.common.stages.serialization import load_clusters, load_episodes, serialize_episode, write_json
from benchmarks.common.stages.types import StageStats
from everalgo.types import Episode
from everalgo.user_memory.reflect import EpisodeReflector

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.common.services import EmbeddingClient
    from benchmarks.common.stages.types import StageContext

logger = logging.getLogger(__name__)


def _episode_dict_to_model(ep: dict[str, Any]) -> Episode:
    """Convert a serialised episode dict to an ``Episode`` domain object."""
    return Episode(
        owner_id=ep.get("owner_id"),
        episode=ep["episode"],
        subject=ep.get("subject", ""),
        timestamp=int(ep["timestamp"]),
    )


async def _reflect_cluster(
    cluster: dict[str, Any],
    episodes_by_id: dict[str, dict[str, Any]],
    reflector: EpisodeReflector,
    embedding_client: EmbeddingClient,
) -> tuple[dict[str, Any], list[float]]:
    """Reflect a multi-member cluster into a single merged episode.

    Args:
        cluster: Cluster dict with ``episode_ids`` (len >= 2).
        episodes_by_id: Lookup from episode ID to episode dict.
        reflector: EpisodeReflector instance bound to an LLM client.
        embedding_client: Used to embed the merged episode body + subject.

    Returns:
        Tuple of (reflected_episode_dict, body_embedding_vector).  The dict is built
        via ``serialize_episode`` with a placeholder ``ep_idx=0`` — the caller
        re-numbers IDs after collecting all episodes.
    """
    source_eps = [episodes_by_id[eid] for eid in cluster["episode_ids"]]
    models = [_episode_dict_to_model(ep) for ep in source_eps]
    models.sort(key=lambda e: e.timestamp)

    merged: Episode = await reflector.areflect(models)

    merged_body = merged.episode
    merged_subject = merged.subject

    body_vec, subject_vec = await _embed_pair(merged_body, merged_subject, embedding_client)

    union_memcell_ids: list[str] = []
    seen: set[str] = set()
    for ep in source_eps:
        for mid in ep.get("memcell_ids", []):
            if mid not in seen:
                union_memcell_ids.append(mid)
                seen.add(mid)

    ep_dict = serialize_episode(
        0,  # placeholder; re-numbered by caller
        subject=merged_subject,
        episode_text=merged_body,
        memcell_ids=union_memcell_ids,
        timestamp=merged.timestamp,
        owner_id=None,
        embeddings={"episode": body_vec, "subject": subject_vec},
    )
    return ep_dict, body_vec


async def _embed_pair(
    body: str,
    subject: str,
    embedding_client: EmbeddingClient,
) -> tuple[list[float], list[float] | None]:
    """Embed episode body and (optionally) subject in a single batch call.

    Returns:
        ``(body_vec, subject_vec)`` where ``subject_vec`` is ``None`` when ``subject`` is empty.
    """
    texts = [body]
    has_subject = bool(subject)
    if has_subject:
        texts.append(subject)
    vecs = await embedding_client.embed(texts)
    body_vec = list(vecs[0])
    subject_vec = list(vecs[1]) if has_subject else None
    return body_vec, subject_vec


async def _reflect_one_conversation(
    conv_idx: int,
    input_dir: Path,
    output_dir: Path,
    reflector: EpisodeReflector,
    embedding_client: EmbeddingClient,
) -> bool:
    """Run reflection for one conversation: merge multi-member clusters, write outputs.

    Errors are isolated per conversation; full traceback written to
    ``reflect_conv_<conv_idx>.error.txt``.

    Returns:
        ``True`` on success, ``False`` on any error.
    """
    try:
        episodes = load_episodes(input_dir / f"episodes_conv_{conv_idx}.json")
        cluster_data = load_clusters(input_dir / f"clusters_conv_{conv_idx}.json")
        final_episodes, updated_clusters = await _merge_and_rebuild(
            episodes, cluster_data, reflector, embedding_client, conv_idx
        )
    except Exception:
        err_path = output_dir / f"reflect_conv_{conv_idx}.error.txt"
        err_path.write_text(traceback.format_exc())
        logger.exception("conv_%d reflect failed; traceback in %s", conv_idx, err_path)
        return False

    write_json(output_dir / f"episodes_conv_{conv_idx}.json", final_episodes)
    write_json(output_dir / f"clusters_conv_{conv_idx}.json", updated_clusters)

    # Copy memcells through unchanged — downstream stages may need them.
    memcells_src = input_dir / f"memcells_conv_{conv_idx}.json"
    if memcells_src.exists():
        shutil.copy2(memcells_src, output_dir / f"memcells_conv_{conv_idx}.json")

    return True


def _assemble_final_episodes(
    passthrough_eids: list[str],
    reflected_by_cluster: dict[str, dict[str, Any]],
    episodes_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combine passthrough and reflected episodes into a sorted, re-numbered list.

    Args:
        passthrough_eids: Episode IDs that pass through unchanged (orphans + single-member clusters).
        reflected_by_cluster: Cluster ID → reflected episode dict for multi-member clusters.
        episodes_by_id: Lookup from episode ID to original episode dict.

    Returns:
        Final episode list sorted by timestamp with contiguous IDs starting from ``"0"``.
    """
    final_episodes: list[dict[str, Any]] = [episodes_by_id[eid] for eid in passthrough_eids if eid in episodes_by_id]
    final_episodes.extend(reflected_by_cluster.values())
    final_episodes.sort(key=lambda e: int(e["timestamp"]))
    for idx, ep in enumerate(final_episodes):
        ep["id"] = str(idx)
    return final_episodes


async def _process_clusters(
    clusters: list[dict[str, Any]],
    cluster_episode_map: dict[str, list[str]],
    episodes_by_id: dict[str, dict[str, Any]],
    reflector: EpisodeReflector,
    embedding_client: EmbeddingClient,
    *,
    orphan_eids: list[str],
    conv_idx: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[float]], list[str]]:
    """Iterate clusters: reflect multi-member ones, collect passthrough IDs for the rest.

    Args:
        clusters: All cluster dicts for this conversation.
        cluster_episode_map: Cluster ID → list of episode IDs belonging to it.
        episodes_by_id: Lookup from episode ID to episode dict.
        reflector: EpisodeReflector instance bound to an LLM client.
        embedding_client: For embedding reflected episodes.
        orphan_eids: Episode IDs not assigned to any cluster.
        conv_idx: Conversation index (for log messages only).

    Returns:
        Tuple of (reflected_by_cluster, reflected_vecs, passthrough_eids).
    """
    reflected_by_cluster: dict[str, dict[str, Any]] = {}
    reflected_vecs: dict[str, list[float]] = {}
    passthrough_eids: list[str] = list(orphan_eids)

    for cluster in clusters:
        cid = cluster["id"]
        member_eids = cluster_episode_map.get(cid, [])

        if len(member_eids) < 2:
            passthrough_eids.extend(member_eids)
            continue

        try:
            ep_dict, body_vec = await _reflect_cluster(cluster, episodes_by_id, reflector, embedding_client)
            reflected_by_cluster[cid] = ep_dict
            reflected_vecs[cid] = body_vec
        except Exception:
            logger.warning("conv_%d cluster %s reflect failed; keeping original episodes", conv_idx, cid, exc_info=True)
            passthrough_eids.extend(member_eids)

    return reflected_by_cluster, reflected_vecs, passthrough_eids


async def _merge_and_rebuild(
    episodes: list[dict[str, Any]],
    cluster_data: dict[str, Any],
    reflector: EpisodeReflector,
    embedding_client: EmbeddingClient,
    conv_idx: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Core merge logic: reflect multi-member clusters, rebuild episode list + cluster metadata.

    Args:
        episodes: Original episode dicts from Extract Base.
        cluster_data: Cluster dict with ``clusters`` and ``episode_to_cluster``.
        reflector: EpisodeReflector instance.
        embedding_client: For embedding reflected episodes.
        conv_idx: Conversation index (for log messages only).

    Returns:
        Tuple of (final_episodes_list, updated_cluster_data_dict).
    """
    episodes_by_id: dict[str, dict[str, Any]] = {str(ep["id"]): ep for ep in episodes}
    episode_to_cluster: dict[str, str] = cluster_data.get("episode_to_cluster", {})
    clusters: list[dict[str, Any]] = cluster_data.get("clusters", [])

    cluster_episode_map: dict[str, list[str]] = {}
    for eid, cid in episode_to_cluster.items():
        cluster_episode_map.setdefault(cid, []).append(eid)

    clustered_eids: set[str] = set(episode_to_cluster.keys())
    orphan_eids = [str(ep["id"]) for ep in episodes if str(ep["id"]) not in clustered_eids]

    reflected_by_cluster, reflected_vecs, passthrough_eids = await _process_clusters(
        clusters,
        cluster_episode_map,
        episodes_by_id,
        reflector,
        embedding_client,
        orphan_eids=orphan_eids,
        conv_idx=conv_idx,
    )

    final_episodes = _assemble_final_episodes(passthrough_eids, reflected_by_cluster, episodes_by_id)

    updated_clusters = _rebuild_cluster_metadata(
        clusters, final_episodes, reflected_by_cluster, reflected_vecs, episodes_by_id=episodes_by_id
    )
    return final_episodes, updated_clusters


def _remap_reflected_cluster(
    cluster: dict[str, Any],
    reflected_ep: dict[str, Any],
    reflected_vecs: dict[str, list[float]],
    new_id_by_key: dict[str, str],
    new_episode_to_cluster: dict[str, str],
) -> dict[str, Any]:
    """Build updated metadata for a reflected (multi-member) cluster."""
    cid: str = cluster["id"]
    key = f"{reflected_ep['episode']}|{reflected_ep['timestamp']}"
    new_ep_id: str = new_id_by_key[key] if key in new_id_by_key else str(reflected_ep["id"])
    new_episode_to_cluster[new_ep_id] = cid
    return {
        **cluster,
        "episode_ids": [new_ep_id],
        "centroid": reflected_vecs[cid],
        "preview": [reflected_ep["episode"][:200]],
    }


def _remap_passthrough_cluster(
    cluster: dict[str, Any],
    episodes_by_id: dict[str, dict[str, Any]],
    new_id_by_key: dict[str, str],
    new_episode_to_cluster: dict[str, str],
) -> dict[str, Any]:
    """Build updated metadata for a passthrough (single-member or un-reflected) cluster."""
    cid: str = cluster["id"]
    new_eids: list[str] = []
    for old_eid in cluster.get("episode_ids", []):
        if old_eid in episodes_by_id:
            old_ep = episodes_by_id[old_eid]
            key = f"{old_ep['episode']}|{old_ep['timestamp']}"
            new_id = new_id_by_key.get(key)
            if new_id is not None:
                new_eids.append(new_id)
                new_episode_to_cluster[new_id] = cid
    return {**cluster, "episode_ids": new_eids}


def _rebuild_cluster_metadata(
    clusters: list[dict[str, Any]],
    final_episodes: list[dict[str, Any]],
    reflected_by_cluster: dict[str, dict[str, Any]],
    reflected_vecs: dict[str, list[float]],
    *,
    episodes_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Rebuild cluster metadata after reflection.

    For reflected clusters: ``episode_ids`` = [reflected_ep_id], ``centroid`` = reflected embedding,
    ``preview`` = [reflected episode text snippet].
    For passthrough clusters: ``episode_ids`` remapped to new IDs.

    Returns:
        Updated cluster data dict with ``clusters`` and ``episode_to_cluster``.
    """
    new_id_by_key: dict[str, str] = {f"{ep['episode']}|{ep['timestamp']}": str(ep["id"]) for ep in final_episodes}

    updated_clusters: list[dict[str, Any]] = []
    new_episode_to_cluster: dict[str, str] = {}

    for cluster in clusters:
        cid = cluster["id"]
        if cid in reflected_by_cluster:
            updated_clusters.append(
                _remap_reflected_cluster(
                    cluster, reflected_by_cluster[cid], reflected_vecs, new_id_by_key, new_episode_to_cluster
                )
            )
        else:
            updated_clusters.append(
                _remap_passthrough_cluster(cluster, episodes_by_id, new_id_by_key, new_episode_to_cluster)
            )

    return {"clusters": updated_clusters, "episode_to_cluster": new_episode_to_cluster}


def _passthrough_files(input_dir: Path, output_dir: Path) -> None:
    """Copy all conv entity files from input to output unchanged (used when Reflection is disabled)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("memcells_conv_*.json", "episodes_conv_*.json", "clusters_conv_*.json"):
        for src in sorted(input_dir.glob(pattern)):
            shutil.copy2(src, output_dir / src.name)


async def run_reflect_stage(ctx: StageContext) -> StageStats:
    """Stage 2 — Reflect: merge episodes within multi-member clusters.

    When ``ctx.config.enable_reflection`` is ``False``, returns immediately with empty stats.
    Otherwise reads episode and cluster files from ``ctx.input_dir``, merges multi-member
    clusters via ``EpisodeReflector``, and writes updated files to ``ctx.output_dir``.

    Also copies ``memcells_conv_<i>.json`` through unchanged for downstream stages.

    Args:
        ctx: Stage execution context providing config, services, I/O dirs, and smoke flags.

    Returns:
        ``StageStats`` with ``stage_name="reflect"``.
    """
    if not ctx.config.enable_reflection:
        _passthrough_files(ctx.input_dir, ctx.output_dir)
        return StageStats(stage_name="reflect", success=0, failed=0, duration_seconds=0.0)

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage_name="reflect")
    started = time.monotonic()

    llm = build_llm_client(ctx.config)
    reflector = EpisodeReflector(llm=llm)
    embedding_client = ctx.services.embedding

    episode_files = sorted(ctx.input_dir.glob("episodes_conv_*.json"))
    if ctx.smoke:
        episode_files = episode_files[: ctx.smoke_conv_limit]

    from tqdm import tqdm as _tqdm  # Deferred: optional dependency, avoid top-level import

    for ep_file in _tqdm(episode_files, desc="reflect", unit="conv", dynamic_ncols=True):
        stem = ep_file.stem  # "episodes_conv_<i>"
        conv_idx = int(stem.split("_")[-1])

        ok = await _reflect_one_conversation(conv_idx, ctx.input_dir, ctx.output_dir, reflector, embedding_client)
        if ok:
            stats.success += 1
        else:
            stats.failed += 1

    stats.duration_seconds = time.monotonic() - started
    return stats
