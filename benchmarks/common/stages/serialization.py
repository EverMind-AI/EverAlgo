"""Shared serialization / deserialization for the entity-split data model.

Each entity (MemCell, Episode, AtomicFact, Cluster) is stored as a separate JSON file per
conversation, linked by string IDs. This module centralizes the schema so that all stages
produce and consume a consistent format.

ID scheme: string-ified integers ("0", "1", "2", …). Field names follow the EverAlgo algo
contract — Episode text is ``episode`` (not ``content``); AtomicFact text is ``content``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def serialize_memcell(mc_idx: int, mc: Any) -> dict[str, Any]:
    """Serialize a MemCell — pure boundary-detection output, no episode/facts.

    Args:
        mc_idx: Zero-based index used as the string ID.
        mc: MemCell-like object with ``timestamp`` and ``items`` (each having
            ``id``, ``role``, ``content``, ``timestamp``, ``sender_id``, ``sender_name``).

    Returns:
        JSON-serialisable dict with ``id``, ``timestamp``, and ``items``.
    """
    return {
        "id": str(mc_idx),
        "timestamp": mc.timestamp,
        "items": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "sender_id": m.sender_id,
                "sender_name": m.sender_name,
            }
            for m in mc.items
        ],
    }


def serialize_episode(
    ep_idx: int,
    *,
    subject: str,
    episode_text: str,
    memcell_ids: list[str],
    timestamp: int,
    owner_id: str | None,
    embeddings: dict[str, list[float] | None],
) -> dict[str, Any]:
    """Serialize an Episode entity (original or reflected).

    Args:
        ep_idx: Zero-based index used as the string ID.
        subject: Episode subject line.
        episode_text: Narrative text; stored under the algo field name ``episode``.
        memcell_ids: MemCell IDs this episode was extracted from (one for original, many for reflected).
        timestamp: Unix timestamp of the originating MemCell.
        owner_id: Speaker / owner identifier, or None.
        embeddings: Dict with ``"episode"`` (body vector) and ``"subject"`` (subject vector or None).

    Returns:
        JSON-serialisable dict following the entity-split schema.
    """
    return {
        "id": str(ep_idx),
        "owner_id": owner_id,
        "memcell_ids": memcell_ids,
        "subject": subject,
        "episode": episode_text,
        "timestamp": timestamp,
        "embeddings": embeddings,
    }


def serialize_atomic_fact(
    af_idx: int,
    *,
    content: str,
    episode_id: str,
    timestamp: int,
    owner_id: str | None,
    embeddings: list[float],
) -> dict[str, Any]:
    """Serialize a single AtomicFact entity (per-item, not grouped).

    Args:
        af_idx: Zero-based index across all facts in a conversation; used as the string ID.
        content: Fact text; stored under the algo field name ``content``.
        episode_id: ID of the parent Episode.
        timestamp: Unix timestamp of the originating MemCell.
        owner_id: Speaker / owner identifier, or None.
        embeddings: Dense vector for this fact.

    Returns:
        JSON-serialisable dict for one atomic fact.
    """
    return {
        "id": str(af_idx),
        "episode_id": episode_id,
        "owner_id": owner_id,
        "content": content,
        "timestamp": timestamp,
        "embeddings": embeddings,
    }


def serialize_clusters(
    clusters_data: list[dict[str, Any]],
    memcell_to_episode: dict[str, str],
) -> dict[str, Any]:
    """Serialize clusters with episode_ids and reverse map.

    Args:
        clusters_data: List of raw cluster dicts, each with keys ``id``, ``centroid``,
            ``count``, ``last_ts``, ``members`` (memcell ID list), and ``preview``.
        memcell_to_episode: Map from memcell ID → episode ID, used to populate
            ``episode_ids`` and build the ``episode_to_cluster`` reverse map.
            Members absent from this map are silently excluded from ``episode_ids``.

    Returns:
        Dict with:
        - ``clusters``: enriched cluster list (each entry has ``episode_ids``).
        - ``episode_to_cluster``: episode ID → cluster ID reverse map.
    """
    enriched: list[dict[str, Any]] = []
    episode_to_cluster: dict[str, str] = {}

    for cl in clusters_data:
        episode_ids = [memcell_to_episode[mid] for mid in cl["members"] if mid in memcell_to_episode]
        enriched.append(
            {
                "id": cl["id"],
                "centroid": cl["centroid"],
                "count": cl["count"],
                "last_ts": cl["last_ts"],
                "episode_ids": episode_ids,
                "preview": cl.get("preview", []),
            }
        )
        for eid in episode_ids:
            episode_to_cluster[eid] = cl["id"]

    return {
        "clusters": enriched,
        "episode_to_cluster": episode_to_cluster,
    }


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` to ``path`` as pretty-printed UTF-8 JSON, creating parent dirs as needed.

    Args:
        path: Destination file path.
        data: JSON-serialisable value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_memcells(path: Path) -> list[dict[str, Any]]:
    """Load a memcells JSON file.

    Args:
        path: Path to the ``memcells_conv_<i>.json`` file.

    Returns:
        List of memcell dicts.
    """
    result: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return result


def load_episodes(path: Path) -> list[dict[str, Any]]:
    """Load an episodes JSON file.

    Args:
        path: Path to the ``episodes_conv_<i>.json`` file.

    Returns:
        List of episode dicts.
    """
    result: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return result


def load_atomic_facts(path: Path) -> list[dict[str, Any]]:
    """Load an atomic facts JSON file.

    Args:
        path: Path to the ``atomic_facts_conv_<i>.json`` file.

    Returns:
        List of atomic fact dicts.
    """
    result: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return result


def load_clusters(path: Path) -> dict[str, Any]:
    """Load a clusters JSON file.

    Args:
        path: Path to the ``clusters_conv_<i>.json`` file.

    Returns:
        Dict with ``clusters`` and ``episode_to_cluster`` keys.
    """
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result
