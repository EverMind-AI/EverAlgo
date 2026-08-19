"""Tests for Stage 2 — Reflect: episode merging within multi-member clusters.

Unit tests cover:
- ``run_reflect_stage`` skips when ``enable_reflection=False``
- ``run_reflect_stage`` is an async function (structural check)
- ``_reflect_one_conversation`` — multi-member clusters produce 1 reflected episode
- Single-member clusters pass through unchanged
- Reflected episodes have union of source memcell_ids
- Total episode count is reduced after reflection
- Cluster centroids are updated for reflected clusters
- Per-conversation error isolation (writes ``.error.txt``)
- Memcells file is copied through unchanged
- End-to-end ``run_reflect_stage`` with mocked LLM + embedding
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.stages.reflect import (
    _merge_and_rebuild,
    _reflect_one_conversation,
    run_reflect_stage,
)
from benchmarks.common.stages.serialization import write_json
from benchmarks.common.stages.types import StageContext

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    ep_id: str = "0",
    episode_text: str = "Alice went to the store.",
    subject: str = "shopping",
    timestamp: int = 1_700_000_000_000,
    owner_id: str | None = None,
    memcell_ids: list[str] | None = None,
    embedding: list[float] | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """Return a minimal episode dict matching the Extract Base output schema."""
    return {
        "id": ep_id,
        "owner_id": owner_id,
        "memcell_ids": memcell_ids or [ep_id],
        "subject": subject,
        "episode": episode_text,
        "summary": summary if summary is not None else f"Preview of {episode_text}",
        "timestamp": timestamp,
        "embeddings": {"episode": embedding or [0.1, 0.2, 0.3], "subject": None},
    }


def _make_cluster_data(
    episodes: list[dict[str, Any]],
    cluster_assignments: dict[str, str],
) -> dict[str, Any]:
    """Build cluster data dict from episode list and episode→cluster assignments.

    Args:
        episodes: List of episode dicts.
        cluster_assignments: Maps episode ID → cluster ID.
    """
    clusters_map: dict[str, dict[str, Any]] = {}
    for ep in episodes:
        eid = str(ep["id"])
        cid = cluster_assignments.get(eid, f"cluster_{eid}")
        if cid not in clusters_map:
            clusters_map[cid] = {
                "id": cid,
                "centroid": ep["embeddings"]["episode"],
                "count": 0,
                "last_ts": 0,
                "episode_ids": [],
                "preview": [],
            }
        cl = clusters_map[cid]
        cl["episode_ids"].append(eid)
        cl["count"] += 1
        cl["last_ts"] = max(cl["last_ts"], int(ep["timestamp"]))
        cl["preview"].append(ep["episode"][:50])

    clusters = list(clusters_map.values())
    episode_to_cluster: dict[str, str] = {}
    for cl in clusters:
        for eid in cl["episode_ids"]:
            episode_to_cluster[eid] = cl["id"]

    return {
        "clusters": clusters,
        "episode_to_cluster": episode_to_cluster,
    }


def _make_embedding_client(vector: list[float] | None = None) -> MagicMock:
    """Return a mock EmbeddingClient whose embed() returns one vector per text."""
    vec = vector or [0.9, 0.8, 0.7]
    client = MagicMock()
    client.embed = AsyncMock(side_effect=lambda texts: [[*vec] for _ in texts])
    return client


def _make_reflector_mock(
    merged_text: str = "Merged episode narrative.",
    merged_subject: str = "merged topic",
) -> MagicMock:
    """Return a mock EpisodeReflector whose areflect() returns a merged Episode."""
    from everalgo.types import Episode

    mock = MagicMock()

    async def _areflect(episodes: Any, **kwargs: Any) -> Episode:
        return Episode(
            owner_id=None,
            episode=merged_text,
            subject=merged_subject,
            summary=f"Preview of {merged_subject}",
            timestamp=max(ep.timestamp for ep in episodes),
        )

    mock.areflect = AsyncMock(side_effect=_areflect)
    return mock


def _make_stage_context(
    tmp_path: Path,
    *,
    input_dir: Path | None = None,
    enable_reflection: bool = True,
    smoke: bool = False,
) -> StageContext:
    """Build a minimal StageContext for testing."""
    cfg = BenchmarkConfig(enable_reflection=enable_reflection)
    mock_services = MagicMock()
    mock_services.embedding = _make_embedding_client()
    return StageContext(
        config=cfg,
        services=mock_services,
        dataset=MagicMock(),
        input_dir=input_dir or tmp_path / "input",
        output_dir=tmp_path / "output",
        smoke=smoke,
    )


def _write_test_fixtures(
    input_dir: Path,
    conv_idx: int = 0,
    n_episodes: int = 4,
    n_multi_member_clusters: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Write episode + cluster fixture files for one conversation.

    Creates ``n_episodes`` episodes.  The first ``n_multi_member_clusters`` clusters each get 2
    episodes; remaining episodes go into single-member clusters.

    Returns:
        (episodes_list, cluster_data_dict) for assertion convenience.
    """
    episodes: list[dict[str, Any]] = [
        _make_episode(
            ep_id=str(i),
            episode_text=f"Episode {i} narrative text about event {i}.",
            subject=f"topic_{i}",
            timestamp=1_700_000_000_000 + i * 1000,
            memcell_ids=[str(i)],
            embedding=[float(i) * 0.1 + 0.01 * j for j in range(3)],
        )
        for i in range(n_episodes)
    ]

    # Build cluster assignments: pair first 2*n_multi_member episodes into multi-member clusters
    assignments: dict[str, str] = {}
    cluster_idx = 0
    ep_idx = 0
    for _ in range(n_multi_member_clusters):
        cid = f"cluster_{cluster_idx}"
        assignments[str(ep_idx)] = cid
        assignments[str(ep_idx + 1)] = cid
        ep_idx += 2
        cluster_idx += 1
    for i in range(ep_idx, n_episodes):
        assignments[str(i)] = f"cluster_{cluster_idx}"
        cluster_idx += 1

    cluster_data = _make_cluster_data(episodes, assignments)

    write_json(input_dir / f"episodes_conv_{conv_idx}.json", episodes)
    write_json(input_dir / f"clusters_conv_{conv_idx}.json", cluster_data)
    return episodes, cluster_data


# ---------------------------------------------------------------------------
# Structural check
# ---------------------------------------------------------------------------


def test_run_reflect_stage_is_coroutine() -> None:
    """Structural check: ``run_reflect_stage`` is an async function."""
    import inspect

    assert inspect.iscoroutinefunction(run_reflect_stage)


# ---------------------------------------------------------------------------
# Skip when disabled
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skipped_when_disabled(tmp_path: Path) -> None:
    """When ``enable_reflection=False``, stage returns immediately with 0 counts."""
    ctx = _make_stage_context(tmp_path, enable_reflection=False)
    stats = await run_reflect_stage(ctx)

    assert stats.stage_name == "reflect"
    assert stats.success == 0
    assert stats.failed == 0
    assert stats.duration_seconds == 0.0


# ---------------------------------------------------------------------------
# Unit: _merge_and_rebuild
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_member_cluster_merged() -> None:
    """Clusters with 2+ episodes produce 1 reflected episode after merge."""
    episodes = [
        _make_episode("0", "Event A happened.", "topicA", 1_700_000_000_000, memcell_ids=["0"]),
        _make_episode("1", "Event B happened.", "topicB", 1_700_000_001_000, memcell_ids=["1"]),
        _make_episode("2", "Event C happened.", "topicC", 1_700_000_002_000, memcell_ids=["2"]),
    ]
    # Cluster 0 has episodes 0,1 (multi); cluster 1 has episode 2 (single)
    cluster_data = _make_cluster_data(episodes, {"0": "c0", "1": "c0", "2": "c1"})

    reflector = _make_reflector_mock()
    embedding_client = _make_embedding_client()

    final_episodes, updated_clusters = await _merge_and_rebuild(
        episodes, cluster_data, reflector, embedding_client, conv_idx=0
    )

    # Should have 2 episodes: 1 reflected (from c0) + 1 passthrough (from c1)
    assert len(final_episodes) == 2
    reflector.areflect.assert_awaited_once()

    # The reflected cluster should have exactly 1 episode_id
    for cl in updated_clusters["clusters"]:
        assert len(cl["episode_ids"]) == 1


@pytest.mark.asyncio
async def test_single_member_cluster_unchanged() -> None:
    """Clusters with 1 episode pass through without reflection."""
    episodes = [
        _make_episode("0", "Solo event.", "solo", 1_700_000_000_000, memcell_ids=["0"]),
    ]
    cluster_data = _make_cluster_data(episodes, {"0": "c0"})

    reflector = _make_reflector_mock()
    embedding_client = _make_embedding_client()

    final_episodes, _ = await _merge_and_rebuild(episodes, cluster_data, reflector, embedding_client, conv_idx=0)

    assert len(final_episodes) == 1
    assert final_episodes[0]["episode"] == "Solo event."
    reflector.areflect.assert_not_awaited()


@pytest.mark.asyncio
async def test_reflected_episode_has_union_memcell_ids() -> None:
    """Reflected episode's memcell_ids is the union of all source episodes' memcell_ids."""
    episodes = [
        _make_episode("0", "A", "t0", 1_700_000_000_000, memcell_ids=["0", "1"]),
        _make_episode("1", "B", "t1", 1_700_000_001_000, memcell_ids=["2"]),
    ]
    cluster_data = _make_cluster_data(episodes, {"0": "c0", "1": "c0"})

    reflector = _make_reflector_mock()
    embedding_client = _make_embedding_client()

    final_episodes, _ = await _merge_and_rebuild(episodes, cluster_data, reflector, embedding_client, conv_idx=0)

    reflected = [ep for ep in final_episodes if len(ep["memcell_ids"]) > 1]
    assert len(reflected) == 1
    assert set(reflected[0]["memcell_ids"]) == {"0", "1", "2"}


@pytest.mark.asyncio
async def test_total_episodes_reduced() -> None:
    """Reflection should reduce total episode count when multi-member clusters exist."""
    episodes = [
        _make_episode("0", "A", "t0", 1_700_000_000_000),
        _make_episode("1", "B", "t1", 1_700_000_001_000),
        _make_episode("2", "C", "t2", 1_700_000_002_000),
    ]
    # 2 in cluster, 1 standalone
    cluster_data = _make_cluster_data(episodes, {"0": "c0", "1": "c0", "2": "c1"})

    reflector = _make_reflector_mock()
    embedding_client = _make_embedding_client()

    final_episodes, _ = await _merge_and_rebuild(episodes, cluster_data, reflector, embedding_client, conv_idx=0)

    assert len(final_episodes) < len(episodes)


@pytest.mark.asyncio
async def test_cluster_centroid_updated() -> None:
    """Reflected clusters should have updated centroids matching the reflected embedding."""
    episodes = [
        _make_episode("0", "A", "t0", 1_700_000_000_000, embedding=[0.1, 0.2, 0.3]),
        _make_episode("1", "B", "t1", 1_700_000_001_000, embedding=[0.4, 0.5, 0.6]),
    ]
    cluster_data = _make_cluster_data(episodes, {"0": "c0", "1": "c0"})

    new_vec = [0.9, 0.8, 0.7]
    reflector = _make_reflector_mock()
    embedding_client = _make_embedding_client(vector=new_vec)

    _, updated_clusters = await _merge_and_rebuild(episodes, cluster_data, reflector, embedding_client, conv_idx=0)

    reflected_cl = [cl for cl in updated_clusters["clusters"] if cl["id"] == "c0"]
    assert len(reflected_cl) == 1
    assert reflected_cl[0]["centroid"] == new_vec


@pytest.mark.asyncio
async def test_ids_renumbered_contiguously() -> None:
    """After reflection, episode IDs should be contiguous starting from 0."""
    episodes = [
        _make_episode("0", "A", "t0", 1_700_000_000_000),
        _make_episode("1", "B", "t1", 1_700_000_001_000),
        _make_episode("2", "C", "t2", 1_700_000_002_000),
        _make_episode("3", "D", "t3", 1_700_000_003_000),
    ]
    # Two multi-member clusters: (0,1) and (2,3) → 2 reflected episodes
    cluster_data = _make_cluster_data(episodes, {"0": "c0", "1": "c0", "2": "c1", "3": "c1"})

    reflector = _make_reflector_mock()
    embedding_client = _make_embedding_client()

    final_episodes, _ = await _merge_and_rebuild(episodes, cluster_data, reflector, embedding_client, conv_idx=0)

    ids = [ep["id"] for ep in final_episodes]
    assert ids == [str(i) for i in range(len(final_episodes))]


@pytest.mark.asyncio
async def test_reflect_failure_keeps_original_episodes() -> None:
    """If areflect() fails for a cluster, original episodes are kept as passthrough."""
    episodes = [
        _make_episode("0", "A", "t0", 1_700_000_000_000),
        _make_episode("1", "B", "t1", 1_700_000_001_000),
    ]
    cluster_data = _make_cluster_data(episodes, {"0": "c0", "1": "c0"})

    reflector = MagicMock()
    reflector.areflect = AsyncMock(side_effect=RuntimeError("LLM exploded"))
    embedding_client = _make_embedding_client()

    final_episodes, _ = await _merge_and_rebuild(episodes, cluster_data, reflector, embedding_client, conv_idx=0)

    # Both originals should be kept since reflection failed
    assert len(final_episodes) == 2


# ---------------------------------------------------------------------------
# Integration: _reflect_one_conversation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflect_one_conversation_writes_files(tmp_path: Path) -> None:
    """Happy path: writes updated episodes + clusters + copies memcells."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    _write_test_fixtures(input_dir, conv_idx=0, n_episodes=4, n_multi_member_clusters=1)
    # Write memcells file
    write_json(input_dir / "memcells_conv_0.json", [{"id": str(i)} for i in range(4)])

    reflector = _make_reflector_mock()
    embedding_client = _make_embedding_client()

    ok = await _reflect_one_conversation(0, input_dir, output_dir, reflector, embedding_client)

    assert ok is True
    assert (output_dir / "episodes_conv_0.json").exists()
    assert (output_dir / "clusters_conv_0.json").exists()
    assert (output_dir / "memcells_conv_0.json").exists()

    out_episodes = json.loads((output_dir / "episodes_conv_0.json").read_text())
    # 4 episodes, 1 multi-member cluster (2 eps merged → 1) + 2 single → 3 total
    assert len(out_episodes) == 3


@pytest.mark.asyncio
async def test_reflect_one_conversation_error_isolation(tmp_path: Path) -> None:
    """On exception, writes .error.txt and returns False."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Write invalid JSON to trigger load failure
    (input_dir / "episodes_conv_5.json").write_text("not valid json")
    (input_dir / "clusters_conv_5.json").write_text("{}")

    reflector = _make_reflector_mock()
    embedding_client = _make_embedding_client()

    ok = await _reflect_one_conversation(5, input_dir, output_dir, reflector, embedding_client)

    assert ok is False
    err_file = output_dir / "reflect_conv_5.error.txt"
    assert err_file.exists()


# ---------------------------------------------------------------------------
# Integration: run_reflect_stage — end-to-end with file I/O
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_reflect_stage_end_to_end(tmp_path: Path) -> None:
    """End-to-end: stage reads episode + cluster files, writes reflected output."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    _write_test_fixtures(input_dir, conv_idx=0, n_episodes=4, n_multi_member_clusters=1)
    write_json(input_dir / "memcells_conv_0.json", [{"id": "0"}])

    ctx = _make_stage_context(tmp_path, input_dir=input_dir, enable_reflection=True)

    with (
        patch("benchmarks.common.stages.reflect.build_llm_client", return_value=MagicMock()),
        patch("benchmarks.common.stages.reflect.EpisodeReflector", return_value=_make_reflector_mock()),
    ):
        stats = await run_reflect_stage(ctx)

    assert stats.stage_name == "reflect"
    assert stats.success == 1
    assert stats.failed == 0
    assert stats.duration_seconds > 0

    out_episodes = json.loads((ctx.output_dir / "episodes_conv_0.json").read_text())
    assert isinstance(out_episodes, list)
    assert len(out_episodes) < 4  # reflection reduced count


@pytest.mark.asyncio
async def test_run_reflect_stage_smoke_limits_convs(tmp_path: Path) -> None:
    """Smoke mode: only the first ``smoke_conv_limit`` conversation files are processed."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    for i in range(3):
        _write_test_fixtures(input_dir, conv_idx=i, n_episodes=3, n_multi_member_clusters=1)
        write_json(input_dir / f"memcells_conv_{i}.json", [])

    ctx = _make_stage_context(tmp_path, input_dir=input_dir, enable_reflection=True, smoke=True)

    with (
        patch("benchmarks.common.stages.reflect.build_llm_client", return_value=MagicMock()),
        patch("benchmarks.common.stages.reflect.EpisodeReflector", return_value=_make_reflector_mock()),
    ):
        stats = await run_reflect_stage(ctx)

    assert stats.success == 1  # only conv_0 processed
    assert stats.failed == 0
    assert (ctx.output_dir / "episodes_conv_0.json").exists()
    assert not (ctx.output_dir / "episodes_conv_1.json").exists()


@pytest.mark.asyncio
async def test_run_reflect_stage_no_input_files(tmp_path: Path) -> None:
    """Stage must succeed gracefully with 0 episode files (no-op)."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    ctx = _make_stage_context(tmp_path, input_dir=input_dir, enable_reflection=True)

    with patch("benchmarks.common.stages.reflect.build_llm_client", return_value=MagicMock()):
        stats = await run_reflect_stage(ctx)

    assert stats.success == 0
    assert stats.failed == 0


@pytest.mark.asyncio
async def test_run_reflect_stage_counts_failed_conversations(tmp_path: Path) -> None:
    """Failed conversation must increment ``failed`` counter, not ``success``."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    # Write valid episodes but invalid clusters to force a failure
    write_json(input_dir / "episodes_conv_0.json", [_make_episode("0")])
    (input_dir / "clusters_conv_0.json").write_text("not json")

    ctx = _make_stage_context(tmp_path, input_dir=input_dir, enable_reflection=True)

    with patch("benchmarks.common.stages.reflect.build_llm_client", return_value=MagicMock()):
        stats = await run_reflect_stage(ctx)

    assert stats.success == 0
    assert stats.failed == 1
