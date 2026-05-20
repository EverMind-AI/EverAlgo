"""Tests for Stage 1 MemCell extraction.

Most tests are skipped without API credentials (integration). A structural
test verifies the stage entry point exists and imports. Unit tests cover the
clustering helpers with mocked embedding and operator calls.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.stages.extract import (
    _cluster_one_memcell,
    _run_clustering_pass,
    _serialize_cluster_file,
    run_extract_stage,
)
from benchmarks.common.stages.types import StageContext
from everalgo.clustering.state import Cluster

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_embedding_client(vector: list[float] | None = None) -> MagicMock:
    """Return a mock EmbeddingClient whose embed() returns a single vector."""
    vec = vector or [0.1, 0.2, 0.3]
    client = MagicMock()
    client.embed = AsyncMock(return_value=[vec])
    return client


def _make_memcell(mc_id: str = "0", timestamp: int = 1_000_000, content: str = "hello world") -> dict[str, Any]:
    """Return a minimal memcell dict matching the stage 1 output schema."""
    return {
        "id": mc_id,
        "timestamp": timestamp,
        "items": [],
        "episode": {"subject": "test", "summary": "test", "content": content},
        "atomic_facts": [],
    }


def test_run_extract_stage_callable():
    """Structural check: function imports and is async."""
    import inspect

    assert inspect.iscoroutinefunction(run_extract_stage)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="requires OPENROUTER_API_KEY for real LLM call",
)
@pytest.mark.asyncio
async def test_extract_writes_memcell_json_for_mini_fixture(tmp_path: Path) -> None:
    """End-to-end: run extract on 1-conv fixture; verify output shape."""
    import json

    from benchmarks.common.services import Services
    from benchmarks.datasets.locomo.loader import LocomoDataset

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    dataset = LocomoDataset(data_path=fixture)
    cfg = BenchmarkConfig()
    ctx = StageContext(
        config=cfg,
        services=Services.from_config(cfg),
        dataset=dataset,
        input_dir=tmp_path,
        output_dir=tmp_path / "stage1_extract",
        smoke=True,
    )
    stats = await run_extract_stage(ctx)
    assert stats.stage_name == "extract"
    assert stats.success >= 1
    assert stats.duration_seconds > 0

    out_file = tmp_path / "stage1_extract" / "memcells_conv_0.json"
    assert out_file.exists()
    data: list[Any] = json.loads(out_file.read_text())
    assert isinstance(data, list)
    assert len(data) >= 1
    mc: dict[str, Any] = data[0]
    # Required EverAlgo-native fields
    for required in (
        "id",
        "timestamp",
        "items",
        "episode",
        "atomic_facts",
    ):
        assert required in mc, f"missing field: {required}"
    # Verify removed multi-tenant fields are absent
    assert "event_id" not in mc
    assert "group_id" not in mc
    assert "participants" not in mc
    assert "sender_ids" not in mc
    assert "original_data" not in mc
    # id is session-local sequence string
    assert mc["id"] == "0"
    # episode is a nested dict with subject + content
    assert isinstance(mc["episode"], dict)
    assert "subject" in mc["episode"]
    assert "content" in mc["episode"]
    # atomic_facts is a flat list
    assert isinstance(mc["atomic_facts"], list)


# ---------------------------------------------------------------------------
# Clustering unit tests (no API credentials required)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cluster_one_memcell_mints_first_cluster() -> None:
    """First memcell in an empty list must always create scene_0."""
    client = _make_embedding_client([1.0, 0.0, 0.0])
    mc = _make_memcell("mc0", timestamp=1000, content="first episode")

    result = await _cluster_one_memcell(mc, [], client, threshold=0.70, time_window_days=7.0)

    assert len(result) == 1
    assert result[0].id == "scene_0"
    assert result[0].members == ["mc0"]
    assert result[0].last_ts == 1000
    client.embed.assert_awaited_once_with(["first episode"])


@pytest.mark.asyncio
async def test_cluster_one_memcell_merges_into_existing() -> None:
    """A memcell similar to an existing cluster must merge, not mint a new one."""
    # Both vectors are identical → cosine similarity == 1.0, well above threshold.
    vec = [1.0, 0.0, 0.0]
    client = _make_embedding_client(vec)

    existing = Cluster(
        id="scene_0",
        centroid=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        count=1,
        last_ts=500,
        members=["mc0"],
        preview=["old preview"],
    )
    mc = _make_memcell("mc1", timestamp=1000, content="similar episode")

    result = await _cluster_one_memcell(mc, [existing], client, threshold=0.70, time_window_days=7.0)

    # Must still have exactly one cluster (merged, not appended).
    assert len(result) == 1
    assert result[0].id == "scene_0"
    assert "mc1" in result[0].members
    assert result[0].count == 2


@pytest.mark.asyncio
async def test_cluster_one_memcell_skips_empty_episode() -> None:
    """Empty episode body must be skipped; embedding client must not be called."""
    client = _make_embedding_client()
    mc = _make_memcell("mc0", content="")

    result = await _cluster_one_memcell(mc, [], client, threshold=0.70, time_window_days=7.0)

    assert result == []
    client.embed.assert_not_awaited()


@pytest.mark.asyncio
async def test_cluster_one_memcell_appends_new_cluster_when_dissimilar() -> None:
    """A dissimilar memcell must mint a second cluster, not overwrite the first."""
    # Orthogonal vectors → cosine == 0.0, below threshold.
    existing = Cluster(
        id="scene_0",
        centroid=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        count=1,
        last_ts=0,
        members=["mc0"],
        preview=["old"],
    )
    # New vector is orthogonal.
    client = _make_embedding_client([0.0, 1.0, 0.0])
    mc = _make_memcell("mc1", timestamp=0, content="orthogonal topic")

    result = await _cluster_one_memcell(mc, [existing], client, threshold=0.70, time_window_days=7.0)

    assert len(result) == 2
    ids = {c.id for c in result}
    assert ids == {"scene_0", "scene_1"}


def test_serialize_cluster_file_shape() -> None:
    """Output dict must have 'clusters' list and 'memcell_to_cluster' map."""
    clusters = [
        Cluster(
            id="scene_0",
            centroid=np.array([0.5, 0.5], dtype=np.float32),
            count=2,
            last_ts=2000,
            members=["a", "b"],
            preview=["preview text"],
        )
    ]
    out = _serialize_cluster_file(clusters)

    assert "clusters" in out
    assert "memcell_to_cluster" in out

    c = out["clusters"][0]
    assert c["id"] == "scene_0"
    assert isinstance(c["centroid"], list)  # tolist() → plain Python floats
    assert all(isinstance(v, float) for v in c["centroid"])
    assert c["count"] == 2
    assert c["last_ts"] == 2000
    assert c["members"] == ["a", "b"]

    m2c = out["memcell_to_cluster"]
    assert m2c == {"a": "scene_0", "b": "scene_0"}


def test_serialize_cluster_file_is_json_serialisable() -> None:
    """Centroid stored as np.float32 must serialise without TypeError."""
    clusters = [
        Cluster(
            id="scene_0",
            centroid=np.array([0.1, 0.2], dtype=np.float32),
            count=1,
            last_ts=0,
            members=["x"],
            preview=[],
        )
    ]
    out = _serialize_cluster_file(clusters)
    # Must not raise
    serialised = json.dumps(out)
    assert "scene_0" in serialised


@pytest.mark.asyncio
async def test_run_clustering_pass_writes_json(tmp_path: Path) -> None:
    """Clustering pass must write clusters_conv_<i>.json with correct shape."""
    client = _make_embedding_client([1.0, 0.0])
    memcells = [_make_memcell("0", timestamp=1000, content="episode one")]

    await _run_clustering_pass(3, memcells, tmp_path, client, threshold=0.70, time_window_days=7.0)

    out_file = tmp_path / "clusters_conv_3.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert "clusters" in data
    assert "memcell_to_cluster" in data
    assert len(data["clusters"]) == 1
    assert data["clusters"][0]["id"] == "scene_0"
    assert data["memcell_to_cluster"] == {"0": "scene_0"}


@pytest.mark.asyncio
async def test_clustering_skipped_when_disabled(tmp_path: Path) -> None:
    """When enable_clustering=False, no clusters_conv_*.json must be written."""
    from benchmarks.common.stages.extract import _process_conversation

    cfg = BenchmarkConfig(enable_clustering=False)

    # Minimal mock services with a callable embedding (must NOT be invoked).
    mock_embedding = MagicMock()
    mock_embedding.embed = AsyncMock(return_value=[[0.1, 0.2]])
    mock_services = MagicMock()
    mock_services.embedding = mock_embedding

    ctx = MagicMock(spec=StageContext)
    ctx.config = cfg
    ctx.services = mock_services

    fake_payload = [_make_memcell("0", content="something")]

    with (
        patch(
            "benchmarks.common.stages.extract._extract_one_conversation",
            new=AsyncMock(return_value=(fake_payload, 10, 5)),
        ),
    ):
        ok, _pt, _ct = await _process_conversation(
            0,
            MagicMock(),  # conv
            MagicMock(),  # llm
            asyncio.Semaphore(1),
            asyncio.Semaphore(20),
            tmp_path,
            boundary_batch_size=20,
            max_attempts=1,
            ctx=ctx,
        )

    assert ok is True
    assert not (tmp_path / "clusters_conv_0.json").exists()
    mock_embedding.embed.assert_not_awaited()
