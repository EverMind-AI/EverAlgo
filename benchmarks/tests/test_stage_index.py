"""Tests for Stage 2 index building."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pytest import MonkeyPatch

from benchmarks.common.config import BenchmarkConfig
from benchmarks.common.services import Services
from benchmarks.common.stages.index import _build_cluster_index, run_index_stage
from benchmarks.common.stages.types import StageContext

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CLUSTERS_DATA: dict[str, Any] = {
    "clusters": [
        {
            "id": "cluster_0",
            "centroid": [0.1, 0.2, 0.3],
            "count": 2,
            "last_ts": 1_700_000_000_000,
            "members": ["mc_1", "mc_2"],
            "preview": ["Alice went fishing", "Bob joined"],
        },
        {
            "id": "cluster_1",
            "centroid": [0.4, 0.5, 0.6],
            "count": 1,
            "last_ts": 1_700_100_000_000,
            "members": ["mc_3"],
            "preview": ["Charlie cooked"],
        },
    ],
    "memcell_to_cluster": {
        "mc_1": "cluster_0",
        "mc_2": "cluster_0",
        "mc_3": "cluster_1",
    },
}


# ---------------------------------------------------------------------------
# Unit: _build_cluster_index
# ---------------------------------------------------------------------------


def test_build_cluster_index_returns_list_of_cluster_dumps() -> None:
    """``_build_cluster_index`` returns ``list[Cluster.model_dump()]`` aligned with algo schema."""
    import numpy as np

    from everalgo.clustering import Cluster

    result = _build_cluster_index(_CLUSTERS_DATA)
    assert isinstance(result, list)
    assert len(result) == 2

    # Each entry is a Cluster.model_dump() — round-trip via Cluster.model_validate.
    clusters = [Cluster.model_validate(d) for d in result]
    by_id = {c.id: c for c in clusters}
    assert set(by_id) == {"cluster_0", "cluster_1"}
    c0 = by_id["cluster_0"]
    assert c0.members == ["mc_1", "mc_2"]
    assert c0.count == 2
    assert c0.last_ts == 1_700_000_000_000
    assert np.allclose(c0.centroid, np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# End-to-end: cluster pickle is written when cluster file is present
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_run_index_stage_writes_cluster_pickle(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """When enable_cluster_retrieval=True and clusters_conv_0.json exists, cluster pickle is written."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    respx.post("https://api.deepinfra.com/v1/openai/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1] * 4, "index": i} for i in range(2)],
                "model": "Qwen/Qwen3-Embedding-4B",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )
    )

    stage1_dir = tmp_path / "stage1_extract"
    stage1_dir.mkdir()
    (stage1_dir / "memcells_conv_0.json").write_text(
        json.dumps(
            [
                {
                    "id": "mc_1",
                    "timestamp": 0,
                    "items": [],
                    "episode": {"subject": "fishing trip", "content": "Alice caught a fish"},
                    "atomic_facts": {
                        "time": "T",
                        "timestamp": 0,
                        "atomic_fact": ["Alice fished"],
                        "fact_embeddings": [],
                    },
                }
            ]
        )
    )
    # Pre-stage the cluster file produced by Stage 1.
    (stage1_dir / "clusters_conv_0.json").write_text(json.dumps(_CLUSTERS_DATA))

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    from benchmarks.datasets.locomo.loader import LocomoDataset

    cfg = BenchmarkConfig(enable_cluster_retrieval=True)
    ctx = StageContext(
        config=cfg,
        services=Services.from_config(cfg),
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage1_dir,
        output_dir=tmp_path / "stage2_index",
    )
    stats = await run_index_stage(ctx)
    assert stats.success >= 1
    assert stats.failed == 0

    cluster_pkl = tmp_path / "stage2_index" / "cluster_index_conv_0.pkl"
    assert cluster_pkl.exists(), "cluster_index_conv_0.pkl was not written"

    from everalgo.clustering import Cluster

    with cluster_pkl.open("rb") as fh:
        raw: list[dict[str, Any]] = pickle.load(fh)
    assert isinstance(raw, list)
    clusters = [Cluster.model_validate(d) for d in raw]
    assert {c.id for c in clusters} == {"cluster_0", "cluster_1"}
    assert sum(c.count for c in clusters) == 3


# ---------------------------------------------------------------------------
# Fast-fail when cluster file is missing with enable_cluster_retrieval=True
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_run_index_stage_raises_when_cluster_file_missing(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """enable_cluster_retrieval=True but no cluster file: stage raises FileNotFoundError.

    The cluster-index failure is intentionally un-caught so a missing cluster file
    terminates the pipeline rather than silently producing a partial index that
    would corrupt Stage 3 metrics.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    respx.post("https://api.deepinfra.com/v1/openai/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1] * 4, "index": i} for i in range(2)],
                "model": "Qwen/Qwen3-Embedding-4B",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )
    )

    stage1_dir = tmp_path / "stage1_extract"
    stage1_dir.mkdir()
    (stage1_dir / "memcells_conv_0.json").write_text(
        json.dumps(
            [
                {
                    "id": "mc_1",
                    "timestamp": 0,
                    "items": [],
                    "episode": {"subject": "hiking", "content": "Bob hiked"},
                    "atomic_facts": {
                        "time": "T",
                        "timestamp": 0,
                        "atomic_fact": ["Bob hiked a trail"],
                        "fact_embeddings": [],
                    },
                }
            ]
        )
    )
    # Deliberately omit clusters_conv_0.json.

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    from benchmarks.datasets.locomo.loader import LocomoDataset

    cfg = BenchmarkConfig(enable_cluster_retrieval=True)
    ctx = StageContext(
        config=cfg,
        services=Services.from_config(cfg),
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage1_dir,
        output_dir=tmp_path / "stage2_index",
    )
    with pytest.raises(FileNotFoundError, match="enable_cluster_retrieval=True but cluster file missing"):
        await run_index_stage(ctx)


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_index_writes_bm25_and_emb_pickles(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """End-to-end: pre-staged memcell JSON -> bm25 + emb pickles."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")
    # Mock embedding endpoint -- new fact-level scheme: 1 atomic_fact + subject
    # (content is only a fallback when nothing else has been queued).
    respx.post("https://api.deepinfra.com/v1/openai/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.1] * 4, "index": i} for i in range(2)],
                "model": "Qwen/Qwen3-Embedding-4B",
                "usage": {"prompt_tokens": 10, "total_tokens": 10},
            },
        )
    )

    # Prepare Stage 1 output (EverAlgo-native schema)
    stage1_dir = tmp_path / "stage1_extract"
    stage1_dir.mkdir()
    (stage1_dir / "memcells_conv_0.json").write_text(
        json.dumps(
            [
                {
                    "id": "0",
                    "timestamp": 0,
                    "items": [
                        {
                            "id": "m1",
                            "role": "user",
                            "content": "Alice went fishing",
                            "timestamp": 0,
                            "sender_id": "u_alice",
                            "sender_name": "Alice",
                        }
                    ],
                    "episode": {
                        "subject": "fishing trip",
                        "content": "Alice caught a fish",
                    },
                    "atomic_facts": {
                        "time": "T",
                        "timestamp": 0,
                        "atomic_fact": ["Alice fished"],
                        "fact_embeddings": [],
                    },
                }
            ]
        )
    )

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    from benchmarks.datasets.locomo.loader import LocomoDataset

    cfg = BenchmarkConfig(enable_cluster_retrieval=False)
    ctx = StageContext(
        config=cfg,
        services=Services.from_config(cfg),
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage1_dir,
        output_dir=tmp_path / "stage2_index",
    )
    stats = await run_index_stage(ctx)
    assert stats.stage_name == "index"
    assert stats.success >= 1
    assert stats.failed == 0

    bm25_pkl = tmp_path / "stage2_index" / "bm25_conv_0.pkl"
    emb_pkl = tmp_path / "stage2_index" / "emb_conv_0.pkl"
    assert bm25_pkl.exists()
    assert emb_pkl.exists()

    # Verify fact-level BM25 payload shape: bm25 + docs + fact_to_doc_idx + index_type
    with bm25_pkl.open("rb") as f:
        bm25_data = pickle.load(f)
    assert "bm25" in bm25_data
    assert "docs" in bm25_data
    assert "fact_to_doc_idx" in bm25_data
    assert bm25_data["index_type"] == "maxsim"
    assert len(bm25_data["docs"]) == 1
    # 1 atomic_fact + 1 subject -> 2 fact-rows, both mapping back to doc index 0.
    assert bm25_data["fact_to_doc_idx"] == [0, 0]

    # Verify embedding pickle shape: atomic_facts + subject, NO content (only
    # populated as a final fallback when nothing else is queued).
    with emb_pkl.open("rb") as f:
        emb_data: list[dict[str, Any]] = pickle.load(f)
    assert isinstance(emb_data, list)
    assert len(emb_data) == 1
    item: dict[str, Any] = emb_data[0]
    assert "doc" in item
    assert "embeddings" in item
    embeddings: dict[str, Any] = item["embeddings"]
    assert "atomic_facts" in embeddings
    assert len(embeddings["atomic_facts"]) == 1
    assert "subject" in embeddings
    assert "summary" not in embeddings
    # 93 alignment: "episode" fallback row is only emitted when atomic_facts is missing.
    # This fixture has 1 atomic_fact so the fallback path is suppressed.
    assert "episode" not in embeddings
    assert "content" not in embeddings  # legacy field name (pre-93-alignment); confirmed dropped


def test_run_index_stage_callable():
    import inspect

    assert inspect.iscoroutinefunction(run_index_stage)
