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
from benchmarks.common.stages.index import _build_scene_index, run_index_stage
from benchmarks.common.stages.types import StageContext

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CLUSTERS_DATA: dict[str, Any] = {
    "clusters": [
        {
            "id": "scene_0",
            "centroid": [0.1, 0.2, 0.3],
            "count": 2,
            "last_ts": 1_700_000_000_000,
            "members": ["mc_1", "mc_2"],
            "preview": ["Alice went fishing", "Bob joined"],
        },
        {
            "id": "scene_1",
            "centroid": [0.4, 0.5, 0.6],
            "count": 1,
            "last_ts": 1_700_100_000_000,
            "members": ["mc_3"],
            "preview": ["Charlie cooked"],
        },
    ],
    "memcell_to_cluster": {
        "mc_1": "scene_0",
        "mc_2": "scene_0",
        "mc_3": "scene_1",
    },
}


# ---------------------------------------------------------------------------
# Unit: _build_scene_index
# ---------------------------------------------------------------------------


def test_build_scene_index_reshape() -> None:
    """Pure reshape: cluster JSON -> scene-index dict has the exact target shape."""
    result = _build_scene_index(_CLUSTERS_DATA)

    assert result["total_scenes"] == 2
    assert result["total_memcells"] == 3
    assert result["memcell_to_scene"] == {
        "mc_1": "scene_0",
        "mc_2": "scene_0",
        "mc_3": "scene_1",
    }

    scenes: list[dict[str, Any]] = result["scenes"]
    assert len(scenes) == 2

    s0 = next(s for s in scenes if s["scene_id"] == "scene_0")
    assert s0["centroid"] == [0.1, 0.2, 0.3]
    assert s0["memcell_ids"] == ["mc_1", "mc_2"]
    assert s0["memcell_count"] == 2
    assert s0["last_timestamp"] == 1_700_000_000_000

    s1 = next(s for s in scenes if s["scene_id"] == "scene_1")
    assert s1["centroid"] == [0.4, 0.5, 0.6]
    assert s1["memcell_ids"] == ["mc_3"]
    assert s1["memcell_count"] == 1
    assert s1["last_timestamp"] == 1_700_100_000_000

    # Centroid must be plain list[float], not numpy array.
    assert isinstance(s0["centroid"], list)
    assert all(isinstance(v, float) for v in s0["centroid"])


# ---------------------------------------------------------------------------
# End-to-end: scene pickle is written when cluster file is present
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_run_index_stage_writes_scene_pickle(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """When enable_scene_retrieval=True and clusters_conv_0.json exists, scene pickle is written."""
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
                    "atomic_facts": [{"fact": "Alice fished"}],
                }
            ]
        )
    )
    # Pre-stage the cluster file produced by Stage 1.
    (stage1_dir / "clusters_conv_0.json").write_text(json.dumps(_CLUSTERS_DATA))

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    from benchmarks.datasets.locomo.loader import LocomoDataset

    cfg = BenchmarkConfig(enable_scene_retrieval=True)
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

    scene_pkl = tmp_path / "stage2_index" / "scene_index_conv_0.pkl"
    assert scene_pkl.exists(), "scene_index_conv_0.pkl was not written"

    with scene_pkl.open("rb") as fh:
        loaded: dict[str, Any] = pickle.load(fh)

    assert loaded["total_scenes"] == 2
    assert loaded["total_memcells"] == 3
    assert set(loaded["memcell_to_scene"].keys()) == {"mc_1", "mc_2", "mc_3"}
    assert len(loaded["scenes"]) == 2
    scene_ids = {s["scene_id"] for s in loaded["scenes"]}
    assert scene_ids == {"scene_0", "scene_1"}
    # Centroid survives the pickle round-trip as plain list.
    for scene in loaded["scenes"]:
        assert isinstance(scene["centroid"], list)


# ---------------------------------------------------------------------------
# Graceful skip when cluster file is missing
# ---------------------------------------------------------------------------


@respx.mock
@pytest.mark.asyncio
async def test_run_index_stage_skips_scene_when_cluster_file_missing(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """enable_scene_retrieval=True but no cluster file: no scene pkl, no crash."""
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
                    "atomic_facts": [{"fact": "Bob hiked a trail"}],
                }
            ]
        )
    )
    # Deliberately omit clusters_conv_0.json.

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    from benchmarks.datasets.locomo.loader import LocomoDataset

    cfg = BenchmarkConfig(enable_scene_retrieval=True)
    ctx = StageContext(
        config=cfg,
        services=Services.from_config(cfg),
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage1_dir,
        output_dir=tmp_path / "stage2_index",
    )
    # Must not raise.
    stats = await run_index_stage(ctx)
    assert stats.success >= 1
    assert stats.failed == 0

    # BM25 + emb are still written.
    assert (tmp_path / "stage2_index" / "bm25_conv_0.pkl").exists()
    assert (tmp_path / "stage2_index" / "emb_conv_0.pkl").exists()
    # No scene index, no error sidecar.
    assert not (tmp_path / "stage2_index" / "scene_index_conv_0.pkl").exists()
    assert not (tmp_path / "stage2_index" / "scene_index_conv_0.error.txt").exists()


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
                    "atomic_facts": [{"fact": "Alice fished"}],
                }
            ]
        )
    )

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    from benchmarks.datasets.locomo.loader import LocomoDataset

    cfg = BenchmarkConfig()
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
    assert "content" not in embeddings  # not embedded when fact / subject already present
    assert "episode" not in embeddings  # episode is now a nested dict, not a string field


def test_run_index_stage_callable():
    import inspect

    assert inspect.iscoroutinefunction(run_index_stage)
