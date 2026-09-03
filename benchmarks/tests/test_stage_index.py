"""Tests for Stage 4 index building (entity-split data model)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from benchmarks.common.stages.index import _build_cluster_index, extract_searchable_units, run_index_stage
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
            "episode_ids": ["0", "1"],
            "preview": ["Alice went fishing", "Bob joined"],
        },
        {
            "id": "cluster_1",
            "centroid": [0.4, 0.5, 0.6],
            "count": 1,
            "last_ts": 1_700_100_000_000,
            "episode_ids": ["2"],
            "preview": ["Charlie cooked"],
        },
    ],
    "episode_to_cluster": {
        "0": "cluster_0",
        "1": "cluster_0",
        "2": "cluster_1",
    },
}


def _make_episode(ep_id: str, subject: str, episode_text: str) -> dict[str, Any]:
    """Build a minimal episode dict for testing."""
    return {
        "id": ep_id,
        "owner_id": None,
        "memcell_ids": [ep_id],
        "subject": subject,
        "episode": episode_text,
        "timestamp": 0,
        "embeddings": {
            "episode": [0.1, 0.2, 0.3, 0.4],
            "subject": [0.5, 0.6, 0.7, 0.8],
        },
    }


def _make_fact(af_id: str, episode_id: str, content: str) -> dict[str, Any]:
    """Build a minimal atomic-fact dict for testing."""
    return {
        "id": af_id,
        "episode_id": episode_id,
        "owner_id": None,
        "content": content,
        "timestamp": 0,
        "embeddings": [0.1, 0.2, 0.3, 0.4],
    }


# ---------------------------------------------------------------------------
# Unit: extract_searchable_units
# ---------------------------------------------------------------------------


def test_extract_searchable_units_returns_facts_plus_subject_plus_summary() -> None:
    """Facts + subject + BM25 summary (first 200 chars of episode body)."""
    ep = _make_episode("0", "fishing trip", "Alice caught a fish on the lake")
    facts = [_make_fact("0", "0", "Alice fished"), _make_fact("1", "0", "She caught a trout")]
    units = extract_searchable_units(ep, facts)
    assert "Alice fished" in units
    assert "She caught a trout" in units
    assert "fishing trip" in units
    assert "Alice caught a fish on the lake" in units  # BM25 summary (body[:200])
    assert len(units) == 4


def test_extract_searchable_units_raises_on_empty_facts() -> None:
    """ValueError when no atomic facts exist."""
    ep = _make_episode("0", "fishing trip", "Alice caught a fish")
    with pytest.raises(ValueError, match="No atomic facts"):
        extract_searchable_units(ep, [])


# ---------------------------------------------------------------------------
# Unit: _build_cluster_index
# ---------------------------------------------------------------------------


def test_build_cluster_index_returns_list_of_cluster_dumps() -> None:
    """``_build_cluster_index`` returns ``list[Cluster.model_dump()]`` with episode_ids as members."""
    import numpy as np

    from everalgo.clustering import Cluster

    result = _build_cluster_index(_CLUSTERS_DATA)
    assert isinstance(result, list)
    assert len(result) == 2

    clusters = [Cluster.model_validate(d) for d in result]
    by_id = {c.id: c for c in clusters}
    assert set(by_id) == {"cluster_0", "cluster_1"}
    c0 = by_id["cluster_0"]
    assert c0.members == ["0", "1"]  # episode_ids
    assert c0.count == 2
    assert c0.last_ts == 1_700_000_000_000
    assert np.allclose(c0.centroid, np.array([0.1, 0.2, 0.3]))


# ---------------------------------------------------------------------------
# End-to-end: index stage writes bm25 + emb pickles from entity-split files
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_writes_bm25_and_emb_pickles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: entity-split episode + atomic_facts files -> bm25 + emb pickles."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")

    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.datasets.locomo.loader import LocomoDataset

    stage1_dir = tmp_path / "stage3_enrich"
    stage1_dir.mkdir()

    episodes = [_make_episode("0", "fishing trip", "Alice caught a fish")]
    facts = [_make_fact("0", "0", "Alice fished")]
    clusters = {
        "clusters": [
            {
                "id": "cluster_0",
                "centroid": [0.1] * 10,
                "last_ts": 1000,
                "members": ["0"],
                "preview": ["fishing"],
                "episode_ids": ["0"],
            }
        ],
        "episode_to_cluster": {"0": "cluster_0"},
    }
    (stage1_dir / "episodes_conv_0.json").write_text(json.dumps(episodes))
    (stage1_dir / "atomic_facts_conv_0.json").write_text(json.dumps(facts))
    (stage1_dir / "clusters_conv_0.json").write_text(json.dumps(clusters))

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    cfg = BenchmarkConfig()
    ctx = StageContext(
        config=cfg,
        services=Services.from_config(cfg),
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage1_dir,
        output_dir=tmp_path / "stage4_index",
    )
    stats = await run_index_stage(ctx)
    assert stats.stage_name == "index"
    assert stats.success >= 1
    assert stats.failed == 0

    bm25_pkl = tmp_path / "stage4_index" / "bm25_conv_0.pkl"
    emb_pkl = tmp_path / "stage4_index" / "emb_conv_0.pkl"
    assert bm25_pkl.exists()
    assert emb_pkl.exists()

    # Verify fact-level BM25 payload shape
    with bm25_pkl.open("rb") as f:
        bm25_data = pickle.load(f)
    assert "bm25" in bm25_data
    assert "docs" in bm25_data
    assert "fact_to_doc_idx" in bm25_data
    assert bm25_data["index_type"] == "maxsim"
    assert len(bm25_data["docs"]) == 1
    # 1 atomic_fact + 1 subject + 1 BM25 summary -> 3 fact-rows, all mapping back to doc index 0.
    assert bm25_data["fact_to_doc_idx"] == [0, 0, 0]
    # docs are now episode dicts (not monolithic memcells)
    assert bm25_data["docs"][0]["id"] == "0"
    assert bm25_data["docs"][0]["episode"] == "Alice caught a fish"

    # Verify embedding pickle shape
    with emb_pkl.open("rb") as f:
        emb_data: list[dict[str, Any]] = pickle.load(f)
    assert isinstance(emb_data, list)
    assert len(emb_data) == 1
    item: dict[str, Any] = emb_data[0]
    assert item["doc_id"] == "0"
    assert "embeddings" in item
    embeddings: dict[str, Any] = item["embeddings"]
    assert "atomic_facts" in embeddings
    assert len(embeddings["atomic_facts"]) == 1
    assert "subject" in embeddings
    assert "episode" in embeddings


# ---------------------------------------------------------------------------
# End-to-end: cluster pickle is written when cluster file is present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_index_stage_writes_cluster_pickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When clusters_conv_0.json exists, cluster pickle is written."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")

    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.datasets.locomo.loader import LocomoDataset

    stage1_dir = tmp_path / "stage3_enrich"
    stage1_dir.mkdir()

    episodes = [_make_episode("0", "fishing trip", "Alice caught a fish")]
    facts = [_make_fact("0", "0", "Alice fished")]
    (stage1_dir / "episodes_conv_0.json").write_text(json.dumps(episodes))
    (stage1_dir / "atomic_facts_conv_0.json").write_text(json.dumps(facts))
    (stage1_dir / "clusters_conv_0.json").write_text(json.dumps(_CLUSTERS_DATA))

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    cfg = BenchmarkConfig()
    ctx = StageContext(
        config=cfg,
        services=Services.from_config(cfg),
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage1_dir,
        output_dir=tmp_path / "stage4_index",
    )
    stats = await run_index_stage(ctx)
    assert stats.success >= 1
    assert stats.failed == 0

    cluster_pkl = tmp_path / "stage4_index" / "cluster_index_conv_0.pkl"
    assert cluster_pkl.exists(), "cluster_index_conv_0.pkl was not written"

    from everalgo.clustering import Cluster

    with cluster_pkl.open("rb") as fh:
        raw: list[dict[str, Any]] = pickle.load(fh)
    assert isinstance(raw, list)
    clusters = [Cluster.model_validate(d) for d in raw]
    assert {c.id for c in clusters} == {"cluster_0", "cluster_1"}
    assert sum(c.count for c in clusters) == 3


# ---------------------------------------------------------------------------
# Fast-fail when cluster file is missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_index_stage_raises_when_cluster_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No cluster file: stage raises FileNotFoundError."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")

    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.datasets.locomo.loader import LocomoDataset

    stage1_dir = tmp_path / "stage3_enrich"
    stage1_dir.mkdir()

    episodes = [_make_episode("0", "hiking", "Bob hiked")]
    facts = [_make_fact("0", "0", "Bob hiked a trail")]
    (stage1_dir / "episodes_conv_0.json").write_text(json.dumps(episodes))
    (stage1_dir / "atomic_facts_conv_0.json").write_text(json.dumps(facts))
    # Deliberately omit clusters_conv_0.json.

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    cfg = BenchmarkConfig()
    ctx = StageContext(
        config=cfg,
        services=Services.from_config(cfg),
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage1_dir,
        output_dir=tmp_path / "stage4_index",
    )
    with pytest.raises(FileNotFoundError, match="Cluster file missing"):
        await run_index_stage(ctx)


def test_run_index_stage_callable():
    import inspect

    assert inspect.iscoroutinefunction(run_index_stage)
