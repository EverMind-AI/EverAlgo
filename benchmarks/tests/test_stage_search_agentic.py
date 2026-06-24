"""Tests for Stage 3 run_search_stage (agentic path via aagentic_retrieve, entity-split model)."""

from __future__ import annotations

import inspect
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

from benchmarks.common.stages.search import run_search_stage
from everalgo.rank.protocols import AgenticDecision
from everalgo.types import Candidate


def test_run_search_stage_is_async() -> None:
    assert inspect.iscoroutinefunction(run_search_stage)


@pytest.mark.asyncio
async def test_run_search_stage_writes_search_results_json(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """End-to-end with mocks: stage2 input → search_results.json output."""
    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.common.stages.types import StageContext
    from benchmarks.datasets.locomo.loader import LocomoDataset

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")

    # Episode dict (entity-split model: flat fields, no nesting)
    episode: dict[str, Any] = {
        "id": "0",
        "owner_id": None,
        "memcell_ids": ["0"],
        "subject": "alice",
        "episode": "X happened",
        "timestamp": 0,
        "embeddings": {"episode": [1.0, 0.0], "subject": [1.0, 0.0]},
    }

    # Mock aagentic_retrieve to return a Candidate list + AgenticDecision (sufficient, no round 2).
    mock_candidate = Candidate(id="0", score=0.9, metadata={"_doc": episode, **episode})
    mock_decision = AgenticDecision(
        is_multi_round=False,
        is_sufficient=True,
        reasoning="sufficient",
        key_information_found=["fact_X"],
    )

    async def fake_aagentic_retrieve(*args: Any, **kwargs: Any) -> tuple[list[Candidate], AgenticDecision]:
        return [mock_candidate], mock_decision

    cfg = BenchmarkConfig()
    services = Services.from_config(cfg)
    services.embedding.embed = AsyncMock(return_value=[[1.0, 0.0]])  # type: ignore[method-assign]
    services.rerank.rerank = AsyncMock(return_value=[(0, 0.9)])  # type: ignore[method-assign]

    with patch("benchmarks.common.stages.search.aagentic_retrieve", side_effect=fake_aagentic_retrieve):
        # Prepare stage 2 output (entity-split format)
        stage2_dir = tmp_path / "stage4_index"
        stage2_dir.mkdir()
        docs = [episode]
        bm25 = BM25Okapi([["alice"]])
        with (stage2_dir / "bm25_conv_0.pkl").open("wb") as f:
            pickle.dump(
                {
                    "bm25": bm25,
                    "docs": docs,
                    "fact_to_doc_idx": [0],
                    "index_type": "maxsim",
                },
                f,
            )
        with (stage2_dir / "emb_conv_0.pkl").open("wb") as f:
            pickle.dump(
                [
                    {
                        "doc_id": "0",
                        "embeddings": {"subject": np.array([1.0, 0.0], dtype=np.float32)},
                    }
                ],
                f,
            )
        from everalgo.clustering import Cluster

        cluster = Cluster(
            id="cluster_0",
            centroid=np.array([1.0, 0.0], dtype=np.float32),
            last_ts=1000,
            members=["0"],
            preview=["fishing"],
        )
        with (stage2_dir / "cluster_index_conv_0.pkl").open("wb") as f:
            pickle.dump([cluster.model_dump()], f)

        fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
        ctx = StageContext(
            config=cfg,
            services=services,
            dataset=LocomoDataset(data_path=fixture),
            input_dir=stage2_dir,
            output_dir=tmp_path / "stage5_search",
            smoke=True,
        )
        stats = await run_search_stage(ctx)

    assert stats.stage_name == "search"

    out = tmp_path / "stage5_search" / "search_results.json"
    assert out.exists()
    data: dict[str, Any] = json.loads(out.read_text())
    assert "locomo_exp_user_0" in data
    items: list[dict[str, Any]] = data["locomo_exp_user_0"]
    assert len(items) >= 1
    item = items[0]
    for field in ("question_id", "query", "members", "original_qa", "retrieval_metadata"):
        assert field in item, f"missing field: {field}"
    # Category 5 (adversarial) should be filtered out
    for it in items:
        assert it["original_qa"]["category"] != "5"
