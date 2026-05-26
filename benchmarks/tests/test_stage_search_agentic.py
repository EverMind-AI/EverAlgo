"""Tests for Stage 3 run_search_stage (agentic path via aagentic_retrieve)."""

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
from everalgo.retrieval.protocols import AgenticDecision
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

    memcell: dict[str, Any] = {
        "id": "0",
        "timestamp": 0,
        "items": [],
        "episode": {"subject": "alice", "content": "X happened"},
        "atomic_facts": {
            "time": "T",
            "timestamp": 0,
            "atomic_fact": ["alice X"],
            "fact_embeddings": [],
        },
    }

    # Mock aagentic_retrieve to return a Candidate list + AgenticDecision (sufficient, no round 2).
    mock_candidate = Candidate(id="0", score=0.9, metadata={"_doc": memcell, **memcell})
    mock_decision = AgenticDecision(
        is_multi_round=False,
        is_sufficient=True,
        reasoning="sufficient",
        key_information_found=["fact_X"],
    )

    async def fake_aagentic_retrieve(*args: Any, **kwargs: Any) -> tuple[list[Candidate], AgenticDecision]:
        return [mock_candidate], mock_decision

    # Make services use mocks. Disable cluster retrieval: this test exercises the flat
    # agentic path (patches aagentic_retrieve), and the loader now fast-fails when
    # enable_cluster_retrieval=True but the cluster pkl is missing.
    cfg = BenchmarkConfig(enable_cluster_retrieval=False)
    services = Services.from_config(cfg)
    services.embedding.embed = AsyncMock(return_value=[[1.0, 0.0]])  # type: ignore[method-assign]
    services.rerank.rerank = AsyncMock(return_value=[(0, 0.9)])  # type: ignore[method-assign]

    # Mock the algo's aagentic_retrieve at the benchmarks.common.stages.search import site.
    with patch("benchmarks.common.stages.search.aagentic_retrieve", side_effect=fake_aagentic_retrieve):
        # Prepare stage 2 output
        stage2_dir = tmp_path / "stage2_index"
        stage2_dir.mkdir()
        docs = [memcell]
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
                        "doc": memcell,
                        "embeddings": {"subject": np.array([1.0, 0.0], dtype=np.float32)},
                    }
                ],
                f,
            )

        fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
        ctx = StageContext(
            config=cfg,
            services=services,
            dataset=LocomoDataset(data_path=fixture),
            input_dir=stage2_dir,
            output_dir=tmp_path / "stage3_search",
            smoke=True,
        )
        stats = await run_search_stage(ctx)

    assert stats.stage_name == "search"

    out = tmp_path / "stage3_search" / "search_results.json"
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
