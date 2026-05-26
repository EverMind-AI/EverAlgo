"""Tests for Stage 3 agentic retrieval + run_search_stage."""

from __future__ import annotations

import inspect
import json
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

from benchmarks.common.stages.search import agentic_retrieval, run_search_stage


def test_run_search_stage_is_async() -> None:
    assert inspect.iscoroutinefunction(run_search_stage)


def test_agentic_retrieval_is_async() -> None:
    assert inspect.iscoroutinefunction(agentic_retrieval)


@pytest.mark.asyncio
async def test_agentic_retrieval_returns_round1_when_sufficient(monkeypatch: MonkeyPatch) -> None:
    """If LLM judges Round 1 sufficient, return reranked top 10 without Round 2."""
    from benchmarks.common.config import BenchmarkConfig

    # Mock check_sufficiency to return sufficient=True
    async def fake_sufficiency(*args: Any, **kwargs: Any) -> tuple[bool, str, list[str], list[str], dict[str, int]]:
        return (True, "all needed info present", [], ["fact_X"], {})

    monkeypatch.setattr(
        "benchmarks.common.stages._agentic_utils.check_sufficiency",
        fake_sufficiency,
    )

    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(return_value=[[1.0, 0.0]])

    rerank_client = MagicMock()
    rerank_client.rerank = AsyncMock(return_value=[(0, 0.9)])

    llm = MagicMock()

    docs: list[Any] = [{"id": "0", "episode": {"subject": "alice", "content": "alice did x"}, "atomic_facts": []}]
    bm25_index: dict[str, Any] = {
        "bm25": BM25Okapi([["alice"]]),
        "docs": docs,
        "fact_to_doc_idx": [0],
        "index_type": "maxsim",
    }
    emb_index: list[Any] = [
        {
            "doc": docs[0],
            "embeddings": {"subject": np.array([1.0, 0.0], dtype=np.float32)},
        }
    ]

    cfg = BenchmarkConfig()
    final, meta = await agentic_retrieval(
        "find alice",
        config=cfg,
        llm=llm,
        embedding_client=embedding_client,
        rerank_client=rerank_client,
        emb_index=emb_index,
        bm25_index=bm25_index,
    )
    assert meta["is_sufficient"] is True
    assert meta["is_multi_round"] is False
    assert len(final) >= 1


@pytest.mark.asyncio
async def test_run_search_stage_writes_search_results_json(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """End-to-end with mocks: stage2 input → search_results.json output."""
    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.common.stages.types import StageContext
    from benchmarks.datasets.locomo.loader import LocomoDataset

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")

    # Mock all LLM/embedding/rerank calls
    async def fake_sufficiency(*args: Any, **kwargs: Any) -> tuple[bool, str, list[str], list[str], dict[str, int]]:
        return (True, "ok", [], [], {})

    monkeypatch.setattr(
        "benchmarks.common.stages._agentic_utils.check_sufficiency",
        fake_sufficiency,
    )

    # Make services use mocks
    cfg = BenchmarkConfig()
    services = Services.from_config(cfg)
    services.embedding.embed = AsyncMock(return_value=[[1.0, 0.0]])  # type: ignore[method-assign]
    services.rerank.rerank = AsyncMock(return_value=[(0, 0.9)])  # type: ignore[method-assign]
    services.llm.chat = AsyncMock(return_value=MagicMock(content='{"is_correct": true}'))  # type: ignore[method-assign]

    # Prepare stage 2 output
    stage2_dir = tmp_path / "stage2_index"
    stage2_dir.mkdir()
    memcell: dict[str, Any] = {
        "id": "0",
        "timestamp": 0,
        "items": [],
        "episode": {"subject": "alice", "content": "X happened"},
        "atomic_facts": [{"fact": "alice X"}],
    }
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
    for field in ("question_id", "query", "memcell_ids", "original_qa", "retrieval_metadata"):
        assert field in item, f"missing field: {field}"
    # Category 5 (adversarial) should be filtered out
    for it in items:
        assert it["original_qa"]["category"] != "5"
