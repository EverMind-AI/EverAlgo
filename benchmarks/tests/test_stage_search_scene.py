"""Tests for Stage 3 scene-agentic retrieval path (scene_search.py + search.py routing)."""

from __future__ import annotations

import inspect
import pickle
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from benchmarks.common.stages.scene_search import (
    scene_agentic_retrieval,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_docs(n: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "id": str(i),
            "episode": {"subject": f"subject_{i}", "content": f"content about topic {i}"},
            "atomic_facts": [{"fact": f"fact_{i}"}],
        }
        for i in range(n)
    ]


def _make_bm25_index(docs: list[dict[str, Any]]) -> dict[str, Any]:
    corpus = [[f"fact_{i}"] for i in range(len(docs))]
    return {
        "bm25": BM25Okapi(corpus),
        "docs": docs,
        "fact_to_doc_idx": list(range(len(docs))),
        "index_type": "maxsim",
    }


def _make_emb_index(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "doc": doc,
            "embeddings": {"subject": np.array([1.0, 0.0], dtype=np.float32)},
        }
        for doc in docs
    ]


def _make_scene_index(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """One scene containing all docs."""
    mc_ids = [str(d["id"]) for d in docs]
    scenes = [
        {
            "scene_id": "scene_0",
            "centroid": [1.0, 0.0],
            "memcell_ids": mc_ids,
            "memcell_count": len(mc_ids),
            "last_timestamp": 0,
        }
    ]
    return {
        "scenes": scenes,
        "memcell_to_scene": dict.fromkeys(mc_ids, "scene_0"),
        "total_scenes": 1,
        "total_memcells": len(mc_ids),
    }


def _make_config() -> Any:
    from benchmarks.common.config import BenchmarkConfig

    return BenchmarkConfig()


# ---------------------------------------------------------------------------
# Test 1: sufficient path — results are in-scene docs, ≤ response_top_k
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scene_agentic_retrieval_returns_in_scene_results_when_sufficient() -> None:
    """When sufficiency=True, returned results come only from in-scene docs and count <= response_top_k."""
    docs = _make_docs(3)
    bm25_index = _make_bm25_index(docs)
    emb_index = _make_emb_index(docs)
    scene_index = _make_scene_index(docs)
    config = _make_config()

    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(return_value=[[1.0, 0.0]])
    rerank_client = MagicMock()
    # Reranker returns one scored result
    rerank_client.rerank = AsyncMock(return_value=[(0, 0.95)])
    llm = MagicMock()

    async def fake_sufficient(*args: Any, **kwargs: Any) -> tuple[bool, str, list[str], list[str], dict[str, int]]:
        return (True, "sufficient", [], ["key_fact"], {"prompt_tokens": 10, "completion_tokens": 5})

    with patch("benchmarks.common.stages._agentic_utils.check_sufficiency", side_effect=fake_sufficient):
        results, metadata = await scene_agentic_retrieval(
            "query about topic",
            scene_index=scene_index,
            emb_index=emb_index,
            bm25_index=bm25_index,
            config=config,
            llm=llm,
            embedding_client=embedding_client,
            rerank_client=rerank_client,
        )

    assert metadata["is_sufficient"] is True
    assert metadata["is_multi_round"] is False
    assert len(results) <= config.response_top_k
    # All returned doc IDs must be from the scene's memcell set
    scene_ids = {str(d["id"]) for d in docs}
    for doc, _score in results:
        assert str(doc.get("id", "")) in scene_ids
    assert metadata["prompt_tokens"] == 10
    assert metadata["completion_tokens"] == 5


# ---------------------------------------------------------------------------
# Test 2: insufficient → round 2 searches FULL corpus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scene_agentic_retrieval_round2_searches_full_corpus() -> None:
    """When sufficiency=False, hybrid_search_with_rrf is called for each refined query on the full corpus."""
    docs = _make_docs(5)
    # Scene covers only the first 2 docs
    scene_docs = docs[:2]
    out_of_scene_docs = docs[2:]
    scene_index = _make_scene_index(scene_docs)
    # bm25_index and emb_index contain ALL docs (full corpus)
    bm25_index = _make_bm25_index(docs)
    emb_index = _make_emb_index(docs)
    config = _make_config()

    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(return_value=[[1.0, 0.0]])
    rerank_client = MagicMock()
    rerank_client.rerank = AsyncMock(return_value=[(0, 0.85)])
    llm = MagicMock()

    async def fake_insufficient(*args: Any, **kwargs: Any) -> tuple[bool, str, list[str], list[str], dict[str, int]]:
        return (False, "missing info", ["missing X"], [], {})

    async def fake_multi_queries(*args: Any, **kwargs: Any) -> tuple[list[str], str, dict[str, int]]:
        return (["refined query 1", "refined query 2"], "strategy", {})

    # Track how hybrid_search_with_rrf is called and what emb_index it receives
    hybrid_calls: list[dict[str, Any]] = []

    async def fake_hybrid(query: str, *, emb_index: Any, bm25_index: Any, **kwargs: Any) -> list[Any]:
        hybrid_calls.append({"query": query, "emb_index_len": len(emb_index), "bm25_docs_len": len(bm25_index["docs"])})
        # Return one result from out-of-scene doc to confirm full-corpus search
        return [(out_of_scene_docs[0], 0.5)]

    with (
        patch("benchmarks.common.stages._agentic_utils.check_sufficiency", side_effect=fake_insufficient),
        patch("benchmarks.common.stages._agentic_utils.generate_multi_queries", side_effect=fake_multi_queries),
        patch("benchmarks.common.stages.scene_search.hybrid_search_with_rrf", side_effect=fake_hybrid),
    ):
        _results, metadata = await scene_agentic_retrieval(
            "query missing info",
            scene_index=scene_index,
            emb_index=emb_index,
            bm25_index=bm25_index,
            config=config,
            llm=llm,
            embedding_client=embedding_client,
            rerank_client=rerank_client,
        )

    assert metadata["is_multi_round"] is True
    # hybrid_search_with_rrf called once per refined query
    assert len(hybrid_calls) == 2
    # Each call received the FULL emb_index (5 docs), not just the 2 scene docs
    for call in hybrid_calls:
        assert call["emb_index_len"] == len(docs)
        assert call["bm25_docs_len"] == len(docs)
    assert metadata["refined_queries"] == ["refined query 1", "refined query 2"]


# ---------------------------------------------------------------------------
# Test 3: run_search_stage routes to scene path when scene index pickle present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_search_stage_routes_to_scene_when_index_present(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """When scene_index pickle exists and enable_scene_retrieval=True, scene_agentic_retrieval is called."""
    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.common.stages.search import run_search_stage
    from benchmarks.common.stages.types import StageContext
    from benchmarks.datasets.locomo.loader import LocomoDataset

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")

    docs = _make_docs(2)
    bm25_index = _make_bm25_index(docs)
    emb_index = _make_emb_index(docs)
    scene_index = _make_scene_index(docs)

    stage2_dir = tmp_path / "stage2"
    stage2_dir.mkdir()

    with (stage2_dir / "bm25_conv_0.pkl").open("wb") as f:
        pickle.dump(bm25_index, f)
    with (stage2_dir / "emb_conv_0.pkl").open("wb") as f:
        pickle.dump(emb_index, f)
    with (stage2_dir / "scene_index_conv_0.pkl").open("wb") as f:
        pickle.dump(scene_index, f)

    cfg = BenchmarkConfig(enable_scene_retrieval=True)
    services = Services.from_config(cfg)
    services.embedding.embed = AsyncMock(return_value=[[1.0, 0.0]])  # type: ignore[method-assign]
    services.rerank.rerank = AsyncMock(return_value=[(0, 0.9)])  # type: ignore[method-assign]

    scene_called = []
    agentic_called = []

    async def fake_scene(query: str, *, scene_index: Any, **kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
        scene_called.append(query)
        return ([], {"retrieval_mode": "scene_agentic", "prompt_tokens": 0, "completion_tokens": 0})

    async def fake_agentic(query: str, **kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
        agentic_called.append(query)
        return ([], {"retrieval_mode": "agentic", "prompt_tokens": 0, "completion_tokens": 0})

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    ctx = StageContext(
        config=cfg,
        services=services,
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage2_dir,
        output_dir=tmp_path / "out",
        smoke=True,
    )

    with (
        patch("benchmarks.common.stages.scene_search.scene_agentic_retrieval", side_effect=fake_scene),
        patch("benchmarks.common.stages.search.agentic_retrieval", side_effect=fake_agentic),
    ):
        await run_search_stage(ctx)

    assert len(scene_called) >= 1, "scene_agentic_retrieval should have been called"
    assert len(agentic_called) == 0, "flat agentic_retrieval must NOT be called when scene index is present"


# ---------------------------------------------------------------------------
# Test 4: run_search_stage falls back to agentic when scene index missing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_search_stage_falls_back_when_scene_missing(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """When no scene pickle exists, the flat agentic_retrieval path is taken."""
    from benchmarks.common.config import BenchmarkConfig
    from benchmarks.common.services import Services
    from benchmarks.common.stages.search import run_search_stage
    from benchmarks.common.stages.types import StageContext
    from benchmarks.datasets.locomo.loader import LocomoDataset

    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test")

    docs = _make_docs(2)
    bm25_index = _make_bm25_index(docs)
    emb_index = _make_emb_index(docs)

    stage2_dir = tmp_path / "stage2"
    stage2_dir.mkdir()

    with (stage2_dir / "bm25_conv_0.pkl").open("wb") as f:
        pickle.dump(bm25_index, f)
    with (stage2_dir / "emb_conv_0.pkl").open("wb") as f:
        pickle.dump(emb_index, f)
    # Intentionally omit scene_index_conv_0.pkl

    cfg = BenchmarkConfig(enable_scene_retrieval=True)
    services = Services.from_config(cfg)
    services.embedding.embed = AsyncMock(return_value=[[1.0, 0.0]])  # type: ignore[method-assign]
    services.rerank.rerank = AsyncMock(return_value=[(0, 0.9)])  # type: ignore[method-assign]

    scene_called = []
    agentic_called = []

    async def fake_scene(query: str, *, scene_index: Any, **kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
        scene_called.append(query)
        return ([], {"retrieval_mode": "scene_agentic", "prompt_tokens": 0, "completion_tokens": 0})

    async def fake_agentic(query: str, **kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
        agentic_called.append(query)
        return ([], {"retrieval_mode": "agentic", "prompt_tokens": 0, "completion_tokens": 0})

    fixture = Path(__file__).parent / "fixtures" / "locomo_mini.json"
    ctx = StageContext(
        config=cfg,
        services=services,
        dataset=LocomoDataset(data_path=fixture),
        input_dir=stage2_dir,
        output_dir=tmp_path / "out",
        smoke=True,
    )

    with (
        patch("benchmarks.common.stages.scene_search.scene_agentic_retrieval", side_effect=fake_scene),
        patch("benchmarks.common.stages.search.agentic_retrieval", side_effect=fake_agentic),
    ):
        await run_search_stage(ctx)

    assert len(agentic_called) >= 1, "agentic_retrieval should be called when scene index is absent"
    assert len(scene_called) == 0, "scene_agentic_retrieval must NOT be called when scene index is missing"


# ---------------------------------------------------------------------------
# Structural sanity
# ---------------------------------------------------------------------------


def test_scene_agentic_retrieval_is_async() -> None:
    assert inspect.iscoroutinefunction(scene_agentic_retrieval)
