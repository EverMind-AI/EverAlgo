"""Tests for hybrid retrieval + reranker orchestration."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from benchmarks.common.stages.search import (
    _format_doc_for_rerank,
    hybrid_search_with_rrf,
    reranker_search,
)


def test_format_doc_prefers_episode_content():
    """Mirror locomo-benchmark generic fallback: episode body is the first match."""
    doc = {
        "episode": {"subject": "birthday", "content": "Alice's birthday party"},
        "atomic_facts": [{"fact": "Alice ate cake"}],
    }
    assert _format_doc_for_rerank(doc) == "Alice's birthday party"


def test_format_doc_falls_back_to_first_atomic_fact():
    """Episode content empty -> step to atomic_facts; only the first fact is returned."""
    doc = {
        "atomic_facts": [
            {"fact": "Alice ate cake"},
            {"fact": "It was chocolate"},
        ],
    }
    assert _format_doc_for_rerank(doc) == "Alice ate cake"


def test_format_doc_falls_back_to_summary_then_subject():
    """No content, no facts -> walk summary -> subject."""
    assert _format_doc_for_rerank({"episode": {"summary": "s", "subject": "x"}}) == "s"
    assert _format_doc_for_rerank({"episode": {"subject": "x"}}) == "x"


def test_format_doc_returns_none_when_empty():
    """All probe fields absent -> None."""
    assert _format_doc_for_rerank({}) is None
    assert _format_doc_for_rerank({"atomic_facts": []}) is None
    assert _format_doc_for_rerank({"episode": {"subject": "", "content": ""}}) is None


def test_format_doc_accepts_legacy_string_episode():
    """Tolerate the legacy schema where episode was a plain string."""
    assert _format_doc_for_rerank({"episode": "raw episode body"}) == "raw episode body"


def _build_bm25_index(
    docs: list[dict[str, Any]], fact_corpus: list[list[str]], fact_to_doc_idx: list[int]
) -> dict[str, Any]:
    """Helper: assemble a stage-2-shaped fact-level BM25 payload for the tests."""
    return {
        "bm25": BM25Okapi(fact_corpus),
        "docs": docs,
        "fact_to_doc_idx": fact_to_doc_idx,
        "index_type": "maxsim",
    }


@pytest.mark.asyncio
async def test_hybrid_search_runs_emb_and_bm25_in_parallel():
    """Verify both branches called, results fused."""
    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(return_value=[[1.0, 0.0]])

    emb_index = [
        {
            "doc": {"id": "0"},
            "embeddings": {"subject": np.array([1.0, 0.0], dtype=np.float32)},
        }
    ]
    docs = [{"id": "0"}, {"id": "1"}]
    bm25_index = _build_bm25_index(docs, [["alice"], ["bob"]], [0, 1])

    out = await hybrid_search_with_rrf(
        "alice",
        emb_index=emb_index,
        bm25_index=bm25_index,
        embedding_client=embedding_client,
        top_n=10,
        rrf_k=60,
    )
    assert len(out) >= 1
    # doc "0" should rank highest (in both emb and bm25 top results)
    assert out[0][0]["id"] == "0"


@pytest.mark.asyncio
async def test_hybrid_search_empty_emb_falls_back_to_bm25():
    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(return_value=[[0.0, 1.0]])

    emb_index: list[dict[str, Any]] = []
    docs = [{"id": "0"}, {"id": "1"}]
    bm25_index = _build_bm25_index(docs, [["alice"], ["bob"]], [0, 1])

    out = await hybrid_search_with_rrf(
        "alice",
        emb_index=emb_index,
        bm25_index=bm25_index,
        embedding_client=embedding_client,
        top_n=10,
    )
    assert len(out) >= 1


@pytest.mark.asyncio
async def test_reranker_search_returns_top_n_by_rerank_score():
    """3 docs, reranker scores [0.2, 0.9, 0.5], expect top-2 = [doc1, doc2]."""
    rerank_client = MagicMock()
    rerank_client.rerank = AsyncMock(return_value=[(1, 0.9), (2, 0.5), (0, 0.2)])

    docs = [
        ({"id": "0", "episode": {"subject": "a", "content": "doc a body"}}, 0.5),
        ({"id": "1", "episode": {"subject": "b", "content": "doc b body"}}, 0.4),
        ({"id": "2", "episode": {"subject": "c", "content": "doc c body"}}, 0.3),
    ]
    out = await reranker_search(
        "q",
        results=docs,
        rerank_client=rerank_client,
        top_n=2,
        batch_size=10,
        concurrent_batches=1,
    )
    assert len(out) == 2
    assert out[0][0]["id"] == "1"  # highest rerank score
    assert out[1][0]["id"] == "2"


@pytest.mark.asyncio
async def test_reranker_search_all_failures_fallback_to_original():
    """When all batches fail, return original ranking top-n."""
    rerank_client = MagicMock()
    rerank_client.rerank = AsyncMock(side_effect=RuntimeError("API down"))

    docs = [
        ({"id": "0", "episode": {"subject": "x", "content": "x body"}}, 0.5),
        ({"id": "1", "episode": {"subject": "y", "content": "y body"}}, 0.4),
    ]
    out = await reranker_search(
        "q",
        results=docs,
        rerank_client=rerank_client,
        top_n=2,
        batch_size=10,
        concurrent_batches=1,
        max_retries=1,
        retry_delay=0.001,
    )
    # Fallback to original order
    assert out == docs[:2]


@pytest.mark.asyncio
async def test_reranker_search_empty_input_returns_empty():
    rerank_client = MagicMock()
    out = await reranker_search(
        "q",
        results=[],
        rerank_client=rerank_client,
        top_n=5,
    )
    assert out == []


@pytest.mark.asyncio
async def test_reranker_search_docs_without_text_filtered_out():
    """Docs with no atomic_fact or episode should not be sent to reranker."""
    rerank_client = MagicMock()
    rerank_client.rerank = AsyncMock(return_value=[(0, 0.9)])

    docs: list[tuple[dict[str, Any], float]] = [
        ({"id": "0"}, 0.5),  # no text
        ({"id": "1", "episode": {"subject": "hi", "content": "hello"}}, 0.4),  # usable
    ]
    out = await reranker_search(
        "q",
        results=docs,
        rerank_client=rerank_client,
        top_n=5,
        batch_size=10,
        concurrent_batches=1,
    )
    # Only the one with text was reranked
    assert len(out) == 1
    assert out[0][0]["id"] == "1"
    rerank_client.rerank.assert_called_once()
    # Verify the reranker only saw the one usable doc
    call_args = rerank_client.rerank.call_args
    assert call_args.kwargs.get("documents") == ["hello"] or call_args.args[1] == ["hello"]
