"""Tests for reranker orchestration (hybrid_search_with_rrf was deleted; ported to algo tests)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmarks.common.stages.search import (
    _format_doc_for_rerank,
    reranker_search,
)


def test_format_doc_prefers_episode_content():
    """Generic fallback: episode body is the first match."""
    doc = {
        "episode": {"subject": "birthday", "content": "Alice's birthday party"},
        "atomic_facts": {"atomic_fact": ["Alice ate cake"]},
    }
    assert _format_doc_for_rerank(doc) == "Alice's birthday party"


def test_format_doc_falls_back_to_first_atomic_fact():
    """Episode content empty -> step to atomic_facts; only the first fact is returned.

    Stage 1 emits ``atomic_facts`` as a dict
    ``{"time", "timestamp", "atomic_fact": list[str], "fact_embeddings": ...}``.
    """
    doc = {
        "atomic_facts": {
            "atomic_fact": ["Alice ate cake", "It was chocolate"],
        },
    }
    assert _format_doc_for_rerank(doc) == "Alice ate cake"


def test_format_doc_falls_back_to_summary_then_subject():
    """No content, no facts -> walk summary -> subject."""
    assert _format_doc_for_rerank({"episode": {"summary": "s", "subject": "x"}}) == "s"
    assert _format_doc_for_rerank({"episode": {"subject": "x"}}) == "x"


def test_format_doc_returns_none_when_empty():
    """All probe fields absent -> None."""
    assert _format_doc_for_rerank({}) is None
    assert _format_doc_for_rerank({"atomic_facts": {"atomic_fact": []}}) is None
    assert _format_doc_for_rerank({"episode": {"subject": "", "content": ""}}) is None


def test_format_doc_accepts_legacy_string_episode():
    """Tolerate the legacy schema where episode was a plain string."""
    assert _format_doc_for_rerank({"episode": "raw episode body"}) == "raw episode body"


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
async def test_reranker_search_raises_when_all_batches_fail():
    """Fail-loud: when every rerank batch fails, raise instead of silently degrading.

    Silent fallback to ``results[:top_n]`` would degenerate to dict-insertion order
    on the cluster path (acluster_retrieve returns ``score=0.0`` candidates), masking
    real reranker outages.
    """
    rerank_client = MagicMock()
    rerank_client.rerank = AsyncMock(side_effect=RuntimeError("API down"))

    docs = [
        ({"id": "0", "episode": {"subject": "x", "content": "x body"}}, 0.5),
        ({"id": "1", "episode": {"subject": "y", "content": "y body"}}, 0.4),
    ]
    with pytest.raises(RuntimeError, match="reranker batch success rate"):
        await reranker_search(
            "q",
            results=docs,
            rerank_client=rerank_client,
            top_n=2,
            batch_size=10,
            concurrent_batches=1,
            max_retries=1,
            retry_delay=0.001,
        )


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
