"""Tests for reranker orchestration (entity-split data model)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmarks.common.stages.search import (
    _format_doc_for_rerank,
    reranker_search,
)


def test_format_doc_returns_episode_text():
    """Flat episode text field is used for reranker input."""
    doc = {"id": "0", "subject": "birthday", "episode": "Alice's birthday party"}
    assert _format_doc_for_rerank(doc) == "Alice's birthday party"


def test_format_doc_raises_on_missing_episode():
    """No episode text → ValueError (fail-loud, no fallback)."""
    with pytest.raises(ValueError, match=r"no episode text"):
        _format_doc_for_rerank({})
    with pytest.raises(ValueError, match=r"no episode text"):
        _format_doc_for_rerank({"id": "0", "subject": "x"})
    with pytest.raises(ValueError, match=r"no episode text"):
        _format_doc_for_rerank({"id": "0", "episode": ""})
    with pytest.raises(ValueError, match=r"no episode text"):
        _format_doc_for_rerank({"id": "0", "episode": "   "})


@pytest.mark.asyncio
async def test_reranker_search_returns_top_n_by_rerank_score():
    """3 docs, reranker scores [0.2, 0.9, 0.5], expect top-2 = [doc1, doc2]."""
    rerank_client = MagicMock()
    rerank_client.rerank = AsyncMock(return_value=[(1, 0.9), (2, 0.5), (0, 0.2)])

    docs = [
        ({"id": "0", "subject": "a", "episode": "doc a body"}, 0.5),
        ({"id": "1", "subject": "b", "episode": "doc b body"}, 0.4),
        ({"id": "2", "subject": "c", "episode": "doc c body"}, 0.3),
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
    """Fail-loud: when every rerank batch fails, raise instead of silently degrading."""
    rerank_client = MagicMock()
    rerank_client.rerank = AsyncMock(side_effect=RuntimeError("API down"))

    docs = [
        ({"id": "0", "subject": "x", "episode": "x body"}, 0.5),
        ({"id": "1", "subject": "y", "episode": "y body"}, 0.4),
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
async def test_reranker_search_raises_on_doc_without_episode_text():
    """Docs with no episode text raise ValueError (fail-loud, no silent filtering)."""
    rerank_client = MagicMock()
    docs: list[tuple[dict[str, Any], float]] = [
        ({"id": "0"}, 0.5),  # no episode text
        ({"id": "1", "subject": "hi", "episode": "hello"}, 0.4),
    ]
    with pytest.raises(ValueError, match=r"no episode text"):
        await reranker_search(
            "q",
            results=docs,
            rerank_client=rerank_client,
            top_n=5,
            batch_size=10,
            concurrent_batches=1,
        )
