"""Unit tests for Stage 3 retrieval primitives."""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from benchmarks.common.stages.search import (
    _fuse_with_algo_rrf,
    _tokenize,
    compute_maxsim_score,
    search_with_bm25_index,
    search_with_emb_index,
)


def test_tokenize_basic():
    import nltk  # type: ignore[import-untyped]
    from nltk.corpus import stopwords  # type: ignore[import-untyped]
    from nltk.stem import PorterStemmer  # type: ignore[import-untyped]

    try:
        stopwords.words("english")  # type: ignore[no-untyped-call]
    except LookupError:
        nltk.download("stopwords", quiet=True)  # type: ignore[no-untyped-call]
    try:
        nltk.data.find("tokenizers/punkt")  # type: ignore[no-untyped-call]
    except LookupError:
        nltk.download("punkt", quiet=True)  # type: ignore[no-untyped-call]
    try:
        nltk.data.find("tokenizers/punkt_tab")  # type: ignore[no-untyped-call]
    except LookupError:
        nltk.download("punkt_tab", quiet=True)  # type: ignore[no-untyped-call]

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))  # type: ignore[no-untyped-call]
    toks = _tokenize("The cats are running quickly!", stemmer, stop_words)
    # "the" and "are" are stopwords; "cats" → "cat", "running" → "run", "quickly" → "quickli"
    assert "cat" in toks
    assert "run" in toks
    assert "the" not in toks


def test_compute_maxsim_picks_best_fact():
    query = np.array([1.0, 0.0], dtype=np.float32)
    facts = [
        np.array([0.0, 1.0], dtype=np.float32),  # orthogonal, sim=0
        np.array([1.0, 0.0], dtype=np.float32),  # identical, sim=1
        np.array([0.5, 0.5], dtype=np.float32),
    ]
    score = compute_maxsim_score(query, facts)
    assert abs(score - 1.0) < 1e-5


def test_compute_maxsim_empty_returns_zero():
    query = np.array([1.0], dtype=np.float32)
    assert compute_maxsim_score(query, []) == 0.0


def test_compute_maxsim_zero_query_returns_zero():
    query = np.array([0.0, 0.0], dtype=np.float32)
    facts = [np.array([1.0, 0.0], dtype=np.float32)]
    assert compute_maxsim_score(query, facts) == 0.0


def test_search_with_bm25_fact_level_maxsim():
    """Fact-level scores aggregate to doc level via MAX across the doc's facts.

    BM25's idf is corpus-dependent and can be tiny or negative on toy inputs, so we
    stub the BM25 object with a deterministic ``get_scores`` and verify that the
    MaxSim aggregation step (``max`` per parent doc) wires together correctly.
    """

    class _StubBM25:
        def get_scores(self, _tokens: list[str]) -> list[float]:
            # 4 fact-rows, fact_to_doc_idx below maps them to 3 docs.
            return [10.0, 5.0, 8.0, 3.0]

    docs = [{"id": "0"}, {"id": "1"}, {"id": "2"}]
    bm25_index = {
        "bm25": _StubBM25(),
        "docs": docs,
        "fact_to_doc_idx": [0, 0, 1, 2],
        "index_type": "maxsim",
    }
    out = search_with_bm25_index("alice fish", bm25_index, top_n=3)
    assert [d["id"] for d, _ in out] == ["0", "1", "2"]
    # doc 0 takes max(10, 5) = 10; doc 1 takes 8; doc 2 takes 3.
    assert out[0][1] == 10.0
    assert out[1][1] == 8.0
    assert out[2][1] == 3.0


def test_search_with_bm25_empty_query_returns_empty():
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

    bm25_index = {
        "bm25": BM25Okapi([["alice"]]),
        "docs": [{"id": "0"}],
        "fact_to_doc_idx": [0],
        "index_type": "maxsim",
    }
    # Stopwords-only query → tokenizer returns nothing → empty result, no exception.
    out = search_with_bm25_index("the and a", bm25_index, top_n=5)
    assert out == []


@pytest.mark.asyncio
async def test_search_with_emb_index_maxsim_strategy():
    """Picks doc with highest MaxSim over atomic_facts."""
    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(return_value=[[1.0, 0.0]])

    emb_index = [
        {
            "doc": {"id": "0"},
            "embeddings": {
                "atomic_facts": [
                    np.array([0.0, 1.0], dtype=np.float32),  # orthogonal, sim=0
                    np.array([0.5, 0.5], dtype=np.float32),  # partial match, sim≈0.707
                ]
            },
        },
        {
            "doc": {"id": "1"},
            "embeddings": {
                "atomic_facts": [
                    np.array([1.0, 0.0], dtype=np.float32),  # perfect match
                ]
            },
        },
    ]
    results = await search_with_emb_index("query", emb_index, top_n=2, embedding_client=embedding_client)
    assert results[0][0]["id"] == "1"  # higher score
    assert results[1][0]["id"] == "0"


@pytest.mark.asyncio
async def test_search_with_emb_index_uses_subject_alongside_facts():
    """No short-circuit: subject embedding participates in MaxSim even when atomic_facts exist."""
    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(return_value=[[1.0, 0.0]])

    emb_index = [
        {
            "doc": {"id": "0"},
            "embeddings": {
                "atomic_facts": [
                    np.array([0.0, 1.0], dtype=np.float32),  # orthogonal, sim=0
                ],
                # subject perfectly matches the query — must win over the orthogonal fact.
                "subject": np.array([1.0, 0.0], dtype=np.float32),
            },
        }
    ]
    results = await search_with_emb_index("q", emb_index, top_n=1, embedding_client=embedding_client)
    assert len(results) == 1
    assert results[0][1] > 0.9


@pytest.mark.asyncio
async def test_search_with_emb_index_falls_back_to_field_embeddings():
    """When no atomic_facts, scores against subject/summary/content fields."""
    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(return_value=[[1.0, 0.0]])

    emb_index = [
        {
            "doc": {"id": "0"},
            "embeddings": {
                "subject": np.array([1.0, 0.0], dtype=np.float32),
                "content": np.array([0.0, 1.0], dtype=np.float32),
            },
        }
    ]
    results = await search_with_emb_index("q", emb_index, top_n=1, embedding_client=embedding_client)
    assert len(results) == 1
    assert results[0][1] > 0.9  # high sim via subject


def test_fuse_with_algo_rrf_combines_lists():
    """Algo-rrf adapter combines emb + bm25 ranked lists and preserves doc refs."""
    doc_a = {"id": "0"}
    doc_b = {"id": "1"}
    doc_c = {"id": "2"}
    emb_results = [(doc_a, 0.9), (doc_b, 0.7)]
    bm25_results = [(doc_b, 15.0), (doc_c, 10.0)]
    fused = _fuse_with_algo_rrf([emb_results, bm25_results], k=60)
    # doc_b appears in both → highest RRF.
    assert fused[0][0]["id"] == "1"
    # All 3 documents appear in the fused list.
    assert {d["id"] for d, _ in fused} == {"0", "1", "2"}


def test_fuse_with_algo_rrf_multiple_sources():
    """N-source fusion: doc ranked highly across many sources accumulates the highest score."""
    doc_a = {"id": "0"}
    doc_b = {"id": "1"}
    doc_c = {"id": "2"}
    sources = [
        [(doc_a, 0.9), (doc_b, 0.7)],
        [(doc_b, 0.88), (doc_c, 0.5)],
        [(doc_a, 0.92), (doc_b, 0.6)],
    ]
    fused = _fuse_with_algo_rrf(sources, k=60)
    ids = [d["id"] for d, _ in fused]
    # b appears 3x, a appears 2x, c appears 1x → b should be top
    assert ids[0] == "1"


def test_fuse_with_algo_rrf_empty_inputs():
    """All-empty sources fuse to an empty list."""
    assert _fuse_with_algo_rrf([[], []], k=60) == []
