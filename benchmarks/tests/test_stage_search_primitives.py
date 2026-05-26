"""Unit tests for Stage 3 retrieval primitives."""

from unittest.mock import AsyncMock

import numpy as np
import pytest

from benchmarks.common.stages.search import (
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
async def test_search_with_emb_index_atomic_facts_short_circuit():
    """atomic_facts non-empty → field embeddings are NOT included (short-circuit path C).

    When ``atomic_facts`` is present and non-empty, ``_score_emb_item`` returns
    MaxSim over those facts only — ``subject`` / ``summary`` / ``episode`` are excluded.
    Score = max cosine over [orthogonal fact] = 0.0 (not boosted by the perfect-match subject).
    """
    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(return_value=[[1.0, 0.0]])

    emb_index = [
        {
            "doc": {"id": "0"},
            "embeddings": {
                "atomic_facts": [
                    np.array([0.0, 1.0], dtype=np.float32),  # orthogonal, sim=0
                ],
                # subject perfectly matches the query, but is excluded by the short-circuit.
                "subject": np.array([1.0, 0.0], dtype=np.float32),
            },
        }
    ]
    results = await search_with_emb_index("q", emb_index, top_n=1, embedding_client=embedding_client)
    assert len(results) == 1
    # atomic_facts short-circuit: only the orthogonal fact is scored → MaxSim = 0.0
    assert results[0][1] == 0.0


@pytest.mark.asyncio
async def test_search_with_emb_index_falls_back_to_field_embeddings():
    """When no atomic_facts, scores against subject/summary/episode fields."""
    embedding_client = AsyncMock()
    embedding_client.embed = AsyncMock(return_value=[[1.0, 0.0]])

    emb_index = [
        {
            "doc": {"id": "0"},
            "embeddings": {
                "subject": np.array([1.0, 0.0], dtype=np.float32),
                "episode": np.array([0.0, 1.0], dtype=np.float32),
            },
        }
    ]
    results = await search_with_emb_index("q", emb_index, top_n=1, embedding_client=embedding_client)
    assert len(results) == 1
    assert results[0][1] > 0.9  # high sim via subject
