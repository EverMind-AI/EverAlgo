"""Tests for BenchmarkConfig."""

import pytest
from pydantic import ValidationError

from benchmarks.common.config import BenchmarkConfig


def test_default_values_match_locomo_benchmark():
    """Defaults must mirror locomo-benchmark branch's runtime values."""
    c = BenchmarkConfig()
    # Retrieval
    assert c.retrieval_mode == "agentic"
    assert c.use_hybrid_search is True
    assert c.use_reranker is True
    assert c.use_multi_query is True
    # Top-N
    assert c.emb_recall_top_n == 40
    assert c.reranker_top_n == 20
    assert c.hybrid_emb_candidates == 50
    assert c.hybrid_bm25_candidates == 50
    # 60 mirrors retrieval_utils.py hardcoded k=60 (not the dead config value 40)
    assert c.hybrid_rrf_k == 60
    assert c.response_top_k == 10
    # LLM
    assert c.llm_model == "openai/gpt-4.1-mini"
    assert c.llm_temperature == 0.3  # aligns with locomo-benchmark answer LLM temperature
    assert c.judge_model == "openai/gpt-4o-mini"
    assert c.judge_temperature == 0.0
    assert c.judge_runs == 3


def test_is_frozen():
    """Config must be immutable after construction."""
    c = BenchmarkConfig()
    with pytest.raises(ValidationError):
        c.llm_model = "gpt-3.5"  # type: ignore[misc]
