"""Benchmark configuration.

Defaults are deliberately aligned with the upstream evaluation reference's
``the upstream reference config`` so that benchmark numbers
are directly comparable to the upstream reference's published LoCoMo results.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class BenchmarkConfig(BaseModel):
    """Immutable benchmark configuration."""

    model_config = ConfigDict(frozen=True)

    # === Retrieval ===
    retrieval_mode: Literal["agentic", "lightweight"] = "agentic"
    use_hybrid_search: bool = True
    use_reranker: bool = True
    use_multi_query: bool = True

    # === Top-N parameters (aligned with the upstream reference eval) ===
    emb_recall_top_n: int = 40
    reranker_top_n: int = 20
    hybrid_emb_candidates: int = 50
    hybrid_bm25_candidates: int = 50

    # Two independent fields in the upstream evaluation framework both default to 40: one for Level-1 Hybrid
    # RRF, one for Round-2 multi-query RRF. Collapsed into one knob here while the two values stay equal;
    # split this if a future release sets them apart.
    hybrid_rrf_k: int = 40
    multi_query_num: int = 3
    response_top_k: int = 10

    # === Stage 3 agentic loop ===
    # Number of docs surviving round-1 rerank — also the LLM sufficiency-check window. 10 matches the
    # cluster-path ``round1_top_k = config.response_top_k = 10``. The cluster-path main loop lives in
    # ``scene_retrieval.py``; ``agentic_utils.py`` is a helper-functions module.
    round1_rerank_top_n: int = 10

    # === Reranker ===
    reranker_batch_size: int = 32
    reranker_concurrent_batches: int = 2
    reranker_max_retries: int = 3
    reranker_retry_delay: float = 3.0
    reranker_timeout: float = 60.0
    reranker_fallback_threshold: float = 0.3
    reranker_instruction: str = (
        "Determine if the passage contains specific facts, entities (names, dates, locations), "
        "or details that directly answer the question."
    )

    # === LLM ===
    llm_model: str = "openai/gpt-4.1-mini"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    # 0.3 matches the upstream reference's answer LLM temperature (the upstream reference config:85).
    llm_temperature: float = 0.3
    # CoT answers normally use ~2-4k tokens; the cap removes truncation risk without inflating cost.
    llm_max_tokens: int = 16384
    llm_timeout: float = 60.0
    # Applies to HTTP-level retries in LLMClient and JSON-parse retries in stage 1.
    llm_max_retries: int = 5

    # === Stage 1 (extract) ===
    # Incremental (per-message) boundary detection: front-2-buffer + smart_mask cut-and-bridge
    # (``stage1_memcells_extraction.py:221-360``). Force-split (``hard_token_limit=8192`` /
    # ``hard_message_limit=50``) is owned by ``BoundaryDetector.adetect_step`` and not exposed here.
    extract_smart_mask: bool = True

    # === Clustering (Stage 1 → Stage 2 → Stage 3 cluster path) ===
    # When enabled, stage 1 assigns each memcell to a cluster via
    # ``everalgo.clustering.cluster_by_geometry`` (cosine + time-window). Stage 2
    # turns the cluster state into a cluster index, and stage 3 uses it for 2-level
    # retrieval.
    enable_clustering: bool = True
    cluster_similarity_threshold: float = 0.70
    cluster_max_time_gap_days: float = 7.0

    # === Stage 3 cluster path ===
    # When ``enable_cluster_retrieval`` and a cluster index exists for the conv,
    # stage 3 takes the 2-level path (Level 1 selects top-K clusters via first-hit
    # scan over RRF results, Level 2 reranks inside clusters). Falls back to flat
    # hybrid when the cluster index is missing.
    enable_cluster_retrieval: bool = True
    cluster_top_k: int = 10
    # Level-1 candidate-pool size for cluster selection. ``None`` means no truncation,
    # matching the upstream behavior where the full RRF result set is scanned for
    # cluster selection without a cap (``scene_retrieval.py:164-174``).
    cluster_base_candidates: int | None = None

    # === Stage 2 (index) ===
    # Embedding batching: flatten all searchable units in a conversation, then
    # ship them in ``embedding_batch_size``-sized batches with at most
    # ``embedding_concurrent_batches`` requests in flight.
    # Default BATCH_SIZE=256, MAX_CONCURRENT_BATCHES=5.
    embedding_batch_size: int = 256
    embedding_concurrent_batches: int = 5

    # === Judge ===
    judge_model: str = "openai/gpt-4o-mini"
    # 0.0 matches the upstream reference's judge temperature (stage5_eval.py / llm_judge.py) for deterministic scoring.
    judge_temperature: float = 0.0
    judge_runs: int = 3

    # === Embedding / Reranker model IDs ===
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_base_url: str = "https://api.deepinfra.com/v1/openai"
    # Qwen3-Embedding-4B is a Matryoshka model — DeepInfra returns 2560-dim by default but truncates
    # server-side when ``dimensions`` is passed. 1024 matches the upstream evaluation framework's
    # ``HybridVectorizeConfig.dimensions`` to keep cosine-sim values and RRF ranking comparable.
    embedding_dimensions: int = 1024
    reranker_model: str = "Qwen/Qwen3-Reranker-4B"
    deepinfra_base_url: str = "https://api.deepinfra.com/v1/inference"

    # === Concurrency ===
    # Stage 1 (extract) + Stage 2 (index) — conversation-level concurrency. LoCoMo only
    # has 10 convs so this rarely throttles in practice; the bound mainly protects API
    # rate-limits when scaling to larger datasets.
    max_concurrent_convs: int = 10
    # Stage 3 (search) + Stage 4 (answer) + Stage 5 (evaluate) — QA-level concurrency.
    # Conservative 30. Reranker has its own
    # ``reranker_concurrent_batches`` and is not affected.
    max_concurrent_qa: int = 30
