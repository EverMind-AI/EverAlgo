"""Benchmark configuration.

Defaults are deliberately aligned with EverCore evaluation framework's
``evaluation/src/adapters/evermemos/config.py`` so that benchmark numbers
are directly comparable to EverCore's published LoCoMo results.
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

    # === Top-N parameters (aligned with EverCore eval) ===
    emb_recall_top_n: int = 40
    reranker_top_n: int = 20
    hybrid_emb_candidates: int = 50
    hybrid_bm25_candidates: int = 50
    # 60 mirrors locomo-benchmark's *runtime* RRF constant: ``retrieval_utils.py:243``
    # and ``multi_rrf_fusion`` default both hardcode k=60. The branch's
    # ``evermemos/config.py:hybrid_rrf_k=40`` is a dead value that ``lightweight_retrieval``
    # never consults — do not align to it.
    hybrid_rrf_k: int = 60
    multi_query_num: int = 3
    response_top_k: int = 10

    # === Stage 3 agentic loop ===
    # Number of docs surviving round-1 rerank — also the LLM sufficiency-check
    # window. Mirror locomo-benchmark ``agentic_utils.py:91`` (``round1_rerank_top_n=5``).
    round1_rerank_top_n: int = 5

    # === Reranker ===
    # 32 + 2 mirror locomo-benchmark ``evermemos/config.py:66, 71``.
    reranker_batch_size: int = 32
    reranker_concurrent_batches: int = 2
    reranker_max_retries: int = 3
    reranker_retry_delay: float = 0.8
    reranker_timeout: float = 60.0
    reranker_fallback_threshold: float = 0.3
    reranker_instruction: str = (
        "Determine if the passage contains specific facts, entities (names, dates, locations), "
        "or details that directly answer the question."
    )

    # === LLM ===
    llm_model: str = "openai/gpt-4.1-mini"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    # 0.3 matches EverCore's answer LLM temperature (evaluation/src/adapters/evermemos/config.py:85).
    llm_temperature: float = 0.3
    # 16384 mirrors locomo-benchmark branch ``evermemos/config.py:87``. CoT answers
    # normally use ~2-4k tokens; the cap removes truncation risk without inflating cost.
    llm_max_tokens: int = 16384
    llm_timeout: float = 60.0
    # 5 mirrors locomo-benchmark branch ``evermemos/config.py:99`` (``max_retries=5``).
    # Applies to HTTP-level retries in LLMClient and JSON-parse retries in stage 1.
    llm_max_retries: int = 5

    # === Stage 1 (extract) ===
    # BoundaryDetector chunk size. Streamed in ``extract_boundary_batch_size``-msg
    # batches with tail-carry; the LLM sees one batch + previous tail per call.
    # Smaller batches yield finer slicing (fewer mega-cells, finer temporal grain)
    # at higher LLM-call count.
    extract_boundary_batch_size: int = 20

    # === Clustering (Stage 1 → Stage 2 → Stage 3 scene path) ===
    # When enabled, stage 1 assigns each memcell to a cluster via
    # ``everalgo.clustering.cluster_by_geometry`` (cosine + time-window). Stage 2
    # turns the cluster state into a scene index, and stage 3 uses it for 2-level
    # retrieval. Defaults mirror locomo-benchmark ``ExperimentConfig``
    # (``evermemos/config.py:15-22, 55-59``).
    enable_clustering: bool = True
    cluster_similarity_threshold: float = 0.70
    cluster_max_time_gap_days: float = 7.0

    # === Stage 3 scene path ===
    # When ``enable_scene_retrieval`` and a scene index exists for the conv,
    # stage 3 takes the 2-level path (Level 1 selects top-K scenes via RRF +
    # MaxSim, Level 2 reranks inside scenes). Falls back to flat hybrid when the
    # scene index is missing.
    enable_scene_retrieval: bool = True
    scene_top_k: int = 10
    level1_emb_candidates: int = 50
    level1_bm25_candidates: int = 50
    # Level-1 RRF constant is 40 in scene_retrieval.py:152 — distinct from the
    # round-2 multi-query ``hybrid_rrf_k=60`` used elsewhere.
    level1_rrf_k: int = 40

    # === Stage 2 (index) ===
    # Embedding batching: flatten all searchable units in a conversation, then
    # ship them in ``embedding_batch_size``-sized batches with at most
    # ``embedding_concurrent_batches`` requests in flight. Mirror
    # locomo-benchmark ``build_emb_index`` (``stage2_index_building.py:312-313``):
    # BATCH_SIZE=256, MAX_CONCURRENT_BATCHES=5.
    embedding_batch_size: int = 256
    embedding_concurrent_batches: int = 5

    # === Judge ===
    judge_model: str = "openai/gpt-4o-mini"
    # 0.0 matches EverCore's judge temperature (stage5_eval.py / llm_judge.py) for deterministic scoring.
    judge_temperature: float = 0.0
    judge_runs: int = 3

    # === Embedding / Reranker model IDs ===
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    embedding_base_url: str = "https://api.deepinfra.com/v1/openai"
    # Mirror EverCore main ``HybridVectorizeConfig.dimensions=1024``
    # (``methods/EverCore/src/agentic_layer/vectorize_service.py:69``). Qwen3-Embedding-4B is
    # a Matryoshka model — DeepInfra returns 2560-dim by default but truncates server-side when
    # ``dimensions`` is passed. Keeping this aligned ensures cosine-sim values and RRF ranking
    # are byte-comparable to EverCore's baseline.
    embedding_dimensions: int = 1024
    reranker_model: str = "Qwen/Qwen3-Reranker-4B"
    deepinfra_base_url: str = "https://api.deepinfra.com/v1/inference"

    # === Concurrency ===
    # Conservative 30 (main uses 50). Caps the QA-level Semaphore in every stage;
    # reranker has its own ``reranker_concurrent_batches`` and is not affected.
    max_concurrent_qa: int = 30
