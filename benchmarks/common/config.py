"""Benchmark configuration.

Defaults are deliberately aligned with the upstream evaluation reference's
``the upstream reference config`` so that benchmark numbers
are directly comparable to the upstream reference's published LoCoMo results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class BenchmarkConfig(BaseModel):
    """Immutable benchmark configuration.

    Provides all tunable parameters for the 7-stage LoCoMo benchmark pipeline:
    retrieval strategy, top-N thresholds, LLM / embedding / reranker model IDs,
    concurrency limits, and judge settings. Defaults are aligned with the upstream
    evaluation reference for directly comparable results.
    """

    model_config = ConfigDict(frozen=True)

    # === Retrieval ===
    retrieval_mode: Literal["agentic"] = "agentic"

    # === Top-N parameters (aligned with the upstream reference eval) ===
    hybrid_emb_candidates: int = 50
    hybrid_bm25_candidates: int = 50

    # Two independent fields in the upstream evaluation framework both default to 40: one for Level-1 Hybrid
    # RRF, one for Round-2 multi-query RRF. Collapsed into one knob here while the two values stay equal;
    # split this if a future release sets them apart.
    hybrid_rrf_k: int = 40
    multi_query_num: int = 3
    response_top_k: int = 10

    # === Stage 5 agentic loop ===
    # Number of docs surviving round-1 rerank — also the LLM sufficiency-check window. 10 matches the
    # cluster-path ``round1_top_k = config.response_top_k = 10``. The cluster-path main loop lives in
    # ``scene_retrieval.py``; ``agentic_utils.py`` is a helper-functions module.
    round1_rerank_top_n: int = 10

    # === Reranker ===
    reranker_batch_size: int = 32
    reranker_concurrent_batches: int = 2
    reranker_max_retries: int = 3
    reranker_timeout: float = 60.0
    reranker_fallback_threshold: float = 0.3
    reranker_instruction: str = (
        "Determine if the passage contains specific facts, entities (names, dates, locations), "
        "or details that directly answer the question."
    )

    # === LLM ===
    # Stage 1 (extract_base) / Stage 2 (reflect) / Stage 3 (enrich) / Stage 5 (search) model.
    extract_model: str = "openai/gpt-4.1-mini"
    # Stage 6 (answer) model.
    answer_model: str = "openai/gpt-4.1-mini"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    # 0.3 matches the upstream reference's answer LLM temperature (the upstream reference config:85).
    llm_temperature: float = 0.3
    # CoT answers normally use ~2-4k tokens; the cap removes truncation risk without inflating cost.
    llm_max_tokens: int = 16384
    llm_timeout: float = 60.0
    # Applies to HTTP-level retries in LLMClient and JSON-parse retries in stage 1.
    llm_max_retries: int = 5

    # === Stage 1 (extract_base) ===
    # Incremental (per-message) boundary detection: front-2-buffer + smart_mask cut-and-bridge
    # (``stage1_memcells_extraction.py:221-360``). Force-split (``hard_token_limit=8192`` /
    # ``hard_message_limit=50``) is owned by ``BoundaryDetector.adetect_step`` and not exposed here.
    extract_smart_mask: bool = True

    # Whether to run a LLM reflection pass after initial extraction (Stage 2 — reflect).
    enable_reflection: bool = False

    # === Clustering (Stage 1 extract_base → Stage 2 reflect → Stage 4 index cluster path) ===
    # Clustering is always on; these parameters tune its behaviour.
    # Stage 1 assigns each episode to a cluster via ``everalgo.clustering.cluster_by_geometry``
    # (cosine + time-window). Stage 4 turns the cluster state into a cluster index, and Stage 5
    # uses it for 2-level retrieval.
    cluster_similarity_threshold: float = 0.70
    cluster_max_time_gap_days: float = 7.0

    # === Stage 5 cluster path (always on in agentic mode) ===
    cluster_top_k: int = 10

    # === Stage 4 (index) ===
    # Embedding batching: flatten all searchable units in a conversation, then
    # ship them in ``embedding_batch_size``-sized batches with at most
    # ``embedding_concurrent_batches`` requests in flight.
    # Default BATCH_SIZE=256, MAX_CONCURRENT_BATCHES=5.
    embedding_batch_size: int = 256
    embedding_concurrent_batches: int = 5

    # === Judge ===
    judge_model: str = "openai/gpt-4o-mini"
    # 0.0 matches the upstream reference's judge temperature (llm_judge.py) for deterministic scoring.
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
    # Stages 1-4 (extract_base / reflect / enrich / index) — conversation-level concurrency. LoCoMo only
    # has 10 convs so this rarely throttles in practice; the bound mainly protects API
    # rate-limits when scaling to larger datasets.
    max_concurrent_convs: int = 10
    # Stage 5 (search) + Stage 6 (answer) + Stage 7 (evaluate) — QA-level concurrency.
    # Conservative 30. Reranker has its own
    # ``reranker_concurrent_batches`` and is not affected.
    max_concurrent_qa: int = 30

    # === Session filter ===
    # Maps conversation index (int) to a list of session indices (int) to run. ``None`` means run all.
    # TOML representation uses string keys ("5" = [...]) which are coerced to int by ``from_toml``.
    session_filter: dict[int, list[int]] | None = None

    @classmethod
    def from_toml(cls, name: str = "config", *, config_dir: Path | None = None) -> BenchmarkConfig:
        """Load config from a TOML file under ``benchmarks/``.

        Reads ``config.toml`` when no name is given (the single source of truth for all
        default parameter values). A named config (e.g. ``benchmark_reflection``) overrides
        the defaults with its own values.

        Args:
            name: Config name (without .toml extension). Defaults to ``"config"``.
            config_dir: Directory containing config files. Defaults to ``benchmarks/``.

        Raises:
            FileNotFoundError: When the TOML file does not exist.
        """
        import tomllib

        if config_dir is None:
            config_dir = Path("benchmarks")
        path = config_dir / f"{name}.toml"
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "rb") as f:
            overrides = tomllib.load(f)
        if "session_filter" in overrides:
            overrides["session_filter"] = {int(k): v for k, v in overrides["session_filter"].items()}
        return cls(**overrides)
