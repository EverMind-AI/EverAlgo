"""Stage 3 — 2-level scene-agentic retrieval path.

Level 1 selects top-K scenes via RRF + MaxSim aggregation.
Level 2 reranks docs inside selected scenes, checks sufficiency, and
optionally expands to a full-corpus Round 2 multi-query when insufficient.
All search primitives are imported from search.py — none are reimplemented here.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from benchmarks.common.stages import _agentic_utils
from benchmarks.common.stages.search import (
    _fuse_with_algo_rrf,
    _trace_scored,
    hybrid_search_with_rrf,
    reranker_search,
    search_with_bm25_index,
    search_with_emb_index,
)

if TYPE_CHECKING:
    from benchmarks.common.services import EmbeddingClient, LLMClient, RerankClient

logger = logging.getLogger(__name__)

_Doc = dict[str, Any]
_Scored = tuple[_Doc, float]


async def _level1_rrf(
    query: str,
    *,
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
    config: Any,
    embedding_client: EmbeddingClient,
) -> tuple[list[_Scored], list[_Scored], list[_Scored]] | None:
    """Run parallel emb+BM25 search and fuse with RRF; return None when both are empty."""
    emb_task = search_with_emb_index(
        query, emb_index, top_n=config.level1_emb_candidates, embedding_client=embedding_client
    )
    bm25_task = asyncio.to_thread(search_with_bm25_index, query, bm25_index, config.level1_bm25_candidates)
    emb_results, bm25_results = await asyncio.gather(emb_task, bm25_task)

    if emb_results and bm25_results:
        return _fuse_with_algo_rrf([emb_results, bm25_results], k=config.level1_rrf_k), emb_results, bm25_results
    if emb_results:
        return emb_results, emb_results, []
    if bm25_results:
        return bm25_results, [], bm25_results
    return None


def _select_scene_docs(
    rrf_results: list[_Scored],
    *,
    scene_index: dict[str, Any],
    scene_top_k: int,
    docs: list[_Doc],
) -> tuple[list[_Doc], int, int, int]:
    """MaxSim-aggregate RRF results to scenes; expand and filter docs to the selected scenes.

    Returns ``(scene_docs, scanned, scene_count, total_memcells)`` so the
    caller can populate metadata without duplicating iteration logic.
    """
    memcell_to_scene: dict[str, str] = scene_index["memcell_to_scene"]
    scene_scores: dict[str, float] = {}
    scanned = 0
    for doc, score in rrf_results:
        mc_id = str(doc.get("id", ""))
        scanned += 1
        scene_id = memcell_to_scene.get(mc_id)
        if scene_id and (scene_id not in scene_scores or score > scene_scores[scene_id]):
            scene_scores[scene_id] = score
        if len(scene_scores) >= scene_top_k:
            break

    if not scene_scores:
        return [], scanned, 0, 0

    sorted_scenes = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)[:scene_top_k]
    scene_dict_map = {s["scene_id"]: s for s in scene_index["scenes"]}
    all_memcell_ids: set[str] = set()
    for scene_id, _ in sorted_scenes:
        scene = scene_dict_map.get(scene_id, {})
        all_memcell_ids.update(scene.get("memcell_ids", []))

    scene_docs = [d for d in docs if str(d.get("id", "")) in all_memcell_ids]
    return scene_docs, scanned, len(sorted_scenes), len(all_memcell_ids)


def _reranker_kwargs(config: Any) -> dict[str, Any]:
    """Collect reranker call kwargs from config — avoids repeating 7 args twice."""
    return {
        "reranker_instruction": config.reranker_instruction,
        "batch_size": config.reranker_batch_size,
        "concurrent_batches": config.reranker_concurrent_batches,
        "max_retries": config.reranker_max_retries,
        "retry_delay": config.reranker_retry_delay,
        "timeout": config.reranker_timeout,
        "fallback_threshold": config.reranker_fallback_threshold,
    }


async def _run_round2(
    query: str,
    *,
    reranked: list[_Scored],
    missing_info: list[str],
    key_info: list[str],
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
    config: Any,
    llm: LLMClient,
    embedding_client: EmbeddingClient,
    rerank_client: RerankClient,
    metadata: dict[str, Any],
) -> list[_Scored]:
    """Run multi-query Round 2 on the FULL corpus and return final reranked results.

    Round 2 deliberately searches the full corpus, not the selected scenes (mirror 93 line 300).
    """
    refined_queries, _query_strategy, mq_tokens = await _agentic_utils.generate_multi_queries(
        original_query=query,
        results=reranked,
        missing_info=missing_info,
        llm=llm,
        judge_model=getattr(config, "judge_model", None),
        key_info=key_info,
        max_docs=config.response_top_k,
        num_queries=config.multi_query_num,
    )
    metadata["prompt_tokens"] += mq_tokens.get("prompt_tokens", 0)
    metadata["completion_tokens"] += mq_tokens.get("completion_tokens", 0)
    metadata["refined_queries"] = refined_queries

    multi_query_tasks = [
        hybrid_search_with_rrf(
            q,
            emb_index=emb_index,
            bm25_index=bm25_index,
            embedding_client=embedding_client,
            top_n=50,
            emb_candidates=config.hybrid_emb_candidates,
            bm25_candidates=config.hybrid_bm25_candidates,
            rrf_k=config.hybrid_rrf_k,
            return_components=False,
        )
        for q in refined_queries
    ]
    multi_query_results: list[list[_Scored]] = await asyncio.gather(*multi_query_tasks)  # type: ignore[assignment]

    metadata["trace"]["r2_subqueries"] = [
        {"query": q, "rrf_top": _trace_scored(r, limit=20)}
        for q, r in zip(refined_queries, multi_query_results, strict=False)
    ]

    if len(multi_query_results) == 1:
        round2_results: list[_Scored] = list(multi_query_results[0])[:40]
    else:
        round2_results = _fuse_with_algo_rrf(multi_query_results, k=config.hybrid_rrf_k)[:40]

    metadata["round2_count"] = len(round2_results)

    r1_ids = {str(d.get("id", "")) for d, _ in reranked}
    r2_unique = [(d, s) for d, s in round2_results if str(d.get("id", "")) not in r1_ids]
    combined: list[_Scored] = list(reranked) + r2_unique[: 40 - len(reranked)]

    return await reranker_search(
        query,
        results=combined,
        rerank_client=rerank_client,
        top_n=config.response_top_k,
        **_reranker_kwargs(config),
    )


async def scene_agentic_retrieval(
    query: str,
    *,
    scene_index: dict[str, Any],
    emb_index: list[dict[str, Any]],
    bm25_index: dict[str, Any],
    config: Any,
    llm: LLMClient,
    embedding_client: EmbeddingClient,
    rerank_client: RerankClient,
) -> tuple[list[_Scored], dict[str, Any]]:
    """2-level scene-agentic retrieval (mirrors 93-branch scene_retrieval.agentic_retrieval).

    Level 1 selects scenes, Level 2 reranks inside scenes + checks sufficiency.
    Round 2 (when insufficient) searches the FULL corpus, not just selected scenes.
    """
    start_time = time.monotonic()

    metadata: dict[str, Any] = {
        "retrieval_mode": "scene_agentic",
        "scene_top_k": config.scene_top_k,
        "response_top_k": config.response_top_k,
        "is_multi_round": False,
        "is_sufficient": None,
        "reasoning": None,
        "level1_scanned": 0,
        "level1_scene_count": 0,
        "total_memcells_in_scenes": 0,
        "round1_count": 0,
        "round1_reranked_count": 0,
        "round2_count": 0,
        "final_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_latency_ms": 0.0,
        "trace": {
            "r1_emb_top": [],
            "r1_bm25_top": [],
            "r1_rrf_top": [],
            "r1_rerank_top": [],
            "r2_subqueries": [],
            "final_top": [],
        },
    }

    def _finish() -> tuple[list[_Scored], dict[str, Any]]:
        metadata["total_latency_ms"] = (time.monotonic() - start_time) * 1000
        return [], metadata

    # ---- Level 1: scene selection -------------------------------------------
    level1 = await _level1_rrf(
        query,
        emb_index=emb_index,
        bm25_index=bm25_index,
        config=config,
        embedding_client=embedding_client,
    )
    if level1 is None:
        return _finish()

    rrf_results, emb_results, bm25_results = level1
    metadata["trace"]["r1_emb_top"] = _trace_scored(emb_results, limit=20)
    metadata["trace"]["r1_bm25_top"] = _trace_scored(bm25_results, limit=20)
    metadata["trace"]["r1_rrf_top"] = _trace_scored(rrf_results, limit=20)

    docs: list[_Doc] = bm25_index["docs"]
    scene_docs, scanned, scene_count, total_mc = _select_scene_docs(
        rrf_results, scene_index=scene_index, scene_top_k=config.scene_top_k, docs=docs
    )
    metadata["level1_scanned"] = scanned
    metadata["level1_scene_count"] = scene_count
    metadata["total_memcells_in_scenes"] = total_mc

    if not scene_docs:
        return _finish()

    # ---- Level 2: rerank inside scenes → sufficiency check ------------------
    scene_results_for_rerank: list[_Scored] = [(d, 1.0) for d in scene_docs]
    metadata["round1_count"] = len(scene_results_for_rerank)

    reranked = await reranker_search(
        query,
        results=scene_results_for_rerank,
        rerank_client=rerank_client,
        top_n=config.response_top_k,
        **_reranker_kwargs(config),
    )
    metadata["round1_reranked_count"] = len(reranked)
    metadata["trace"]["r1_rerank_top"] = _trace_scored(reranked, limit=20)

    if not reranked:
        return _finish()

    is_sufficient, reasoning, missing_info, key_info, suff_tokens = await _agentic_utils.check_sufficiency(
        query,
        reranked,
        llm=llm,
        judge_model=getattr(config, "judge_model", None),
        max_docs=config.response_top_k,
    )
    metadata["prompt_tokens"] += suff_tokens.get("prompt_tokens", 0)
    metadata["completion_tokens"] += suff_tokens.get("completion_tokens", 0)
    metadata["is_sufficient"] = is_sufficient
    metadata["reasoning"] = reasoning
    metadata["key_information_found"] = key_info

    if is_sufficient:
        final = reranked[: config.response_top_k]
        metadata["final_count"] = len(final)
        metadata["trace"]["final_top"] = _trace_scored(final, limit=20)
        metadata["total_latency_ms"] = (time.monotonic() - start_time) * 1000
        return final, metadata

    # ---- Round 2: multi-query on FULL corpus (not scene-limited) ------------
    metadata["is_multi_round"] = True
    final = await _run_round2(
        query,
        reranked=reranked,
        missing_info=missing_info,
        key_info=key_info,
        emb_index=emb_index,
        bm25_index=bm25_index,
        config=config,
        llm=llm,
        embedding_client=embedding_client,
        rerank_client=rerank_client,
        metadata=metadata,
    )
    metadata["final_count"] = len(final)
    metadata["trace"]["final_top"] = _trace_scored(final, limit=20)
    metadata["total_latency_ms"] = (time.monotonic() - start_time) * 1000
    return final[: config.response_top_k], metadata
