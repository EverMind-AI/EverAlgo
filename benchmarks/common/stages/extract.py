"""Stage 1 — MemCell extraction.

For each conversation: BoundaryDetector segments messages into MemCells; for
each MemCell run EpisodeExtractor + AtomicFactExtractor. Output is one
``memcells_conv_<i>.json`` per conversation, in a shape compatible with
EverCore's evaluation pipeline.

After the extraction pass completes, an optional clustering pass embeds each
MemCell's episode body and incrementally assigns it to a geometric cluster via
``everalgo.clustering.cluster_by_geometry``. Output is one
``clusters_conv_<i>.json`` per conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import SecretStr

from benchmarks.common.metrics import estimate_tokens
from benchmarks.common.stages.types import StageStats
from everalgo.clustering.algorithm import cluster_by_geometry
from everalgo.clustering.state import Cluster
from everalgo.llm.config import LLMConfig
from everalgo.llm.providers.openai_compat import OpenAICompatClient
from everalgo.types import ChatMessage as EverAlgoMemMessage
from everalgo.user_memory import BoundaryDetector
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.episode import EpisodeExtractor

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from benchmarks.common.dataset import Conversation
    from benchmarks.common.services import EmbeddingClient
    from benchmarks.common.stages.types import StageContext

logger = logging.getLogger(__name__)


# Retry policy for transient LLM JSON-parse failures. OpenRouter occasionally returns
# prose-wrapped or truncated output that `parse_llm_json_object` cannot recover. The
# retries here are scoped to that exact ValueError so unrelated bugs surface immediately.
_JSON_PARSE_ERROR_SIGNATURE = "Failed to parse LLM response as a JSON object"
_BACKOFF_BASE_SECONDS = 0.5


async def _retry_on_json_parse_failure[T](
    factory: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
) -> T:
    """Invoke ``factory()`` up to ``max_attempts`` times when JSON parsing fails."""
    last_exc: ValueError | None = None
    for attempt in range(max_attempts):
        try:
            return await factory()
        except ValueError as exc:
            if _JSON_PARSE_ERROR_SIGNATURE not in str(exc):
                raise
            last_exc = exc
            if attempt < max_attempts - 1:
                await asyncio.sleep(_BACKOFF_BASE_SECONDS * (2**attempt))
    assert last_exc is not None  # loop above always assigns before exhausting attempts
    raise last_exc


def _build_llm(ctx: StageContext) -> OpenAICompatClient:
    """Construct the OpenAI-compatible LLM client from stage config."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    cfg = ctx.config
    return OpenAICompatClient(
        LLMConfig(
            api_key=SecretStr(api_key),
            base_url=cfg.llm_base_url,
            model=cfg.llm_model,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            timeout=cfg.llm_timeout,
        )
    )


def _conv_to_mem_messages(conv: Conversation) -> list[EverAlgoMemMessage]:
    """Convert benchmark Message list to everalgo ChatMessage list.

    Skips messages with roles other than ``user`` / ``assistant`` because
    ``everalgo.types.ChatMessage.role`` is typed as ``Literal["user",
    "assistant"]`` and rejects ``"system"`` messages.
    """
    result: list[EverAlgoMemMessage] = []
    for m in conv.messages:
        if m.role not in ("user", "assistant"):
            continue
        result.append(
            EverAlgoMemMessage(
                id=m.id,
                role=m.role,  # type: ignore[arg-type]  # narrowed by guard above
                content=m.content,
                timestamp=m.timestamp,
                sender_id=m.sender_id,
                sender_name=m.sender_name,
            )
        )
    return result


def _serialize_memcell(
    mc_idx: int,
    mc: Any,
    episode: Any,
    fact_strings: list[str],
) -> dict[str, Any]:
    """Build the EverAlgo-native dict for a single MemCell.

    Schema is a clean superset of EverAlgo's MemCell / Episode types plus a flat
    atomic-fact list. ``fact_strings`` are the sentences returned by
    ``AtomicFactExtractor.aextract_from_text`` (the from-episode path); each is
    wrapped as ``{"fact": <sentence>}`` so downstream stages see the same shape
    as the legacy ``aextract`` output.
    """
    # Episode.episode holds the narrative body (not .content — see memories.py).
    episode_body: str = getattr(episode, "episode", "") or ""
    episode_subject: str = getattr(episode, "subject", "") or ""
    # Mirror EverCore: persist summary alongside subject/content so BM25 / embedding
    # fallback paths (when atomic_facts is empty) have the same three-field input as
    # EverCore's stage1 output. EverAlgo's EpisodeExtractor already populates summary
    # (LLM output or content[:200] fallback) via pydantic extra='allow'.
    episode_summary: str = getattr(episode, "summary", "") or (episode_body[:200] if episode_body else "")

    return {
        "id": str(mc_idx),
        "timestamp": mc.timestamp,
        "items": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "sender_id": m.sender_id,
                "sender_name": m.sender_name,
            }
            for m in mc.items
        ],
        "episode": {
            "subject": episode_subject,
            "summary": episode_summary,
            "content": episode_body,
        },
        "atomic_facts": [{"fact": s} for s in fact_strings],
    }


async def _extract_memcell_data(
    mc: Any,
    mc_idx: int,
    conv: Conversation,
    llm: OpenAICompatClient,
    *,
    max_attempts: int,
) -> tuple[dict[str, Any], int, int]:
    """Extract episode + atomic facts for one MemCell and serialize to dict.

    Pipeline mirrors the locomo-benchmark branch's two-hop semantic: first run
    ``EpisodeExtractor`` to compress the raw MemCell into a narrative episode,
    then feed the episode body (plus the MemCell's closing timestamp) into
    ``AtomicFactExtractor.aextract_from_text`` — equivalent to the branch's
    ``EventLogExtractor.extract_event_log(episode_text, timestamp)``. If the
    Episode extractor returns an empty body, the from-text call is skipped and
    ``atomic_facts`` is persisted empty so downstream BM25 / embedding paths
    still index the MemCell via its episode subject and original messages.

    ``EpisodeExtractor`` receives ``sender_id=None`` to mirror EverCore main's
    ``stage1_memcells_extraction.py:193-194`` (generic third-person episode,
    not user-centred).

    Returns:
        Tuple of (serialized memcell dict, estimated prompt tokens, estimated
        completion tokens). Token counts are tiktoken approximations covering
        the Episode pass plus the atomic-fact-from-text pass.
    """
    episode = await _retry_on_json_parse_failure(
        lambda: EpisodeExtractor(llm=llm).aextract(mc, sender_id=None),
        max_attempts=max_attempts,
    )
    episode_body: str = getattr(episode, "episode", "") or ""
    if episode_body:
        fact_strings = await _retry_on_json_parse_failure(
            lambda: AtomicFactExtractor(llm=llm).aextract_from_text(episode_body, timestamp=mc.timestamp),
            max_attempts=max_attempts,
        )
    else:
        fact_strings = []
    serialized = _serialize_memcell(mc_idx, mc, episode, fact_strings)

    # Tiktoken-based approximation: input messages x 2 extractor calls
    input_tokens = sum(estimate_tokens(m["content"]) for m in serialized["items"])
    estimated_prompt = input_tokens * 2  # BoundaryDetector + EpisodeExtractor + AtomicFactExtractor ≈ 2 passes

    episode_dict: dict[str, Any] = serialized.get("episode") or {}
    output_tokens = estimate_tokens(str(episode_dict.get("content", ""))) + estimate_tokens(
        str(episode_dict.get("subject", ""))
    )
    output_tokens += sum(estimate_tokens(f["fact"]) for f in serialized.get("atomic_facts", []))

    return serialized, estimated_prompt, output_tokens


async def _detect_all_boundaries(
    mem_messages: list[EverAlgoMemMessage],
    llm: OpenAICompatClient,
    *,
    batch_size: int,
    max_attempts: int,
) -> list[Any]:
    """Stream messages through BoundaryDetector in ``batch_size`` chunks.

    Each call receives ``previous_tail + new_batch`` and returns ``(cells, tail)``.
    The tail (messages the LLM could not yet close) carries forward; the final
    batch passes ``is_final=True`` to force-flush any residual tail into a cell.
    Smaller batches yield finer slicing at higher LLM-call count — the knob is
    surfaced as ``BenchmarkConfig.extract_boundary_batch_size``.
    """
    detector = BoundaryDetector(llm=llm)
    cells: list[Any] = []
    tail: list[EverAlgoMemMessage] = []
    total = len(mem_messages)
    for batch_start in range(0, total, batch_size):
        new_batch = mem_messages[batch_start : batch_start + batch_size]
        batch_input = tail + new_batch
        is_final = batch_start + batch_size >= total

        async def _detect_batch(inp: list[EverAlgoMemMessage] = batch_input, *, fin: bool = is_final) -> Any:
            return await detector.adetect(inp, is_final=fin)

        result = await _retry_on_json_parse_failure(
            _detect_batch,
            max_attempts=max_attempts,
        )
        cells.extend(result.cells)
        tail = list(result.tail)
    assert not tail, "is_final=True on last batch must flush remaining tail"
    return cells


async def _extract_one_conversation(
    conv: Conversation,
    llm: OpenAICompatClient,
    *,
    semaphore: asyncio.Semaphore,
    boundary_batch_size: int,
    max_attempts: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Run boundary detection + episode + atomic-fact extraction on one conversation.

    BoundaryDetector streams messages in batches of ``boundary_batch_size`` with
    tail-carry between calls (mirror EverCore main). All MemCell extractors then
    run in parallel via asyncio.gather, bounded by the shared semaphore.

    Returns:
        Tuple of (memcell list, total estimated prompt tokens, total estimated
        completion tokens) for this conversation.
    """
    mem_messages = _conv_to_mem_messages(conv)
    boundary_cells = await _detect_all_boundaries(
        mem_messages, llm, batch_size=boundary_batch_size, max_attempts=max_attempts
    )

    async def _gated(mc: Any, idx: int) -> tuple[dict[str, Any], int, int]:
        async with semaphore:
            return await _extract_memcell_data(mc, idx, conv, llm, max_attempts=max_attempts)

    results = list(await asyncio.gather(*(_gated(mc, idx) for idx, mc in enumerate(boundary_cells))))
    memcells = [r[0] for r in results]
    total_prompt = sum(r[1] for r in results)
    total_completion = sum(r[2] for r in results)
    return memcells, total_prompt, total_completion


async def _cluster_one_memcell(
    mc: dict[str, Any],
    existing_clusters: list[Cluster],
    embedding_client: EmbeddingClient,
    *,
    threshold: float,
    time_window_days: float,
) -> list[Cluster]:
    """Embed one MemCell's episode body and assign it to a cluster in-place.

    Returns the updated ``existing_clusters`` list (input list is replaced, not
    mutated, because Cluster is frozen). The caller owns the list and passes it
    across iterations to accumulate state.

    Skips clustering with a warning when ``episode.content`` is empty — an empty
    embedding is meaningless and would pollute the centroid of any nearby cluster.
    """
    episode_body: str = mc.get("episode", {}).get("content", "") or ""
    if not episode_body:
        logger.warning("memcell id=%s has empty episode body; skipping cluster assignment", mc.get("id"))
        return existing_clusters

    vectors = await embedding_client.embed([episode_body])
    vec = np.asarray(vectors[0], dtype=np.float32)

    new_cluster = Cluster(
        centroid=vec,
        last_ts=int(mc["timestamp"]),
        members=[str(mc["id"])],
        preview=[episode_body[:200]],
    )
    merged = await cluster_by_geometry(
        new_cluster,
        existing_clusters,
        threshold=threshold,
        time_window_days=time_window_days,
    )
    if merged is None:
        new_id = f"scene_{len(existing_clusters)}"
        existing_clusters = [*existing_clusters, new_cluster.model_copy(update={"id": new_id})]
    else:
        existing_clusters = [merged if c.id == merged.id else c for c in existing_clusters]
    return existing_clusters


def _serialize_cluster_file(clusters: list[Cluster]) -> dict[str, Any]:
    """Build the JSON-serialisable dict for ``clusters_conv_<i>.json``.

    Centroid is serialised as a plain Python float list (``tolist()``) so
    ``json.dumps`` does not choke on ``np.float32`` values.
    """
    memcell_to_cluster: dict[str, str] = {
        member_id: cluster.id  # type: ignore[misc]  # id is str after minting
        for cluster in clusters
        for member_id in cluster.members
    }
    return {
        "clusters": [
            {
                "id": cluster.id,
                "centroid": cluster.centroid.tolist(),
                "count": cluster.count,
                "last_ts": cluster.last_ts,
                "members": cluster.members,
                "preview": cluster.preview,
            }
            for cluster in clusters
        ],
        "memcell_to_cluster": memcell_to_cluster,
    }


async def _run_clustering_pass(
    conv_idx: int,
    memcells: list[dict[str, Any]],
    output_dir: Path,
    embedding_client: EmbeddingClient,
    *,
    threshold: float,
    time_window_days: float,
) -> None:
    """Embed each MemCell and assign to a geometric cluster; persist result.

    Runs sequentially (one embed call per MemCell) — correctness over throughput,
    since each assignment depends on the accumulated cluster state from the
    previous step.

    On any failure the error is written to ``clusters_conv_<i>.error.txt`` and
    re-raised so ``_process_conversation`` can mark the conversation as failed.
    """
    clusters: list[Cluster] = []
    for mc in memcells:
        clusters = await _cluster_one_memcell(
            mc,
            clusters,
            embedding_client,
            threshold=threshold,
            time_window_days=time_window_days,
        )

    cluster_data = _serialize_cluster_file(clusters)
    out = output_dir / f"clusters_conv_{conv_idx}.json"
    out.write_text(json.dumps(cluster_data, ensure_ascii=False, indent=2))


async def _process_conversation(
    idx: int,
    conv: Conversation,
    llm: OpenAICompatClient,
    conv_sem: asyncio.Semaphore,
    mc_sem: asyncio.Semaphore,
    output_dir: Any,
    *,
    boundary_batch_size: int,
    max_attempts: int,
    ctx: StageContext,
) -> tuple[bool, int, int]:
    """Process one conversation under the conv semaphore.

    The ``mc_sem`` is shared across all conversations to bound total concurrent
    MemCell LLM calls, matching EverCore's Semaphore(20) design.

    Returns:
        Tuple of (success, estimated_prompt_tokens, estimated_completion_tokens).
    """
    async with conv_sem:
        try:
            payload, prompt_tokens, completion_tokens = await _extract_one_conversation(
                conv,
                llm,
                semaphore=mc_sem,
                boundary_batch_size=boundary_batch_size,
                max_attempts=max_attempts,
            )
        except Exception:  # per-conv isolation; errors written to .error.txt
            err_path = output_dir / f"memcells_conv_{idx}.error.txt"
            err_path.write_text(traceback.format_exc())
            return False, 0, 0

        out = output_dir / f"memcells_conv_{idx}.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

        if ctx.config.enable_clustering:
            try:
                await _run_clustering_pass(
                    idx,
                    payload,
                    output_dir,
                    ctx.services.embedding,
                    threshold=ctx.config.cluster_similarity_threshold,
                    time_window_days=ctx.config.cluster_max_time_gap_days,
                )
            except Exception:  # per-conv isolation; extract output already written
                err_path = output_dir / f"clusters_conv_{idx}.error.txt"
                err_path.write_text(traceback.format_exc())
                return False, prompt_tokens, completion_tokens

        return True, prompt_tokens, completion_tokens


async def run_extract_stage(ctx: StageContext) -> StageStats:
    """Stage 1 — extract all conversations to ``memcells_conv_*.json``.

    Reads conversations from ``ctx.dataset``, runs BoundaryDetector +
    EpisodeExtractor + AtomicFactExtractor on each, and writes one JSON file
    per conversation under ``ctx.output_dir``.  Errors are isolated per
    conversation and written to ``*.error.txt`` files.

    Parallelism:
    - Conversations are processed concurrently (bounded by ``conv_sem``).
    - Within each conversation, MemCell extraction tasks (Episode + AtomicFact)
      are gathered concurrently and bounded by a global ``mc_sem`` (limit 20),
      matching EverCore's Semaphore(20) model.
    """
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage_name="extract")
    started = time.monotonic()

    llm = _build_llm(ctx)

    conversations = list(ctx.dataset.load_conversations())
    if ctx.smoke:
        conversations = conversations[: ctx.smoke_conv_limit]

    # conv_sem: bound simultaneous conversations; mc_sem: bound total MemCell LLM calls.
    conv_sem = asyncio.Semaphore(ctx.config.max_concurrent_qa)
    mc_sem = asyncio.Semaphore(20)  # mirrors EverCore's Semaphore(20)

    from tqdm.asyncio import tqdm as async_tqdm  # type: ignore[import-untyped]

    boundary_batch_size = ctx.config.extract_boundary_batch_size
    max_attempts = ctx.config.llm_max_retries

    results: list[tuple[bool, int, int]] = await async_tqdm.gather(  # type: ignore[attr-defined]
        *[
            _process_conversation(
                i,
                conv,
                llm,
                conv_sem,
                mc_sem,
                ctx.output_dir,
                boundary_batch_size=boundary_batch_size,
                max_attempts=max_attempts,
                ctx=ctx,
            )
            for i, conv in enumerate(conversations)
        ],
        desc="extract",
        unit="conv",
        dynamic_ncols=True,
    )

    stats.success = sum(1 for ok, _, _ in results if ok)
    stats.failed = sum(1 for ok, _, _ in results if not ok)
    stats.prompt_tokens = sum(p for _, p, _ in results)
    stats.completion_tokens = sum(c for _, _, c in results)
    stats.duration_seconds = time.monotonic() - started
    return stats
