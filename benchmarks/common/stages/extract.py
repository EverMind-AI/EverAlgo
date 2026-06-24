"""Stage 1 — Extract Base: boundary detection + episode extraction + clustering.

For each conversation: BoundaryDetector segments messages into MemCells; for
each MemCell run EpisodeExtractor and embed the episode body + subject.
Output is three files per conversation:

- ``memcells_conv_<i>.json`` — pure MemCells (boundary segments only).
- ``episodes_conv_<i>.json`` — Episode entities with embeddings.
- ``clusters_conv_<i>.json`` — Clusters with episode_ids + episode_to_cluster.

AtomicFact extraction is NOT performed here — it moves to the Enrich stage.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
from typing import TYPE_CHECKING, Any, cast

from tqdm import tqdm as _tqdm

from benchmarks.common.metrics import estimate_tokens
from benchmarks.common.retry import llm_retry
from benchmarks.common.services import build_llm_client
from benchmarks.common.stages.serialization import serialize_clusters, serialize_episode, serialize_memcell, write_json
from benchmarks.common.stages.types import StageStats
from everalgo.clustering.algorithm import cluster_by_geometry
from everalgo.clustering.state import Cluster
from everalgo.types import ChatMessage as EverAlgoMemMessage
from everalgo.types import ConversationItem, MemCell
from everalgo.user_memory import BoundaryDetector
from everalgo.user_memory.episode import EpisodeExtractor

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.common.dataset import Conversation
    from benchmarks.common.services import EmbeddingClient
    from benchmarks.common.stages.types import StageContext
    from everalgo.llm.providers.openai_compat import OpenAICompatClient

logger = logging.getLogger(__name__)


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


async def _extract_memcell_data(
    mc: Any,
    mc_idx: int,
    llm: OpenAICompatClient,
    embedding_client: EmbeddingClient,
    *,
    max_attempts: int,
) -> tuple[dict[str, Any], dict[str, Any], int, int]:
    """Extract episode for one MemCell and serialize to separate dicts.

    Pipeline:
    1. EpisodeExtractor — compress raw MemCell into a narrative episode.
       Raises ``ValueError`` if the episode body is empty (fail-loud).
    2. Parallel: embed episode body + embed episode subject.

    Returns:
        Tuple of (memcell_dict, episode_dict, estimated_prompt_tokens,
        estimated_completion_tokens). Token counts are tiktoken approximations.
    """

    @llm_retry(max_attempts=max_attempts)
    async def _do_extract_episode() -> Any:
        return await EpisodeExtractor(llm=llm).aextract(mc, sender_id=None)

    episode = await _do_extract_episode()
    episode_body: str = getattr(episode, "episode", "") or ""
    if not episode_body:
        raise ValueError(f"EpisodeExtractor returned empty episode body (mc_idx={mc_idx})")

    episode_subject: str = getattr(episode, "subject", "") or ""

    async def _do_embed_episode() -> list[float] | None:
        vecs = await embedding_client.embed([episode_body])
        if not vecs:
            raise ValueError(f"embed returned empty vector list for episode body (mc_idx={mc_idx})")
        return list(vecs[0])

    async def _do_embed_subject() -> list[float] | None:
        if not episode_subject:
            return None
        vecs = await embedding_client.embed([episode_subject])
        if not vecs:
            return None
        return list(vecs[0])

    episode_embedding, subject_embedding = await asyncio.gather(_do_embed_episode(), _do_embed_subject())

    memcell_dict = serialize_memcell(mc_idx, mc)
    episode_dict = serialize_episode(
        mc_idx,
        subject=episode_subject,
        episode_text=episode_body,
        memcell_ids=[str(mc_idx)],
        timestamp=mc.timestamp,
        owner_id=None,
        embeddings={"episode": episode_embedding, "subject": subject_embedding},
    )

    estimated_prompt, output_tokens = _estimate_tokens_for_memcell(memcell_dict, episode_body, episode_subject)
    return memcell_dict, episode_dict, estimated_prompt, output_tokens


def _estimate_tokens_for_memcell(
    memcell_dict: dict[str, Any],
    episode_body: str,
    episode_subject: str,
) -> tuple[int, int]:
    """Tiktoken-based token estimation for one MemCell extraction.

    Returns:
        Tuple of ``(estimated_prompt_tokens, output_tokens)``.
    """
    input_tokens = sum(estimate_tokens(m["content"]) for m in memcell_dict["items"])
    estimated_prompt = input_tokens * 2
    output_tokens = estimate_tokens(episode_body) + estimate_tokens(episode_subject)
    return estimated_prompt, output_tokens


async def _detect_all_boundaries(
    mem_messages: list[EverAlgoMemMessage],
    llm: OpenAICompatClient,
    *,
    smart_mask: bool = True,
    max_attempts: int = 5,
    pbar: _tqdm[Any] | None = None,
) -> list[Any]:
    """Incremental boundary detection: front-2-buffer + ``adetect_step`` loop + final-tail flush.

    Args:
        mem_messages: Ordered messages for one conversation.
        llm: LLM client bound to the BoundaryDetector instance.
        smart_mask: Forwarded to ``adetect_step`` (default ``True``).
        max_attempts: Per-step JSON parse retry budget.
        pbar: Optional tqdm bar (caller owns lifecycle).

    Returns:
        Ordered list of MemCells (closed segments + final-tail flush).
    """
    detector = BoundaryDetector(llm=llm)

    @llm_retry(max_attempts=max_attempts)
    async def _step(
        hist: list[EverAlgoMemMessage],
        m: EverAlgoMemMessage,
    ) -> Any:
        return await detector.adetect_step(hist, m, smart_mask=smart_mask)

    cells: list[Any] = []
    history: list[EverAlgoMemMessage] = []

    for msg in mem_messages:
        if pbar is not None:
            pbar.update(1)
        # Front-2 buffer: no LLM until history has >=2 msgs.
        if len(history) < 2:
            history.append(msg)
            continue

        result = await _step(history, msg)
        cells.extend(result.cells)
        history = list(result.tail)

    # Flush remaining history as final MemCell.
    if history:
        cells.append(_make_final_cell(history))

    return cells


def _make_final_cell(slice_msgs: list[EverAlgoMemMessage]) -> Any:
    """Wrap leftover messages as the final MemCell at conversation end.

    Uses ``slice_msgs[-1].timestamp`` as the segment's last-message timestamp.
    """
    return MemCell(
        items=cast("list[ConversationItem]", slice_msgs),
        timestamp=slice_msgs[-1].timestamp,
    )


async def _extract_one_conversation(
    conv: Conversation,
    llm: OpenAICompatClient,
    embedding_client: EmbeddingClient,
    *,
    semaphore: asyncio.Semaphore,
    smart_mask: bool,
    max_attempts: int,
    conv_idx: int = 0,
    pbar: _tqdm[Any] | None = None,
    msg_limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """Run boundary detection + episode extraction on one conversation.

    BoundaryDetector walks messages one-by-one via ``adetect_step`` (incremental
    per-message boundary detection). All MemCell extractors then run in
    parallel via asyncio.gather, bounded by the shared semaphore.

    Args:
        conv: Conversation to process.
        llm: LLM client shared across all extractors.
        embedding_client: Embedding client used to compute episode-body and subject vectors.
        semaphore: Shared semaphore bounding total concurrent MemCell LLM calls.
        smart_mask: Passed to ``_detect_all_boundaries``; mirrors 93 default ``True``.
        max_attempts: JSON-parse retry budget for EpisodeExtractor.
        conv_idx: Conversation index forwarded to pbar description.
        pbar: Optional tqdm bar to update during boundary detection (caller owns lifecycle).
        msg_limit: Truncate mem_messages to this many entries before processing (smoke mode).

    Returns:
        Tuple of (memcell_list, episode_list, total_prompt_tokens, total_completion_tokens).
    """
    mem_messages = _conv_to_mem_messages(conv)
    if msg_limit is not None:
        mem_messages = mem_messages[:msg_limit]
    boundary_cells = await _detect_all_boundaries(
        mem_messages, llm, smart_mask=smart_mask, max_attempts=max_attempts, pbar=pbar
    )

    n_cells = len(boundary_cells)
    if pbar is not None:
        pbar.reset(total=n_cells)
        pbar.set_description(f"conv {conv_idx:02d} | {n_cells} cells")

    async def _gated(mc: Any, idx: int) -> tuple[dict[str, Any], dict[str, Any], int, int]:
        async with semaphore:
            result = await _extract_memcell_data(mc, idx, llm, embedding_client, max_attempts=max_attempts)
            if pbar is not None:
                pbar.update(1)
            return result

    results = list(await asyncio.gather(*(_gated(mc, idx) for idx, mc in enumerate(boundary_cells))))
    memcells = [r[0] for r in results]
    episodes = [r[1] for r in results]
    total_prompt = sum(r[2] for r in results)
    total_completion = sum(r[3] for r in results)
    return memcells, episodes, total_prompt, total_completion


async def _cluster_one_episode(
    episode: dict[str, Any],
    existing_clusters: list[Cluster],
    *,
    threshold: float,
    time_window_days: float,
) -> list[Cluster]:
    """Assign one episode to a geometric cluster using its episode embedding.

    Returns the updated ``existing_clusters`` list (input list is replaced, not
    mutated, because Cluster is frozen). The caller owns the list and passes it
    across iterations to accumulate state.
    """
    embeddings: dict[str, Any] = episode.get("embeddings") or {}
    raw_vec = embeddings.get("episode")
    if raw_vec is None:
        raise ValueError(f"_cluster_one_episode: ep_id={episode.get('id')} — episode embedding is missing")

    import numpy as np  # local import to avoid top-level dep on benchmarks side

    vec = np.asarray(raw_vec, dtype=np.float32)
    episode_text: str = episode.get("episode") or ""
    new_cluster = Cluster(
        centroid=vec,
        last_ts=int(episode["timestamp"]),
        members=[str(episode["id"])],
        preview=[episode_text],
    )
    merged = cluster_by_geometry(
        new_cluster,
        existing_clusters,
        threshold=threshold,
        time_window_days=time_window_days,
    )
    if merged is None:
        new_id = f"cluster_{len(existing_clusters)}"
        existing_clusters = [*existing_clusters, new_cluster.model_copy(update={"id": new_id})]
    else:
        existing_clusters = [merged if c.id == merged.id else c for c in existing_clusters]
    return existing_clusters


def _build_clusters_data(clusters: list[Cluster]) -> list[dict[str, Any]]:
    """Convert Cluster objects to raw dicts suitable for ``serialize_clusters``.

    Centroid is serialised as a plain Python float list (``tolist()``) so
    ``json.dumps`` does not choke on ``np.float32`` values.
    """
    return [
        {
            "id": cluster.id,
            "centroid": cluster.centroid.tolist(),
            "count": cluster.count,
            "last_ts": cluster.last_ts,
            "members": cluster.members,
            "preview": cluster.preview,
        }
        for cluster in clusters
    ]


async def _run_clustering_pass(
    conv_idx: int,
    episodes: list[dict[str, Any]],
    memcell_to_episode: dict[str, str],
    output_dir: Path,
    *,
    threshold: float,
    time_window_days: float,
) -> int:
    """Cluster episodes by geometric similarity; persist result.

    Runs sequentially (one episode per iteration) — correctness over throughput,
    since each assignment depends on the accumulated cluster state from the
    previous step.

    Returns:
        Number of clusters produced.
    """
    clusters: list[Cluster] = []
    for ep in episodes:
        clusters = await _cluster_one_episode(
            ep,
            clusters,
            threshold=threshold,
            time_window_days=time_window_days,
        )

    clusters_data = _build_clusters_data(clusters)
    cluster_dict = serialize_clusters(clusters_data, memcell_to_episode)
    write_json(output_dir / f"clusters_conv_{conv_idx}.json", cluster_dict)
    return len(clusters)


def _write_conv_stats(
    conv_idx: int,
    memcells: list[dict[str, Any]],
    output_dir: Path,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    n_clusters: int | None,
) -> None:
    """Write per-conversation stats to ``stats_conv_{conv_idx}.json``.

    Args:
        conv_idx: Conversation index.
        memcells: Extracted MemCell list (used for count).
        output_dir: Stage output directory.
        prompt_tokens: Prompt tokens consumed for this conversation.
        completion_tokens: Completion tokens consumed for this conversation.
        n_clusters: Number of clusters produced, or None if clustering is disabled/failed.
    """
    stats: dict[str, Any] = {
        "conv_idx": conv_idx,
        "memcells": len(memcells),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "clustering_enabled": n_clusters is not None,
    }
    if n_clusters is not None:
        stats["clustering"] = {"total_memcells": len(memcells), "new_clusters": n_clusters}
    out = output_dir / f"stats_conv_{conv_idx}.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2))


async def _process_conversation(
    idx: int,
    conv: Conversation,
    llm: OpenAICompatClient,
    conv_sem: asyncio.Semaphore,
    mc_sem: asyncio.Semaphore,
    *,
    output_dir: Any,
    smart_mask: bool,
    max_attempts: int,
    ctx: StageContext,
    inner_pbar: _tqdm[Any] | None = None,
    msg_limit: int | None = None,
) -> tuple[bool, int, int]:
    """Process one conversation under the conv semaphore.

    Writes three entity files:
    - ``memcells_conv_<idx>.json`` — pure MemCells (boundary segments).
    - ``episodes_conv_<idx>.json`` — Episode entities with embeddings.
    - ``clusters_conv_<idx>.json`` — Clusters with episode_ids + episode_to_cluster.

    The ``mc_sem`` is shared across all conversations to bound total concurrent
    MemCell LLM calls, matching the upstream reference's Semaphore(20) design.

    Returns:
        Tuple of (success, estimated_prompt_tokens, estimated_completion_tokens).
    """
    async with conv_sem:
        try:
            memcells, episodes, prompt_tokens, completion_tokens = await _extract_one_conversation(
                conv,
                llm,
                ctx.services.embedding,
                semaphore=mc_sem,
                smart_mask=smart_mask,
                max_attempts=max_attempts,
                conv_idx=idx,
                pbar=inner_pbar,
                msg_limit=msg_limit,
            )
        except Exception:  # per-conv isolation; errors written to .error.txt
            err_path = output_dir / f"memcells_conv_{idx}.error.txt"
            err_path.write_text(traceback.format_exc())
            logger.exception("conv_%d extraction failed; full traceback in %s", idx, err_path)
            return False, 0, 0

        write_json(output_dir / f"memcells_conv_{idx}.json", memcells)
        write_json(output_dir / f"episodes_conv_{idx}.json", episodes)

        # 1:1 memcell-to-episode mapping (before Reflection stage which may merge)
        memcell_to_episode = {str(i): str(i) for i in range(len(memcells))}

        n_clusters = await _run_clustering_pass(
            idx,
            episodes,
            memcell_to_episode,
            output_dir,
            threshold=ctx.config.cluster_similarity_threshold,
            time_window_days=ctx.config.cluster_max_time_gap_days,
        )

        _write_conv_stats(idx, memcells, output_dir, prompt_tokens, completion_tokens, n_clusters=n_clusters)
        return True, prompt_tokens, completion_tokens


def _aggregate_extract_stats(
    stats: StageStats,
    results: list[tuple[bool, int, int]],
    started: float,
) -> StageStats:
    """Fill ``stats`` fields from per-conversation results."""
    stats.success = sum(1 for ok, _, _ in results if ok)
    stats.failed = sum(1 for ok, _, _ in results if not ok)
    stats.prompt_tokens = sum(p for _, p, _ in results)
    stats.completion_tokens = sum(c for _, _, c in results)
    stats.duration_seconds = time.monotonic() - started
    return stats


async def _run_one_extract(
    i: int,
    conv: Conversation,
    llm: Any,
    conv_sem: asyncio.Semaphore,
    mc_sem: asyncio.Semaphore,
    *,
    ctx: StageContext,
    smart_mask: bool,
    max_attempts: int,
    smoke_msg_limit: int | None,
    position_pool: asyncio.Queue[int],
    outer_pbar: _tqdm[Any],
) -> tuple[bool, int, int]:
    """Process one conversation with inner progress bar drawn from a position pool."""
    eligible = [m for m in conv.messages if m.role in ("user", "assistant")]
    if smoke_msg_limit is not None:
        eligible = eligible[:smoke_msg_limit]
    pos = await position_pool.get()
    inner_pbar = _tqdm(
        total=len(eligible), desc=f"conv {i:02d}", unit="msg", position=pos, leave=False, dynamic_ncols=True
    )
    try:
        result = await _process_conversation(
            i,
            conv,
            llm,
            conv_sem,
            mc_sem,
            output_dir=ctx.output_dir,
            smart_mask=smart_mask,
            max_attempts=max_attempts,
            ctx=ctx,
            inner_pbar=inner_pbar,
            msg_limit=smoke_msg_limit,
        )
        outer_pbar.update(1)
        return result
    finally:
        inner_pbar.close()
        position_pool.put_nowait(pos)


def _load_extract_conversations(ctx: StageContext) -> list[Conversation]:
    """Load and filter conversations for the extract stage."""
    conversations = list(ctx.dataset.load_conversations())
    if ctx.conv_indices is not None:
        conversations = [c for i, c in enumerate(conversations) if i in set(ctx.conv_indices)]
    elif ctx.smoke:
        conversations = conversations[: ctx.smoke_conv_limit]
    return conversations


async def run_extract_base_stage(ctx: StageContext) -> StageStats:
    """Stage 1 — Extract Base: boundary detection + episode extraction + clustering.

    Reads conversations from ``ctx.dataset``, runs BoundaryDetector +
    EpisodeExtractor on each, and writes three JSON files per conversation
    under ``ctx.output_dir``. Errors are isolated per conversation.
    """
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage_name="extract")
    started = time.monotonic()
    llm = build_llm_client(ctx.config)
    conversations = _load_extract_conversations(ctx)

    conv_sem = asyncio.Semaphore(ctx.config.max_concurrent_convs)
    mc_sem = asyncio.Semaphore(20)
    n_convs = len(conversations)
    smoke_msg_limit = ctx.smoke_msg_limit if ctx.smoke else None

    max_slots = min(n_convs, ctx.config.max_concurrent_convs)
    position_pool: asyncio.Queue[int] = asyncio.Queue()
    for _p in range(1, max_slots + 1):
        position_pool.put_nowait(_p)
    outer_pbar = _tqdm(total=n_convs, desc="extract", unit="conv", position=0, leave=True, dynamic_ncols=True)

    results: list[tuple[bool, int, int]] = list(
        await asyncio.gather(
            *(
                _run_one_extract(
                    i,
                    conv,
                    llm,
                    conv_sem,
                    mc_sem,
                    ctx=ctx,
                    smart_mask=ctx.config.extract_smart_mask,
                    max_attempts=ctx.config.llm_max_retries,
                    smoke_msg_limit=smoke_msg_limit,
                    position_pool=position_pool,
                    outer_pbar=outer_pbar,
                )
                for i, conv in enumerate(conversations)
            )
        )
    )
    outer_pbar.close()
    return _aggregate_extract_stats(stats, results, started)


# Backward-compatible alias for callers that import the old name.
run_extract_stage = run_extract_base_stage
