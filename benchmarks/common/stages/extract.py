"""Stage 1 — MemCell extraction.

For each conversation: BoundaryDetector segments messages into MemCells; for
each MemCell run EpisodeExtractor + AtomicFactExtractor. Output is one
``memcells_conv_<i>.json`` per conversation, in a shape compatible with
the upstream reference's evaluation pipeline.

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

from pydantic import SecretStr
from tqdm import tqdm as _tqdm

from benchmarks.common.dataset import AtomicFactRecord, EpisodeMemoryRecord
from benchmarks.common.metrics import estimate_tokens
from benchmarks.common.stages.types import StageStats
from everalgo.clustering.algorithm import cluster_by_geometry
from everalgo.clustering.state import Cluster
from everalgo.llm.config import LLMConfig
from everalgo.llm.format import format_atomic_fact_time
from everalgo.llm.parse import retry_on_json_parse_failure
from everalgo.llm.providers.openai_compat import OpenAICompatClient
from everalgo.types import AtomicFact
from everalgo.types import ChatMessage as EverAlgoMemMessage
from everalgo.user_memory import BoundaryDetector
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.episode import EpisodeExtractor
from everalgo.user_memory.prompts.en.atomic_fact_from_text import EVENT_LOG_PROMPT

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.common.dataset import Conversation
    from benchmarks.common.services import EmbeddingClient
    from benchmarks.common.stages.types import StageContext

logger = logging.getLogger(__name__)


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
    record: EpisodeMemoryRecord,
    atomic_record: AtomicFactRecord,
) -> dict[str, Any]:
    """Build the EverAlgo-native dict for a single MemCell.

    Schema is a clean superset of EverAlgo's MemCell / Episode types. ``atomic_facts``
    is persisted as a structured dict (time, timestamp, atomic_fact, fact_embeddings)
    matching :class:`AtomicFactRecord`; the episode embedding is stored once under
    ``episode.content_embeddings`` and used only for Stage 1 clustering.
    """
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
            "subject": record.subject,
            "summary": record.summary,
            "content": record.content,
            "content_embeddings": record.embedding,
        },
        "atomic_facts": {
            "time": atomic_record.time,
            "timestamp": atomic_record.timestamp,
            "atomic_fact": atomic_record.atomic_fact,
            "fact_embeddings": atomic_record.fact_embeddings,
        },
    }


async def _extract_memcell_data(
    mc: Any,
    mc_idx: int,
    llm: OpenAICompatClient,
    embedding_client: EmbeddingClient,
    *,
    max_attempts: int,
) -> tuple[dict[str, Any], int, int]:
    """Extract episode + atomic facts for one MemCell and serialize to dict.

    Pipeline:
    1. EpisodeExtractor — compress raw MemCell into a narrative episode.
       Raises ``ValueError`` if the episode body is empty (fail-loud; mirrors 93 stage1).
    2. Parallel: AtomicFactExtractor + episode-body embedding.
       Raises ``ValueError`` if no atomic facts are returned (fail-loud).
    3. Sequential: batch-embed all atomic facts to build ``fact_embeddings``.

    Returns:
        Tuple of (serialized memcell dict, estimated prompt tokens, estimated
        completion tokens). Token counts are tiktoken approximations.
    """

    @retry_on_json_parse_failure(max_retries=max_attempts)
    async def _do_extract_episode() -> Any:
        return await EpisodeExtractor(llm=llm).aextract(mc, sender_id=None)

    episode = await _do_extract_episode()
    episode_body: str = getattr(episode, "episode", "") or ""
    if not episode_body:
        raise ValueError(f"EpisodeExtractor returned empty episode body (mc_idx={mc_idx})")

    @retry_on_json_parse_failure(max_retries=max_attempts)
    async def _do_extract_facts() -> list[AtomicFact]:
        # Use EVENT_LOG_PROMPT (event_log schema key) rather than the algo default
        # ATOMIC_FACT_FROM_TEXT_PROMPT_EN (atomic_facts schema key). Both prompts share the same
        # inner schema; the parser tolerates either top-level key.
        return await AtomicFactExtractor(llm=llm).aextract_from_text(
            episode_body, timestamp=mc.timestamp, prompt=EVENT_LOG_PROMPT
        )

    async def _do_embed_episode() -> list[float] | None:
        vecs = await embedding_client.embed([episode_body])
        if not vecs:
            raise ValueError(f"embed returned empty vector list for episode body (mc_idx={mc_idx})")
        return list(vecs[0])

    facts, embedding = await asyncio.gather(_do_extract_facts(), _do_embed_episode())
    fact_strings = [af.content for af in facts]
    time_str = format_atomic_fact_time(mc.timestamp)

    if not fact_strings:
        raise ValueError(f"AtomicFactExtractor returned no facts for non-empty episode body (mc_idx={mc_idx})")
    raw_fact_vecs = await embedding_client.embed(fact_strings)
    fact_embeddings: list[list[float]] = [list(v) for v in raw_fact_vecs]

    episode_subject: str = getattr(episode, "subject", "") or ""
    # Summary is unconditionally ``episode_body[:200] + "..."`` — no LLM-summary path. The trailing
    # ellipsis is load-bearing: downstream BM25 / embedding indices tokenise this field.
    episode_summary: str = episode_body[:200] + "..."
    record = EpisodeMemoryRecord(
        mc_id=str(mc_idx),
        timestamp=mc.timestamp,
        subject=episode_subject,
        summary=episode_summary,
        content=episode_body,
        embedding=embedding,
    )
    atomic_record = AtomicFactRecord(
        time=time_str,
        atomic_fact=fact_strings,
        fact_embeddings=fact_embeddings,
        timestamp=mc.timestamp,
    )
    serialized = _serialize_memcell(mc_idx, mc, record, atomic_record)

    # Tiktoken-based approximation
    input_tokens = sum(estimate_tokens(m["content"]) for m in serialized["items"])
    estimated_prompt = input_tokens * 2

    episode_dict: dict[str, Any] = serialized.get("episode") or {}
    output_tokens = estimate_tokens(str(episode_dict.get("content", ""))) + estimate_tokens(
        str(episode_dict.get("subject", ""))
    )
    output_tokens += sum(estimate_tokens(f) for f in atomic_record.atomic_fact)

    return serialized, estimated_prompt, output_tokens


async def _detect_all_boundaries(
    mem_messages: list[EverAlgoMemMessage],
    llm: OpenAICompatClient,
    *,
    smart_mask: bool = True,
    max_attempts: int = 5,
    pbar: _tqdm[Any] | None = None,
) -> list[Any]:
    """Incremental boundary detection — thin outer loop over ``adetect_step``.

    Owns only the caller-side concerns that the algorithm does not encapsulate:
    the front-2-buffer (no LLM for the first 2 messages)
    and the final-tail flush at end-of-stream.

    Smart-mask threshold gating, masking, force-split, the LLM call, and the
    cut-and-bridge state transition are all owned by ``adetect_step`` and
    surface here through ``DetectionResult.tail``.

    LLM JSON parse retry is handled by the caller-owned ``@retry_on_json_parse_failure`` wrapper inside
    this function (``max_attempts`` retries per step); if all retries fail, ``ValueError`` propagates up
    and the per-conversation ``try / except`` in ``_process_conversation`` writes ``*.error.txt``.

    Args:
        mem_messages: Ordered messages for one conversation.
        llm: LLM client bound to the BoundaryDetector instance.
        smart_mask: Default ``True``. Forwarded to ``adetect_step``;
            the threshold check (``len(history) > 5``) is enforced inside the algo.
        max_attempts: Maximum number of retries for LLM JSON parse failures per detection step.
        pbar: Optional tqdm bar to update once per message (caller owns lifecycle).

    Returns:
        Ordered list of MemCells (closed segments + final-tail flush).
    """
    detector = BoundaryDetector(llm=llm)

    @retry_on_json_parse_failure(max_retries=max_attempts)
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
    from typing import cast

    from everalgo.types import ConversationItem, MemCell

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
) -> tuple[list[dict[str, Any]], int, int]:
    """Run boundary detection + episode + atomic-fact extraction on one conversation.

    BoundaryDetector walks messages one-by-one via ``adetect_step`` (incremental
    per-message boundary detection). All MemCell extractors then run in
    parallel via asyncio.gather, bounded by the shared semaphore.

    Args:
        conv: Conversation to process.
        llm: LLM client shared across all extractors.
        embedding_client: Embedding client used to compute episode-body vectors concurrently with atomic-fact extraction.
        semaphore: Shared semaphore bounding total concurrent MemCell LLM calls.
        smart_mask: Passed to ``_detect_all_boundaries``; mirrors 93 default ``True``.
        max_attempts: JSON-parse retry budget for EpisodeExtractor / AtomicFactExtractor.
        conv_idx: Conversation index forwarded to pbar description.
        pbar: Optional tqdm bar to update during boundary detection (caller owns lifecycle).
        msg_limit: Truncate mem_messages to this many entries before processing (smoke mode).

    Returns:
        Tuple of (memcell list, total estimated prompt tokens, total estimated
        completion tokens) for this conversation.
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

    async def _gated(mc: Any, idx: int) -> tuple[dict[str, Any], int, int]:
        async with semaphore:
            result = await _extract_memcell_data(mc, idx, llm, embedding_client, max_attempts=max_attempts)
            if pbar is not None:
                pbar.update(1)
            return result

    results = list(await asyncio.gather(*(_gated(mc, idx) for idx, mc in enumerate(boundary_cells))))
    memcells = [r[0] for r in results]
    total_prompt = sum(r[1] for r in results)
    total_completion = sum(r[2] for r in results)
    return memcells, total_prompt, total_completion


async def _cluster_one_memcell(
    mc: dict[str, Any],
    existing_clusters: list[Cluster],
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
    episode: dict[str, Any] = mc.get("episode") or {}
    episode_body: str = episode.get("content") or ""
    raw_vec = episode.get("content_embeddings")
    if raw_vec is None:
        raise ValueError(f"_cluster_one_memcell: mc_id={mc.get('id')} — content_embeddings is missing")

    import numpy as np  # local import to avoid top-level dep on benchmarks side

    vec = np.asarray(raw_vec, dtype=np.float32)
    new_cluster = Cluster(
        centroid=vec,
        last_ts=int(mc["timestamp"]),
        members=[str(mc["id"])],
        preview=[episode_body],
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
    *,
    threshold: float,
    time_window_days: float,
) -> int:
    """Embed each MemCell and assign to a geometric cluster; persist result.

    Runs sequentially (one embed call per MemCell) — correctness over throughput,
    since each assignment depends on the accumulated cluster state from the
    previous step.

    On any failure the error is written to ``clusters_conv_<i>.error.txt`` and
    re-raised so ``_process_conversation`` can mark the conversation as failed.

    Returns:
        Number of clusters produced.
    """
    clusters: list[Cluster] = []
    for mc in memcells:
        clusters = await _cluster_one_memcell(
            mc,
            clusters,
            threshold=threshold,
            time_window_days=time_window_days,
        )

    cluster_data = _serialize_cluster_file(clusters)
    out = output_dir / f"clusters_conv_{conv_idx}.json"
    out.write_text(json.dumps(cluster_data, ensure_ascii=False, indent=2))
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
    output_dir: Any,
    *,
    smart_mask: bool,
    max_attempts: int,
    ctx: StageContext,
    inner_pbar: _tqdm[Any] | None = None,
    msg_limit: int | None = None,
) -> tuple[bool, int, int]:
    """Process one conversation under the conv semaphore.

    The ``mc_sem`` is shared across all conversations to bound total concurrent
    MemCell LLM calls, matching the upstream reference's Semaphore(20) design.

    Returns:
        Tuple of (success, estimated_prompt_tokens, estimated_completion_tokens).
    """
    async with conv_sem:
        try:
            memcells, prompt_tokens, completion_tokens = await _extract_one_conversation(
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

        out = output_dir / f"memcells_conv_{idx}.json"
        out.write_text(json.dumps(memcells, ensure_ascii=False, indent=2))

        n_clusters: int | None = None
        if ctx.config.enable_clustering:
            n_clusters = await _run_clustering_pass(
                idx,
                memcells,
                output_dir,
                threshold=ctx.config.cluster_similarity_threshold,
                time_window_days=ctx.config.cluster_max_time_gap_days,
            )

        _write_conv_stats(idx, memcells, output_dir, prompt_tokens, completion_tokens, n_clusters=n_clusters)
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
      matching the upstream reference's Semaphore(20) model.
    """
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage_name="extract")
    started = time.monotonic()

    llm = _build_llm(ctx)

    conversations = list(ctx.dataset.load_conversations())
    if ctx.smoke:
        conversations = conversations[: ctx.smoke_conv_limit]

    # conv_sem: bound simultaneous conversations; mc_sem: bound total MemCell LLM calls.
    conv_sem = asyncio.Semaphore(ctx.config.max_concurrent_convs)
    mc_sem = asyncio.Semaphore(20)  # bound total concurrent MemCell LLM calls

    smart_mask = ctx.config.extract_smart_mask
    max_attempts = ctx.config.llm_max_retries
    n_convs = len(conversations)

    # Position pool: outer bar at position 0, inner per-conv bars at positions 1..max_slots.
    # Slots are capped so the terminal doesn't overflow with too many bars at once.
    max_slots = min(n_convs, ctx.config.max_concurrent_convs)
    position_pool: asyncio.Queue[int] = asyncio.Queue()
    for _p in range(1, max_slots + 1):
        position_pool.put_nowait(_p)

    outer_pbar = _tqdm(total=n_convs, desc="extract", unit="conv", position=0, leave=True, dynamic_ncols=True)

    smoke_msg_limit = ctx.smoke_msg_limit if ctx.smoke else None

    async def _run_one(i: int, conv: Conversation) -> tuple[bool, int, int]:
        eligible = [m for m in conv.messages if m.role in ("user", "assistant")]
        if smoke_msg_limit is not None:
            eligible = eligible[:smoke_msg_limit]
        n_msgs = len(eligible)
        pos = await position_pool.get()
        inner_pbar = _tqdm(
            total=n_msgs,
            desc=f"conv {i:02d}",
            unit="msg",
            position=pos,
            leave=False,
            dynamic_ncols=True,
        )
        try:
            result = await _process_conversation(
                i,
                conv,
                llm,
                conv_sem,
                mc_sem,
                ctx.output_dir,
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

    results: list[tuple[bool, int, int]] = list(
        await asyncio.gather(*[_run_one(i, conv) for i, conv in enumerate(conversations)])
    )
    outer_pbar.close()

    stats.success = sum(1 for ok, _, _ in results if ok)
    stats.failed = sum(1 for ok, _, _ in results if not ok)
    stats.prompt_tokens = sum(p for _, p, _ in results)
    stats.completion_tokens = sum(c for _, _, c in results)
    stats.duration_seconds = time.monotonic() - started
    return stats
