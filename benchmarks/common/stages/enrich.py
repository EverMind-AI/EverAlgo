"""Stage 3 — Enrich: extract atomic facts + embeddings from final episodes.

Reads ``episodes_conv_<i>.json`` (output of Extract Base or Reflect) and produces
``atomic_facts_conv_<i>.json``.  This is the sole producer of atomic facts in the pipeline.

Concurrency model:
- Conversations are processed sequentially (one file at a time).
- Within each conversation, per-episode fact-extraction tasks run in parallel,
  bounded by ``asyncio.Semaphore(20)`` — matching the upstream reference's Semaphore(20).
- Embeddings for each episode's facts are batch-embedded in a single call.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import traceback
from typing import TYPE_CHECKING, Any

from benchmarks.common.retry import llm_retry
from benchmarks.common.services import build_llm_client
from benchmarks.common.stages.serialization import load_episodes, serialize_atomic_fact, write_json
from benchmarks.common.stages.types import StageStats
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.prompts.en.atomic_fact_from_text import EVENT_LOG_PROMPT

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.common.services import EmbeddingClient
    from benchmarks.common.stages.types import StageContext
    from everalgo.llm.providers.openai_compat import OpenAICompatClient

logger = logging.getLogger(__name__)


async def _extract_facts_for_episode(
    episode: dict[str, Any],
    ep_idx: int,
    llm: OpenAICompatClient,
    embedding_client: EmbeddingClient,
    *,
    max_attempts: int,
) -> list[dict[str, Any]]:
    """Extract and embed atomic facts for one episode.

    Pipeline:
    1. ``AtomicFactExtractor.aextract_from_text`` — LLM call to extract facts from
       the episode body.  Raises ``ValueError`` if zero facts are returned (fail-loud).
    2. Batch-embed all fact texts in a single ``embedding_client.embed`` call.
    3. Return serialised fact dicts ready for ``write_json``.

    Args:
        episode: Episode dict with ``id``, ``episode`` (text), ``timestamp``, and ``owner_id``.
        ep_idx: Zero-based index of this episode within the conversation (for error messages).
        llm: LLM client passed to ``AtomicFactExtractor``.
        embedding_client: Used for batch embedding all extracted facts.
        max_attempts: JSON-parse retry budget for the LLM call.

    Returns:
        List of serialised atomic-fact dicts (``id`` is relative to episode; caller re-indexes).

    Raises:
        ValueError: If ``AtomicFactExtractor`` returns zero facts.
    """
    episode_text: str = episode.get("episode") or ""
    timestamp: int = int(episode.get("timestamp") or 0)
    episode_id: str = str(episode.get("id") or ep_idx)
    owner_id: str | None = episode.get("owner_id")

    extractor = AtomicFactExtractor(llm=llm)

    @llm_retry(max_attempts=max_attempts)
    async def _do_extract() -> list[Any]:
        return await extractor.aextract_from_text(episode_text, timestamp=timestamp, prompt=EVENT_LOG_PROMPT)

    facts = await _do_extract()
    if not facts:
        raise ValueError(f"AtomicFactExtractor returned 0 facts for episode_id={episode_id} (ep_idx={ep_idx})")

    fact_texts = [af.content for af in facts]
    embeddings_batch = await embedding_client.embed(fact_texts)

    return [
        serialize_atomic_fact(
            i,
            content=fact.content,
            episode_id=episode_id,
            timestamp=timestamp,
            owner_id=owner_id,
            embeddings=list(emb),
        )
        for i, (fact, emb) in enumerate(zip(facts, embeddings_batch, strict=True))
    ]


async def _enrich_one_conversation(
    conv_idx: int,
    episodes: list[dict[str, Any]],
    llm: OpenAICompatClient,
    embedding_client: EmbeddingClient,
    *,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    max_attempts: int,
) -> bool:
    """Enrich one conversation: extract facts per episode, write ``atomic_facts_conv_<i>.json``.

    Errors are isolated per conversation; full traceback written to
    ``atomic_facts_conv_<conv_idx>.error.txt``.

    Args:
        conv_idx: Conversation index (used in output file names).
        episodes: List of episode dicts loaded from ``episodes_conv_<conv_idx>.json``.
        llm: LLM client for AtomicFactExtractor.
        embedding_client: Embedding client for batch fact embeddings.
        output_dir: Stage output directory.
        semaphore: Shared semaphore bounding parallel episode fact-extraction tasks.
        max_attempts: JSON-parse retry budget.

    Returns:
        ``True`` on success, ``False`` on any error.
    """
    try:

        async def _gated(ep: dict[str, Any], idx: int) -> list[dict[str, Any]]:
            async with semaphore:
                return await _extract_facts_for_episode(ep, idx, llm, embedding_client, max_attempts=max_attempts)

        results: list[list[dict[str, Any]]] = list(
            await asyncio.gather(*(_gated(ep, idx) for idx, ep in enumerate(episodes)))
        )
    except Exception:
        err_path = output_dir / f"atomic_facts_conv_{conv_idx}.error.txt"
        err_path.write_text(traceback.format_exc())
        logger.exception("conv_%d enrich failed; full traceback in %s", conv_idx, err_path)
        return False

    # Re-index facts globally within the conversation (flat list, IDs "0", "1", …)
    flat_facts: list[dict[str, Any]] = []
    global_idx = 0
    for ep_facts in results:
        for fact in ep_facts:
            flat_facts.append({**fact, "id": str(global_idx)})
            global_idx += 1

    write_json(output_dir / f"atomic_facts_conv_{conv_idx}.json", flat_facts)
    logger.debug("conv_%d: wrote %d atomic facts", conv_idx, len(flat_facts))
    return True


async def run_enrich_stage(ctx: StageContext) -> StageStats:
    """Extract atomic facts from final episodes and embed them.

    Reads ``episodes_conv_<i>.json`` from ``ctx.input_dir`` and writes
    ``atomic_facts_conv_<i>.json`` to ``ctx.output_dir``.

    Per-conversation errors (including ``ValueError`` from zero atomic facts) are caught by
    ``_enrich_one_conversation`` and counted in ``stats.failed`` rather than propagated.

    Concurrency:
    - Conversations are iterated sequentially (I/O bound on file reads).
    - Per-episode extraction tasks within a conversation run in parallel,
      bounded by a shared ``asyncio.Semaphore(20)``.

    Args:
        ctx: Stage execution context providing config, services, I/O dirs, and smoke flags.

    Returns:
        ``StageStats`` with ``stage_name="enrich"``, success/failed counts, and duration.
    """
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage_name="enrich")
    started = time.monotonic()

    llm = build_llm_client(ctx.config)
    embedding_client = ctx.services.embedding

    semaphore = asyncio.Semaphore(20)
    max_attempts = ctx.config.llm_max_retries

    episode_files = sorted(ctx.input_dir.glob("episodes_conv_*.json"))
    if ctx.smoke:
        episode_files = episode_files[: ctx.smoke_conv_limit]

    from tqdm import tqdm as _tqdm  # Deferred: optional dependency, avoid top-level import

    for ep_file in _tqdm(episode_files, desc="enrich", unit="conv", dynamic_ncols=True):
        # Extract conversation index from filename "episodes_conv_<i>.json"
        stem = ep_file.stem  # "episodes_conv_<i>"
        conv_idx = int(stem.split("_")[-1])

        episodes = load_episodes(ep_file)
        ok = await _enrich_one_conversation(
            conv_idx,
            episodes,
            llm,
            embedding_client,
            output_dir=ctx.output_dir,
            semaphore=semaphore,
            max_attempts=max_attempts,
        )
        if ok:
            stats.success += 1
        else:
            stats.failed += 1

    _passthrough_upstream_files(ctx.input_dir, ctx.output_dir)
    stats.duration_seconds = time.monotonic() - started
    return stats


def _passthrough_upstream_files(input_dir: Path, output_dir: Path) -> None:
    """Copy upstream entity files that Enrich doesn't modify (episodes, clusters, memcells)."""
    for pattern in ("memcells_conv_*.json", "episodes_conv_*.json", "clusters_conv_*.json"):
        for src in sorted(input_dir.glob(pattern)):
            dst = output_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
