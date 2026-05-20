"""Stage 2 — BM25 + Embedding index building (fact-level MaxSim).

For each ``memcells_conv_<i>.json`` produced by Stage 1:

- **BM25** (``bm25_conv_<i>.pkl``): a *fact-level* corpus. Every searchable
  unit (each atomic_fact, plus ``episode.subject`` and ``episode.summary``)
  is tokenized into its own BM25 document; ``fact_to_doc_idx`` maps the
  fact-level row back to its parent memcell. At search time the caller takes
  the max BM25 score across the doc's facts (MaxSim aggregation).
  Mirrors locomo-benchmark ``build_bm25_index`` (``stage2_index_building.py:240-293``).

- **Embedding** (``emb_conv_<i>.pkl``): per-memcell dict
  ``{"doc": memcell, "embeddings": {"atomic_facts": [vec, ...], "subject": vec,
  "summary": vec}}``. ``subject`` / ``summary`` embeddings are persisted
  even when atomic_facts are non-empty (mirror locomo-benchmark
  ``build_emb_index`` 332-379) so the MaxSim retrieval also covers
  topic-level signals.

- **Scene index** (``scene_index_conv_<i>.pkl``): when upstream clustering is
  enabled (``config.enable_scene_retrieval=True``) and Stage 1 emits a
  ``clusters_conv_<i>.json``, this stage reshapes those cluster assignments
  into a scene-index dict consumed by Stage 3's 2-level retrieval path.
  Missing cluster files produce a warning and a graceful skip; Stage 3 falls
  back to flat hybrid retrieval when the scene index is absent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import time
import traceback
from typing import TYPE_CHECKING, Any, cast

import nltk  # type: ignore[import-untyped]
import numpy as np
from nltk.corpus import stopwords  # type: ignore[import-untyped]
from nltk.stem import PorterStemmer  # type: ignore[import-untyped]
from nltk.tokenize import word_tokenize  # type: ignore[import-untyped]
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from benchmarks.common.metrics import estimate_tokens
from benchmarks.common.stages.types import StageStats

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.common.stages.types import StageContext

_log = logging.getLogger(__name__)


def _ensure_nltk() -> None:
    """Download required NLTK data if not present."""
    for find_path, download_id in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]:
        try:
            nltk.data.find(find_path)  # type: ignore[no-untyped-call]
        except LookupError:
            nltk.download(download_id, quiet=True)  # type: ignore[no-untyped-call]


def _tokenize(text: str, stemmer: Any, stop_words: set[str]) -> list[str]:
    """Lower -> tokenize -> keep alpha words len>=2 not stopword -> stem."""
    if not text:
        return []
    tokens: list[str] = word_tokenize(text.lower())  # type: ignore[no-untyped-call]
    return [str(stemmer.stem(t)) for t in tokens if t.isalpha() and len(t) >= 2 and t not in stop_words]


# Locomo-benchmark ``build_emb_index`` 343-350 strips these prefixes off summaries
# before embedding so the leading "in this conversation, ..." boilerplate does not
# dilute the vector. In our pipeline summary is the LLM-fallback ``content[:200]``
# truncation (Episode prompt does not emit summary), so the prefixes only hit when
# Episode content itself opens with one — rare but harmless to apply.
_SUMMARY_PREFIX_NOISE: tuple[str, ...] = (
    "In this conversation,",
    "The conversation is about",
    "This dialogue discusses",
    "Here, ",
)


def _clean_summary(text: str) -> str:
    """Strip locomo-benchmark's known summary-prefix noise (case-sensitive, single pass)."""
    if not text:
        return ""
    for prefix in _SUMMARY_PREFIX_NOISE:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _extract_atomic_strings(mc: dict[str, Any]) -> list[str]:
    """Return non-empty fact strings from a memcell's flat ``atomic_facts`` list."""
    results: list[str] = []
    raw_facts: list[Any] = list(mc.get("atomic_facts") or [])
    for f in raw_facts:
        if isinstance(f, dict):
            fact: str = cast("str", cast("dict[str, Any]", f).get("fact") or "")
        else:
            fact = str(f) if f else ""
        if fact and fact.strip():
            results.append(fact)
    return results


def extract_searchable_units(mc: dict[str, Any]) -> list[str]:
    """Return the list of strings to index for one memcell (fact-level granularity).

    Mirror of locomo-benchmark ``extract_atomic_facts``
    (``stage2_index_building.py:129-165``):

    1. Every atomic_fact string (one per row in the BM25 corpus).
    2. ``episode.subject`` (topic-level signal — added once, not replicated).
    3. ``episode.summary`` (cleaned of known LLM-narrated prefix noise).
    4. Fallback: if none of the above produced any text, return ``[episode.content]``
       as a single row so the memcell is at least represented.
    """
    units = _extract_atomic_strings(mc)

    episode: dict[str, Any] = mc.get("episode") or {}
    subject = cast("str", episode.get("subject") or "")
    summary = _clean_summary(cast("str", episode.get("summary") or ""))
    content = cast("str", episode.get("content") or "")

    if subject:
        units.append(subject)
    if summary:
        units.append(summary)

    if not units and content:
        units.append(content)

    return units


def _build_bm25_fact_level(
    memcells: list[dict[str, Any]],
    stemmer: Any,
    stop_words: set[str],
) -> tuple[Any, list[int]] | None:
    """Build a fact-level BM25 corpus + ``fact_to_doc_idx`` parent mapping.

    Returns ``(BM25Okapi, fact_to_doc_idx)`` or ``None`` if no memcell produced
    any tokenizable searchable unit (entire conv would be a degenerate index).
    """
    fact_corpus: list[list[str]] = []
    fact_to_doc_idx: list[int] = []
    for doc_idx, mc in enumerate(memcells):
        for unit in extract_searchable_units(mc):
            tokens = _tokenize(unit, stemmer, stop_words)
            if tokens:
                fact_corpus.append(tokens)
                fact_to_doc_idx.append(doc_idx)
    if not fact_corpus:
        return None
    return BM25Okapi(fact_corpus), fact_to_doc_idx


def _flatten_conv_texts(
    memcells: list[dict[str, Any]],
) -> tuple[list[str], list[tuple[int, str]]]:
    """Flatten all searchable units across every memcell of a conversation.

    Returns ``(texts_to_embed, doc_field_map)`` where ``doc_field_map[k]`` is
    ``(doc_idx, field_name)`` for ``texts_to_embed[k]``.

    Selection rules (mirror locomo-benchmark ``build_emb_index`` 332-386):
    - If ``atomic_facts`` is non-empty: embed each fact individually +
      ``subject`` (if present) + cleaned ``summary`` (if present). No content.
    - If ``atomic_facts`` is empty: embed ``subject`` + cleaned ``summary``
      (if either present). Fall back to ``content`` **only** when nothing else
      was queued for that cell.
    """
    texts: list[str] = []
    mapping: list[tuple[int, str]] = []

    for doc_idx, mc in enumerate(memcells):
        atomic_strs = _extract_atomic_strings(mc)
        episode_dict: dict[str, Any] = mc.get("episode") or {}
        subject = cast("str", episode_dict.get("subject") or "")
        summary = _clean_summary(cast("str", episode_dict.get("summary") or ""))
        content = cast("str", episode_dict.get("content") or "")

        cell_start = len(texts)

        if atomic_strs:
            for fact_idx, fact in enumerate(atomic_strs):
                texts.append(fact)
                mapping.append((doc_idx, f"atomic_fact_{fact_idx}"))

        if subject:
            texts.append(subject)
            mapping.append((doc_idx, "subject"))

        if summary:
            texts.append(summary)
            mapping.append((doc_idx, "summary"))

        # Fallback: content only when nothing else was queued for this cell.
        if len(texts) == cell_start and content:
            texts.append(content)
            mapping.append((doc_idx, "content"))

    return texts, mapping


def _reassemble_embeddings(
    memcells: list[dict[str, Any]],
    all_vectors: list[Any],
    doc_field_map: list[tuple[int, str]],
) -> list[dict[str, Any]]:
    """Rebuild per-doc ``{"doc": cell, "embeddings": {...}}`` dicts from flat vectors.

    ``atomic_fact_*`` fields are collected into an ``"atomic_facts"`` list in
    index order. All other fields are stored directly by name.
    """
    doc_embeddings: list[dict[str, Any]] = [{"doc": mc, "embeddings": {}} for mc in memcells]

    for (doc_idx, field), vec in zip(doc_field_map, all_vectors, strict=True):
        emb_dict: dict[str, Any] = doc_embeddings[doc_idx]["embeddings"]
        if field.startswith("atomic_fact_"):
            if "atomic_facts" not in emb_dict:
                emb_dict["atomic_facts"] = []
            cast("list[Any]", emb_dict["atomic_facts"]).append(np.array(vec, dtype=np.float32))
        else:
            emb_dict[field] = np.array(vec, dtype=np.float32)

    return doc_embeddings


async def _embed_batched(
    texts: list[str],
    ctx: StageContext,
) -> list[Any]:
    """Embed *all* texts using fixed-size batches with bounded group concurrency.

    Mirrors locomo-benchmark ``build_emb_index`` batching loop (lines 428-450):
    batches are grouped into sets of ``embedding_concurrent_batches``; each group
    is awaited with ``asyncio.gather`` before the next group starts.  A 1-second
    sleep is inserted between groups when ``MAX_CONCURRENT_BATCHES > 1`` and
    there is at least one more group pending.

    Returns the flat list of raw embedding vectors aligned with ``texts``.
    """
    batch_size: int = ctx.config.embedding_batch_size
    max_concurrent: int = ctx.config.embedding_concurrent_batches

    batches: list[list[str]] = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    total_batches = len(batches)

    async def _call_batch(batch_texts: list[str]) -> list[Any]:
        return await ctx.services.embedding.embed(batch_texts)

    # Collect (batch_idx, vectors) in submission order; reassemble after.
    ordered_results: list[tuple[int, list[Any]]] = []

    group_size = max_concurrent
    for group_start in range(0, total_batches, group_size):
        group = batches[group_start : group_start + group_size]
        group_indices = list(range(group_start, group_start + len(group)))

        group_vectors: list[list[Any]] = await asyncio.gather(*(_call_batch(b) for b in group))
        ordered_results.extend(zip(group_indices, group_vectors, strict=True))

        is_last_group = (group_start + group_size) >= total_batches
        if max_concurrent > 1 and not is_last_group:
            await asyncio.sleep(1.0)

    # Sort by batch index (gather already preserves order, but be explicit)
    ordered_results.sort(key=lambda x: x[0])
    all_vectors: list[Any] = []
    for _, vecs in ordered_results:
        all_vectors.extend(vecs)

    return all_vectors


def _build_scene_index(clusters_data: dict[str, Any]) -> dict[str, Any]:
    """Reshape a Stage 1 cluster JSON into the scene-index dict Stage 3 consumes.

    Args:
        clusters_data: Parsed content of ``clusters_conv_<i>.json`` as produced
            by EverAlgo Stage 1. Expected keys: ``"clusters"`` (list of cluster
            dicts) and ``"memcell_to_cluster"`` (mc_id -> scene_id mapping).

    Returns:
        Scene-index dict with keys ``"scenes"``, ``"memcell_to_scene"``,
        ``"total_scenes"``, and ``"total_memcells"``.  Centroids are kept as
        plain ``list[float]`` (JSON-native shape); Stage 3 converts to numpy
        when it needs cosine similarity.
    """
    raw_clusters: list[dict[str, Any]] = list(clusters_data.get("clusters") or [])
    memcell_to_cluster: dict[str, str] = dict(clusters_data.get("memcell_to_cluster") or {})

    scenes: list[dict[str, Any]] = [
        {
            "scene_id": cluster["id"],
            "centroid": list(cluster["centroid"]),
            "memcell_ids": list(cluster["members"]),
            "memcell_count": len(cluster["members"]),
            "last_timestamp": cluster["last_ts"],
        }
        for cluster in raw_clusters
    ]

    return {
        "scenes": scenes,
        "memcell_to_scene": memcell_to_cluster,
        "total_scenes": len(scenes),
        "total_memcells": len(memcell_to_cluster),
    }


async def _process_one_conversation(
    conv_idx: int,
    input_path: Path,
    output_dir: Path,
    ctx: StageContext,
    stemmer: Any,
    stop_words: set[str],
    conv_sem: asyncio.Semaphore,
) -> tuple[bool, int]:
    """Build BM25 + embedding indices for one conversation file.

    All memcells are flattened into a single ``texts_to_embed`` list, then
    embedded in fixed-size batches with at most ``embedding_concurrent_batches``
    in-flight at once (locomo-benchmark ``build_emb_index`` strategy).

    Returns:
        Tuple of (success, estimated_prompt_tokens).  Embedding has no
        completion side, so only prompt tokens are counted.
    """
    async with conv_sem:
        try:
            memcells: list[dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))
            if not memcells:
                return False, 0

            bm25_built = _build_bm25_fact_level(memcells, stemmer, stop_words)
            if bm25_built is None:
                return False, 0
            bm25_obj, fact_to_doc_idx = bm25_built
            bm25_payload = {
                "bm25": bm25_obj,
                "docs": memcells,
                "fact_to_doc_idx": fact_to_doc_idx,
                "index_type": "maxsim",
            }

            texts_to_embed, doc_field_map = _flatten_conv_texts(memcells)
            conv_tokens = sum(estimate_tokens(t) for t in texts_to_embed)

            if texts_to_embed:
                all_vectors = await _embed_batched(texts_to_embed, ctx)
                emb_index = _reassemble_embeddings(memcells, all_vectors, doc_field_map)
            else:
                emb_index = [{"doc": mc, "embeddings": {}} for mc in memcells]

            bm25_out = output_dir / f"bm25_conv_{conv_idx}.pkl"
            with bm25_out.open("wb") as fh:
                pickle.dump(bm25_payload, fh)

            emb_out = output_dir / f"emb_conv_{conv_idx}.pkl"
            with emb_out.open("wb") as fh:
                pickle.dump(emb_index, fh)

            if ctx.config.enable_scene_retrieval:
                _write_scene_index(conv_idx, output_dir, ctx)

        except Exception:
            err_path = output_dir / f"index_conv_{conv_idx}.error.txt"
            err_path.write_text(traceback.format_exc())
            return False, 0

    return True, conv_tokens


def _write_scene_index(conv_idx: int, output_dir: Path, ctx: StageContext) -> None:
    """Build and persist the scene index for one conversation.

    Isolated so that a scene-index failure does not roll back the already-written
    BM25 + embedding pickles.  On failure an ``.error.txt`` sidecar is written and
    a warning is logged; the conversation is NOT marked as failed.
    """
    cluster_path = ctx.input_dir / f"clusters_conv_{conv_idx}.json"
    if not cluster_path.exists():
        _log.warning(
            "enable_scene_retrieval=True but cluster file missing for conv_%d; "
            "skipping scene index (Stage 3 will fall back to flat hybrid). "
            "Expected: %s",
            conv_idx,
            cluster_path,
        )
        return

    try:
        clusters_data: dict[str, Any] = json.loads(cluster_path.read_text(encoding="utf-8"))
        scene_index = _build_scene_index(clusters_data)
        scene_out = output_dir / f"scene_index_conv_{conv_idx}.pkl"
        with scene_out.open("wb") as fh:
            pickle.dump(scene_index, fh)
    except Exception:
        err_path = output_dir / f"scene_index_conv_{conv_idx}.error.txt"
        err_path.write_text(traceback.format_exc())
        _log.warning(
            "Scene index build failed for conv_%d (soft warning — BM25/emb already written). "
            "Stage 3 will fall back to flat hybrid. See: %s",
            conv_idx,
            err_path,
        )


async def run_index_stage(ctx: StageContext) -> StageStats:
    """Stage 2 — build BM25 + Embedding indices per conversation.

    Concurrency is two-tiered:
    - ``conv_sem``: limits how many conversations are processed in parallel
      (bound by ``max_concurrent_qa`` from config, same as other stages).
    - Within each conversation the embedding fan-out is bounded by
      ``embedding_concurrent_batches`` (batch-group strategy); no global
      ``emb_sem`` is needed.
    """
    _ensure_nltk()

    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage_name="index")
    started = time.monotonic()

    stemmer: Any = PorterStemmer()
    raw_stop: list[str] = stopwords.words("english")  # type: ignore[no-untyped-call]
    stop_words: set[str] = set(raw_stop)

    input_files = sorted(ctx.input_dir.glob("memcells_conv_*.json"))

    conv_sem = asyncio.Semaphore(ctx.config.max_concurrent_qa)

    coros = [
        _process_one_conversation(int(p.stem.rsplit("_", 1)[-1]), p, ctx.output_dir, ctx, stemmer, stop_words, conv_sem)
        for p in input_files
    ]

    from tqdm.asyncio import tqdm as async_tqdm  # type: ignore[import-untyped]

    results: list[tuple[bool, int]] = await async_tqdm.gather(  # type: ignore[attr-defined]
        *coros,
        desc="index",
        unit="conv",
        dynamic_ncols=True,
    )

    stats.success = sum(1 for ok, _ in results if ok)
    stats.failed = sum(1 for ok, _ in results if not ok)
    stats.prompt_tokens = sum(t for _, t in results)
    stats.duration_seconds = time.monotonic() - started
    return stats
