"""Stage 2 — BM25 + Embedding index building (fact-level MaxSim).

For each ``memcells_conv_<i>.json`` produced by Stage 1:

- **BM25** (``bm25_conv_<i>.pkl``): a *fact-level* corpus. Every searchable
  unit (each atomic_fact, plus ``episode.subject`` and ``episode.summary``)
  is tokenized into its own BM25 document; ``fact_to_doc_idx`` maps the
  fact-level row back to its parent memcell. At search time the caller takes
  the max BM25 score across the doc's facts (MaxSim aggregation).
  See ``stage2_index_building.py:240-293`` for the original reference implementation.

- **Embedding** (``emb_conv_<i>.pkl``): per-memcell dict
  ``{"doc": memcell, "embeddings": {"atomic_facts": [vec, ...], "subject": vec,
  "summary": vec}}``. ``subject`` / ``summary`` embeddings are persisted
  even when atomic_facts are non-empty (``build_emb_index`` 332-379) so the
  MaxSim retrieval also covers topic-level signals.

- **Cluster index** (``cluster_index_conv_<i>.pkl``): when upstream clustering is
  enabled (``config.enable_cluster_retrieval=True``) and Stage 1 emits a
  ``clusters_conv_<i>.json``, this stage reshapes those cluster assignments
  into a cluster-index dict consumed by Stage 3's 2-level retrieval path.
  A missing cluster file raises ``FileNotFoundError`` and terminates the stage
  (fast-fail) rather than silently falling back, which would corrupt Stage 3
  metrics.
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
from everalgo.clustering import Cluster

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


# ``build_emb_index`` 343-350 strips these prefixes off summaries before embedding so the leading
# "in this conversation, ..." boilerplate does not dilute the vector. In this pipeline summary is
# unconditionally ``episode_body[:200] + "..."`` (see ``extract.py:_extract_memcell_data``), so
# the prefixes only hit when Episode content itself opens with one — rare but harmless to apply.
_SUMMARY_PREFIX_NOISE: tuple[str, ...] = (
    "In this conversation,",
    "The conversation is about",
    "This dialogue discusses",
    "Here, ",
)


def _clean_summary(text: str) -> str:
    """Strip known summary-prefix noise (case-sensitive, single pass)."""
    if not text:
        return ""
    for prefix in _SUMMARY_PREFIX_NOISE:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text


def _extract_atomic_strings(mc: dict[str, Any]) -> list[str]:
    """Return non-empty fact strings from a memcell's ``atomic_facts`` field.

    Stage 1 emits ``atomic_facts`` as a dict ``{"time", "timestamp",
    "atomic_fact": list[str], "fact_embeddings": list[list[float]]}`` — this is
    the only supported shape (see ``benchmarks/common/stages/extract.py:127``).
    """
    raw = mc.get("atomic_facts")
    if not isinstance(raw, dict):
        return []
    raw_dict: dict[str, Any] = cast("dict[str, Any]", raw)
    raw_facts: list[Any] = cast("list[Any]", raw_dict.get("atomic_fact") or [])
    return [f.strip() for f in raw_facts if isinstance(f, str) and f.strip()]


def _has_precomputed_fact_embeddings(mc: dict[str, Any]) -> bool:
    """Return True when Stage 1 already embedded atomic facts for this cell.

    Validates length parity between ``atomic_fact`` and ``fact_embeddings`` so a
    partially-failed Stage 1 run does not silently produce a misaligned index.
    """
    af = mc.get("atomic_facts")
    if not isinstance(af, dict):
        return False
    af_dict: dict[str, Any] = cast("dict[str, Any]", af)
    fact_embs: list[Any] = cast("list[Any]", af_dict.get("fact_embeddings") or [])
    atomic_fact_list: list[Any] = cast("list[Any]", af_dict.get("atomic_fact") or [])
    return bool(fact_embs) and len(fact_embs) == len(atomic_fact_list)


def extract_searchable_units(mc: dict[str, Any]) -> list[str]:
    """Return the list of strings to index for one memcell (fact-level granularity).

    Implements the logic of ``extract_atomic_facts`` (``stage2_index_building.py:129-165``):

    1. Every atomic_fact string (one per row in the BM25 corpus).
    2. ``episode.subject`` (topic-level signal — added once, not replicated).
    3. ``episode.summary`` (raw — 93 ``extract_atomic_facts`` uses ``doc["summary"]``
       literal without prefix cleaning; only the embedding path strips prefixes).
    4. Fallback: if none of the above produced any text, return ``[episode.content]``
       as a single row so the memcell is at least represented.
    """
    units = _extract_atomic_strings(mc)

    episode: dict[str, Any] = mc.get("episode") or {}
    subject = cast("str", episode.get("subject") or "")
    summary = cast("str", episode.get("summary") or "")
    content = cast("str", episode.get("content") or "")

    if subject:
        units.append(subject)
    if summary:
        units.append(summary)

    if not units and content:
        units.append(content)

    return units


def _build_bm25_fact_level(memcells: list[dict[str, Any]]) -> tuple[Any, list[int]] | None:
    """Build a fact-level BM25 corpus + ``fact_to_doc_idx`` parent mapping.

    NLTK dependency (data download + stemmer + stopwords) is fully private to
    BM25 indexing; callers don't need to know about it. ``_ensure_nltk`` is
    idempotent and cheap when data is already present (3 ``nltk.data.find``
    calls per invocation, ~ms). Implements the ``build_bm25_index`` logic
    (``stage2_index_building.py:240-293``).

    Returns ``(BM25Okapi, fact_to_doc_idx)`` or ``None`` if no memcell produced
    any tokenizable searchable unit (entire conv would be a degenerate index).
    """
    _ensure_nltk()
    stemmer: Any = PorterStemmer()
    stop_words: set[str] = set(cast("list[str]", stopwords.words("english")))  # type: ignore[no-untyped-call]
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

    Selection rules (``build_emb_index`` 332-386):
    - ``subject`` and cleaned ``summary`` are always embedded when present
      (independent of whether ``atomic_facts`` exists).
    - ``atomic_facts`` is embedded fact-by-fact when present; when Stage 1
      precomputed matching-length ``fact_embeddings``, skip re-embedding and
      let ``_inject_precomputed_fact_embeddings`` fill them in afterwards.
    - **Fallback** to ``episode.content`` (use ``doc["episode"]``) **only when atomic_facts is missing** (i.e.
      no ``atomic_strs`` and no precomputed fact embeddings). ``subject`` /
      ``summary`` being already queued does NOT suppress this fallback.

    Order: atomic_facts → subject → summary → episode-fallback. Different from
    93's subject → summary → atomic_facts → episode-fallback, but algorithmically
    equivalent (embedding API is deterministic: same text → same vector).
    """
    texts: list[str] = []
    mapping: list[tuple[int, str]] = []

    for doc_idx, mc in enumerate(memcells):
        atomic_strs = _extract_atomic_strings(mc)
        episode_dict: dict[str, Any] = mc.get("episode") or {}
        subject = cast("str", episode_dict.get("subject") or "")
        summary = _clean_summary(cast("str", episode_dict.get("summary") or ""))
        content = cast("str", episode_dict.get("content") or "")

        has_precomputed = _has_precomputed_fact_embeddings(mc)

        if atomic_strs and not has_precomputed:
            for fact_idx, fact in enumerate(atomic_strs):
                texts.append(fact)
                mapping.append((doc_idx, f"atomic_fact_{fact_idx}"))

        if subject:
            texts.append(subject)
            mapping.append((doc_idx, "subject"))

        if summary:
            texts.append(summary)
            mapping.append((doc_idx, "summary"))

        # Fallback: episode.content (under "episode" field) only when atomic_facts
        # is missing. Subject/summary already queued do NOT suppress this fallback.
        if not atomic_strs and not has_precomputed and content:
            texts.append(content)
            mapping.append((doc_idx, "episode"))

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


def _inject_precomputed_fact_embeddings(
    doc_embeddings: list[dict[str, Any]],
    memcells: list[dict[str, Any]],
) -> None:
    """Fill ``atomic_facts`` embeddings from Stage 1 ``fact_embeddings`` for cells that skipped re-embedding.

    Called after ``_reassemble_embeddings``. For each cell where ``atomic_facts``
    is not yet populated (because ``_flatten_conv_texts`` skipped it), load the
    pre-computed vectors from ``atomic_facts.fact_embeddings`` and convert to
    ``np.ndarray``.  Cells already populated by the live embedding step are skipped.

    Args:
        doc_embeddings: Output of ``_reassemble_embeddings`` — mutated in place.
        memcells: Original memcell dicts aligned with ``doc_embeddings``.
    """
    for doc_entry, mc in zip(doc_embeddings, memcells, strict=True):
        emb_dict: dict[str, Any] = doc_entry["embeddings"]
        if "atomic_facts" in emb_dict:
            continue  # Already filled by the live embedding step
        af = mc.get("atomic_facts")
        if not isinstance(af, dict):
            continue
        af_dict: dict[str, Any] = cast("dict[str, Any]", af)
        fact_embs: list[Any] = cast("list[Any]", af_dict.get("fact_embeddings") or [])
        if fact_embs:
            emb_dict["atomic_facts"] = [np.array(v, dtype=np.float32) for v in fact_embs]


async def _embed_batched(
    texts: list[str],
    ctx: StageContext,
) -> list[Any]:
    """Embed *all* texts using fixed-size batches with bounded group concurrency.

    Implements the ``build_emb_index`` batching loop (lines 428-450): batches are grouped into sets
    of ``embedding_concurrent_batches``; each group is awaited with ``asyncio.gather`` before the
    next group starts. A 1-second sleep is inserted between groups when ``MAX_CONCURRENT_BATCHES > 1``
    and there is at least one more group pending.

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


def _build_cluster_index(clusters_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Reshape a Stage 1 cluster JSON into ``list[Cluster.model_dump()]`` for Stage 3.

    Aligns the cluster pkl schema with the algo ``Cluster`` type
    (``everalgo.clustering.state.Cluster``): same field names (``id`` / ``count`` /
    ``last_ts`` / ``members`` / ``centroid`` / ``preview``), no reverse map (the algo
    operator builds it on demand), no total counters (caller does not consume them).

    Args:
        clusters_data: Parsed content of ``clusters_conv_<i>.json`` from Stage 1.
            Expected to carry a top-level ``"clusters"`` list whose dicts match the
            algo ``Cluster`` field names already. ``centroid`` values may arrive as
            plain ``list[float]`` (JSON-serialized); they are converted to ``np.ndarray``
            before ``Cluster.model_validate`` is called (pydantic v2 ``arbitrary_types_allowed``
            does not coerce list → ndarray automatically).

    Returns:
        A list of ``Cluster.model_dump()`` dicts. Stage 3 wraps via ``Cluster.model_validate``.
    """
    raw_clusters: list[dict[str, Any]] = list(clusters_data.get("clusters") or [])
    return [Cluster.model_validate({**c, "centroid": np.array(c["centroid"])}).model_dump() for c in raw_clusters]


def _build_and_write_bm25_index(
    conv_idx: int,
    memcells: list[dict[str, Any]],
    output_dir: Path,
) -> bool:
    """Build fact-level BM25 corpus and persist it to ``bm25_conv_<i>.pkl``.

    Returns ``False`` if the corpus is degenerate (no tokenizable searchable
    unit in any memcell) — caller treats this as a legitimate skip, not an error.
    """
    bm25_built = _build_bm25_fact_level(memcells)
    if bm25_built is None:
        return False
    bm25_obj, fact_to_doc_idx = bm25_built
    payload = {
        "bm25": bm25_obj,
        "docs": memcells,
        "fact_to_doc_idx": fact_to_doc_idx,
        "index_type": "maxsim",
    }
    out = output_dir / f"bm25_conv_{conv_idx}.pkl"
    with out.open("wb") as fh:
        pickle.dump(payload, fh)
    return True


async def _build_and_write_emb_index(
    conv_idx: int,
    memcells: list[dict[str, Any]],
    output_dir: Path,
    ctx: StageContext,
) -> int:
    """Build embedding index and persist it to ``emb_conv_<i>.pkl``.

    Returns estimated prompt tokens consumed by the embedding API.
    """
    texts_to_embed, doc_field_map = _flatten_conv_texts(memcells)
    conv_tokens = sum(estimate_tokens(t) for t in texts_to_embed)

    if texts_to_embed:
        all_vectors = await _embed_batched(texts_to_embed, ctx)
        emb_index = _reassemble_embeddings(memcells, all_vectors, doc_field_map)
    else:
        emb_index = [{"doc": mc, "embeddings": {}} for mc in memcells]
    _inject_precomputed_fact_embeddings(emb_index, memcells)

    out = output_dir / f"emb_conv_{conv_idx}.pkl"
    with out.open("wb") as fh:
        pickle.dump(emb_index, fh)
    return conv_tokens


async def _process_one_conversation(
    conv_idx: int,
    input_path: Path,
    output_dir: Path,
    ctx: StageContext,
    conv_sem: asyncio.Semaphore,
) -> tuple[bool, int]:
    """Build BM25 + embedding + cluster indices for one conversation file.

    Three steps fan out sequentially: BM25 → embedding → cluster (gated by
    ``config.enable_cluster_retrieval``). All three are fast-fail: any uncaught
    exception writes ``index_conv_<i>.error.txt`` and re-raises so ``asyncio.gather``
    terminates the entire stage. We do not soft-skip on errors here, because a
    partial stage 2 output would silently corrupt Stage 3 metrics.

    Returns:
        Tuple of (success, estimated_prompt_tokens). ``success=False`` only on
        legitimate empty / degenerate inputs (no memcells, or no tokenizable
        searchable units), not on real errors — those raise instead.
    """
    async with conv_sem:
        try:
            memcells: list[dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))
            if not memcells:
                return False, 0

            if not _build_and_write_bm25_index(conv_idx, memcells, output_dir):
                return False, 0
            conv_tokens = await _build_and_write_emb_index(conv_idx, memcells, output_dir, ctx)
            if ctx.config.enable_cluster_retrieval:
                _build_and_write_cluster_index(conv_idx, output_dir, ctx)
        except Exception:
            err_path = output_dir / f"index_conv_{conv_idx}.error.txt"
            err_path.write_text(traceback.format_exc())
            _log.exception("conv_%d index build failed; full traceback in %s", conv_idx, err_path)
            raise

    return True, conv_tokens


def _build_and_write_cluster_index(conv_idx: int, output_dir: Path, ctx: StageContext) -> None:
    """Build and persist the cluster index for one conversation.

    Failure propagates to the caller — cluster index errors terminate the stage
    rather than silently falling back to flat hybrid retrieval in Stage 3.
    """
    cluster_path = ctx.input_dir / f"clusters_conv_{conv_idx}.json"
    if not cluster_path.exists():
        raise FileNotFoundError(
            f"enable_cluster_retrieval=True but cluster file missing for conv_{conv_idx}; expected: {cluster_path}"
        )

    clusters_data: dict[str, Any] = json.loads(cluster_path.read_text(encoding="utf-8"))
    cluster_index = _build_cluster_index(clusters_data)
    cluster_out = output_dir / f"cluster_index_conv_{conv_idx}.pkl"
    with cluster_out.open("wb") as fh:
        pickle.dump(cluster_index, fh)


async def run_index_stage(ctx: StageContext) -> StageStats:
    """Stage 2 — build BM25 + Embedding indices per conversation.

    Concurrency is two-tiered:
    - ``conv_sem``: limits how many conversations are processed in parallel
      (bound by ``max_concurrent_convs`` from config, shared with Stage 1).
    - Within each conversation the embedding fan-out is bounded by
      ``embedding_concurrent_batches`` (batch-group strategy); no global
      ``emb_sem`` is needed.
    """
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage_name="index")
    started = time.monotonic()

    input_files = sorted(ctx.input_dir.glob("memcells_conv_*.json"))

    conv_sem = asyncio.Semaphore(ctx.config.max_concurrent_convs)

    coros = [
        _process_one_conversation(int(p.stem.rsplit("_", 1)[-1]), p, ctx.output_dir, ctx, conv_sem) for p in input_files
    ]

    from benchmarks.common._progress import gather_with_progress

    results = await gather_with_progress(*coros, desc="index", unit="conv")

    stats.success = sum(1 for ok, _ in results if ok)
    stats.failed = sum(1 for ok, _ in results if not ok)
    stats.prompt_tokens = sum(t for _, t in results)
    stats.duration_seconds = time.monotonic() - started
    return stats
