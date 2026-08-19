"""Stage 2 — BM25 + Embedding index building (fact-level MaxSim).

Reads the entity-split files produced by the upstream Extract + Enrich stages:

- ``episodes_conv_<i>.json`` — Episode entities with embeddings.
- ``atomic_facts_conv_<i>.json`` — AtomicFact entities with embeddings.
- ``clusters_conv_<i>.json`` — Clusters with episode_ids + episode_to_cluster.

Produces per-conversation index artifacts:

- **BM25** (``bm25_conv_<i>.pkl``): a *fact-level* corpus. Every atomic-fact
  ``content`` string plus the parent episode's ``subject`` is tokenized into
  its own BM25 document; ``fact_to_doc_idx`` maps the fact-level row back to
  its parent episode index. At search time the caller takes the max BM25
  score across the doc's facts (MaxSim aggregation).

- **Embedding** (``emb_conv_<i>.pkl``): per-episode dict
  ``{"doc_id": episode_id, "embeddings": {"atomic_facts": [vec, ...],
  "subject": vec, "episode": vec}}``. Embeddings are read directly from
  the entity files (no re-embedding).

- **Cluster index** (``cluster_index_conv_<i>.pkl``): when upstream clustering
  is always built in agentic mode), reshaped cluster
  assignments consumed by Stage 3's 2-level retrieval path.
"""

from __future__ import annotations

import logging
import pickle
import time
import traceback
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from nltk.corpus import stopwords  # type: ignore[import-untyped]
from nltk.stem import PorterStemmer  # type: ignore[import-untyped]
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from benchmarks.common.stages._tokenize import ensure_nltk
from benchmarks.common.stages._tokenize import tokenize as _tokenize
from benchmarks.common.stages.serialization import load_atomic_facts, load_clusters, load_episodes
from benchmarks.common.stages.types import StageStats
from everalgo.clustering import Cluster

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.common.stages.types import StageContext

_log = logging.getLogger(__name__)


def extract_searchable_units(episode: dict[str, Any], episode_facts: list[dict[str, Any]]) -> list[str]:
    """Return the list of strings to index for one episode (fact-level granularity).

    Each atomic fact's ``content`` becomes one BM25 row, plus the episode's ``subject``
    and the first 200 chars of the episode body. That head slice provides additional
    keyword-match signal for BM25 recall — its phrasing differs from atomic facts and
    captures terms that fact extraction may rephrase.

    Deliberately not the episode's ``summary`` field, despite that being the better-formed
    text: the two serve different purposes (``summary`` is a display preview) and swapping
    them would move LoCoMo's recall numbers, which needs its own measured comparison.

    Args:
        episode: Episode dict with ``subject`` and ``episode`` fields.
        episode_facts: List of atomic-fact dicts belonging to this episode, each with a ``content`` field.

    Returns:
        List of searchable text strings (facts + subject + body head slice).

    Raises:
        ValueError: If no atomic facts exist for this episode.
    """
    if not episode_facts:
        raise ValueError(f"No atomic facts for episode id={episode.get('id', 'unknown')}; cannot build search index")

    units: list[str] = [
        f["content"].strip() for f in episode_facts if isinstance(f.get("content"), str) and f["content"].strip()
    ]
    subject = cast("str", episode.get("subject") or "")
    if subject:
        units.append(subject)
    # BM25-only head slice: first 200 chars of episode body as an additional keyword-match row.
    episode_body = cast("str", episode.get("episode") or "")
    if episode_body:
        units.append(episode_body[:200])
    return units


def _group_facts_by_episode(atomic_facts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group atomic facts by their ``episode_id`` field.

    Args:
        atomic_facts: Flat list of atomic-fact dicts from ``atomic_facts_conv_<i>.json``.

    Returns:
        Dict mapping episode_id → list of fact dicts.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for fact in atomic_facts:
        ep_id = str(fact.get("episode_id", ""))
        grouped.setdefault(ep_id, []).append(fact)
    return grouped


def _build_bm25_fact_level(
    episodes: list[dict[str, Any]],
    facts_by_episode: dict[str, list[dict[str, Any]]],
) -> tuple[Any, list[int]] | None:
    """Build a fact-level BM25 corpus + ``fact_to_doc_idx`` parent mapping.

    Each episode is treated as a "document" at its list index. Every searchable unit (each
    atomic fact's ``content`` + episode ``subject``) becomes one BM25 row mapped to that
    episode's index via ``fact_to_doc_idx``.

    Returns ``(BM25Okapi, fact_to_doc_idx)`` or ``None`` if no episode produced any tokenizable unit.
    """
    ensure_nltk()
    stemmer: Any = PorterStemmer()
    stop_words: set[str] = set(cast("list[str]", stopwords.words("english")))  # type: ignore[no-untyped-call]
    fact_corpus: list[list[str]] = []
    fact_to_doc_idx: list[int] = []
    for doc_idx, ep in enumerate(episodes):
        ep_id = str(ep.get("id", doc_idx))
        ep_facts = facts_by_episode.get(ep_id, [])
        for unit in extract_searchable_units(ep, ep_facts):
            tokens = _tokenize(unit, stemmer, stop_words)
            if tokens:
                fact_corpus.append(tokens)
                fact_to_doc_idx.append(doc_idx)
    if not fact_corpus:
        return None
    return BM25Okapi(fact_corpus), fact_to_doc_idx


def _build_emb_index(
    episodes: list[dict[str, Any]],
    facts_by_episode: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build the embedding index from pre-computed embeddings in entity files.

    Each entry contains:
    - ``doc_id``: episode string ID.
    - ``embeddings``: dict with ``atomic_facts`` (list of np.ndarray), ``subject`` (np.ndarray or None),
      and ``episode`` (np.ndarray or None).

    No re-embedding is performed; all vectors come from the entity files.

    Args:
        episodes: List of episode dicts, each with ``embeddings`` containing ``episode`` and ``subject`` vectors.
        facts_by_episode: Atomic facts grouped by episode_id. Each fact has an ``embeddings`` field (list of floats).

    Returns:
        List of embedding index entry dicts.
    """
    emb_index: list[dict[str, Any]] = []
    for ep in episodes:
        ep_id = str(ep.get("id", ""))
        ep_embeddings: dict[str, Any] = ep.get("embeddings") or {}

        entry_embeddings: dict[str, Any] = {}

        # Atomic fact embeddings
        ep_facts = facts_by_episode.get(ep_id, [])
        fact_vecs = [np.array(f["embeddings"], dtype=np.float32) for f in ep_facts if f.get("embeddings")]
        if fact_vecs:
            entry_embeddings["atomic_facts"] = fact_vecs

        # Episode-level embeddings
        ep_vec = ep_embeddings.get("episode")
        if ep_vec is not None:
            entry_embeddings["episode"] = np.array(ep_vec, dtype=np.float32)

        sub_vec = ep_embeddings.get("subject")
        if sub_vec is not None:
            entry_embeddings["subject"] = np.array(sub_vec, dtype=np.float32)

        emb_index.append({"doc_id": ep_id, "embeddings": entry_embeddings})
    return emb_index


def _build_cluster_index(clusters_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Reshape a cluster JSON into ``list[Cluster.model_dump()]`` for Stage 3.

    Aligns the cluster pkl schema with the algo ``Cluster`` type
    (``everalgo.clustering.state.Cluster``): same field names (``id`` / ``count`` /
    ``last_ts`` / ``members`` / ``centroid`` / ``preview``).

    The ``members`` field is populated from ``episode_ids`` (entity-split model)
    to match the episode-based indexing used by the rest of the pipeline.

    Args:
        clusters_data: Parsed content of ``clusters_conv_<i>.json``. Expected to carry a top-level
            ``"clusters"`` list with ``episode_ids`` and standard Cluster fields.

    Returns:
        A list of ``Cluster.model_dump()`` dicts.
    """
    raw_clusters: list[dict[str, Any]] = list(clusters_data.get("clusters") or [])
    result: list[dict[str, Any]] = []
    for c in raw_clusters:
        members = c.get("episode_ids") or c.get("members", [])
        cluster = Cluster.model_validate(
            {
                **c,
                "centroid": np.array(c["centroid"]),
                "members": members,
            }
        )
        result.append(cluster.model_dump())
    return result


def _build_and_write_bm25_index(
    conv_idx: int,
    episodes: list[dict[str, Any]],
    facts_by_episode: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> bool:
    """Build fact-level BM25 corpus and persist it to ``bm25_conv_<i>.pkl``.

    The ``docs`` field in the pickle stores episode dicts (not monolithic memcells),
    keyed by their list position which aligns with ``fact_to_doc_idx``.

    Returns ``False`` if the corpus is degenerate (no tokenizable unit).
    """
    bm25_built = _build_bm25_fact_level(episodes, facts_by_episode)
    if bm25_built is None:
        return False
    bm25_obj, fact_to_doc_idx = bm25_built
    payload = {
        "bm25": bm25_obj,
        "docs": episodes,
        "fact_to_doc_idx": fact_to_doc_idx,
        "index_type": "maxsim",
    }
    out = output_dir / f"bm25_conv_{conv_idx}.pkl"
    with out.open("wb") as fh:
        pickle.dump(payload, fh)
    return True


def _build_and_write_emb_index(
    conv_idx: int,
    episodes: list[dict[str, Any]],
    facts_by_episode: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Build embedding index from pre-computed vectors and persist to ``emb_conv_<i>.pkl``."""
    emb_index = _build_emb_index(episodes, facts_by_episode)
    out = output_dir / f"emb_conv_{conv_idx}.pkl"
    with out.open("wb") as fh:
        pickle.dump(emb_index, fh)


def _process_one_conversation(
    conv_idx: int,
    input_dir: Path,
    output_dir: Path,
    ctx: StageContext,
) -> bool:
    """Build BM25 + embedding + cluster indices for one conversation.

    Reads ``episodes_conv_<i>.json`` and ``atomic_facts_conv_<i>.json`` from the input directory.
    All embeddings are pre-computed in the entity files — no embedding API calls are needed.

    Returns ``True`` on success, ``False`` on degenerate input (no episodes).
    """
    episodes_path = input_dir / f"episodes_conv_{conv_idx}.json"
    facts_path = input_dir / f"atomic_facts_conv_{conv_idx}.json"

    if not episodes_path.exists():
        return False
    episodes = load_episodes(episodes_path)
    if not episodes:
        return False

    if not facts_path.exists():
        raise FileNotFoundError(f"atomic_facts_conv_{conv_idx}.json missing; expected: {facts_path}")
    atomic_facts = load_atomic_facts(facts_path)
    facts_by_episode = _group_facts_by_episode(atomic_facts)

    if not _build_and_write_bm25_index(conv_idx, episodes, facts_by_episode, output_dir):
        return False
    _build_and_write_emb_index(conv_idx, episodes, facts_by_episode, output_dir)

    _build_and_write_cluster_index(conv_idx, output_dir, ctx)

    return True


def _build_and_write_cluster_index(conv_idx: int, output_dir: Path, ctx: StageContext) -> None:
    """Build and persist the cluster index for one conversation.

    Failure propagates to the caller — cluster index errors terminate the stage
    rather than silently falling back to flat hybrid retrieval in Stage 3.
    """
    cluster_path = ctx.input_dir / f"clusters_conv_{conv_idx}.json"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Cluster file missing for conv_{conv_idx}; expected: {cluster_path}. Re-run stage 1.")

    clusters_data = load_clusters(cluster_path)
    cluster_index = _build_cluster_index(clusters_data)
    cluster_out = output_dir / f"cluster_index_conv_{conv_idx}.pkl"
    with cluster_out.open("wb") as fh:
        pickle.dump(cluster_index, fh)


async def run_index_stage(ctx: StageContext) -> StageStats:
    """Stage 2 — build BM25 + Embedding indices per conversation.

    Reads entity-split files (episodes + atomic_facts) from ``ctx.input_dir``.
    All embeddings are pre-computed — this stage performs no embedding API calls.
    """
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage_name="index")
    started = time.monotonic()

    episode_files = sorted(ctx.input_dir.glob("episodes_conv_*.json"))

    from tqdm import tqdm as _tqdm  # Deferred: optional dependency, avoid top-level import

    for ep_file in _tqdm(episode_files, desc="index", unit="conv", dynamic_ncols=True):
        conv_idx = int(ep_file.stem.split("_")[-1])
        try:
            ok = _process_one_conversation(conv_idx, ctx.input_dir, ctx.output_dir, ctx)
        except Exception:
            err_path = ctx.output_dir / f"index_conv_{conv_idx}.error.txt"
            err_path.write_text(traceback.format_exc())
            _log.exception("conv_%d index build failed; full traceback in %s", conv_idx, err_path)
            raise
        if ok:
            stats.success += 1
        else:
            stats.failed += 1

    stats.duration_seconds = time.monotonic() - started
    return stats
