"""Rerank module — LLM rerank tool + unified facade pipeline + dispatch."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from asgiref.sync import async_to_sync
from pydantic import BaseModel, ConfigDict

import everalgo.llm
from everalgo.llm.types import ChatMessage
from everalgo.types import Candidate, RankInput, RankOutput, ScoredItem

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

__all__ = [
    "DEFAULT_RANK_CONFIG",
    "FusionMode",
    "RankConfig",
    "arerank",
    "rerank",
]

logger = logging.getLogger(__name__)

FusionMode = Literal["rrf", "lr", "mrag"]
"""Top-level ranking strategy — parallel to enterprise ``method`` parameter.
"""


class RankConfig(BaseModel):
    """Rank pipeline configuration. Frozen value object."""

    model_config = ConfigDict(frozen=True)

    fusion_mode: FusionMode = "rrf"
    rrf_k: int = 60

    alpha: float = 1.0

    expand_limit: int = 3
    max_convergence_rounds: int = 10


DEFAULT_RANK_CONFIG = RankConfig()


async def arerank(
    items: Sequence[Candidate],
    *,
    prompt: str,
    top_k: int,
    llm: LLMClient | None = None,
) -> list[Candidate]:
    """LLM-driven rerank.

    Args:
        items: Candidates produced by fusion (+ optional weighting).
        prompt: Prompt template; must include ``{query}`` / ``{candidates_json}`` /
            ``{top_k}`` placeholders. Callers typically pass one of the modules
            in ``everalgo.rank.prompts.{en,zh}``.
        top_k: Maximum number of items to return.
        llm: Per-call LLM override; ``None`` resolves through the 3-layer
            fallback (per-call → scoped → default).

    Returns
    -------
        Up to ``top_k`` Candidates, sorted descending by the LLM-assigned score.
        Each result has ``.score`` overwritten with the LLM score; the original
        fusion score is preserved in ``metadata["fusion_score"]``.

    Raises
    ------
        KeyError: ``prompt`` is missing one of the required placeholders.
        LLMError / LLMNotConfiguredError: from ``everalgo.llm.resolve`` + the
            client's ``chat`` call.
    """
    if not items:
        return []

    client = everalgo.llm.resolve(llm)

    query = ""
    for item in items:
        q = item.metadata.get("__rerank_query__")
        if isinstance(q, str):
            query = q
            break

    candidates_payload = [{"id": item.id, "score": item.score, **item.metadata} for item in items]
    rendered = prompt.format(
        query=query,
        candidates_json=json.dumps(candidates_payload, ensure_ascii=False),
        top_k=top_k,
    )

    response = await client.chat(
        messages=[ChatMessage(role="user", content=rendered)],
        response_format={"type": "json_object"},
    )

    return _apply_rerank_scores(items, response.content, top_k)


rerank = async_to_sync(arerank)
"""Sync bridge for non-event-loop contexts (CLI / pytest). Per ADR 010."""


def _apply_rerank_scores(
    items: Sequence[Candidate],
    raw_content: str,
    top_k: int,
) -> list[Candidate]:
    """Parse the LLM's JSON, replace scores, sort, truncate."""

    def _preserve_fusion(item: Candidate) -> Candidate:
        return item.model_copy(update={"metadata": {**item.metadata, "fusion_score": item.score}})

    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        logger.warning("LLM rerank returned non-JSON content; keeping fusion order")
        return [_preserve_fusion(item) for item in list(items)[:top_k]]

    ranked = payload.get("ranked", []) if isinstance(payload, dict) else []
    if not isinstance(ranked, list):
        logger.warning("LLM rerank payload has no 'ranked' list; keeping fusion order")
        return [_preserve_fusion(item) for item in list(items)[:top_k]]

    by_id = {item.id: item for item in items}
    out: list[Candidate] = []
    for entry in ranked:
        if not isinstance(entry, dict):
            continue
        eid = entry.get("id")
        score = entry.get("score")
        if not isinstance(eid, str) or eid not in by_id:
            continue
        if not isinstance(score, (int, float)):
            continue
        original = by_id[eid]
        out.append(
            original.model_copy(
                update={
                    "score": float(score),
                    "metadata": {**original.metadata, "fusion_score": original.score},
                }
            )
        )

    out.sort(key=lambda c: c.score, reverse=True)
    return out[:top_k]


type _AsyncRanker = Callable[..., Awaitable[RankOutput]]


class _RankerSpec(NamedTuple):
    """Per-facade configuration entry stored in ``_ALGO_REGISTRY``.

    Replaces the previous one-callable-per-memory_type registry. Now each
    entry carries:

    - ``arank`` — dispatch target for ``rank.arank(rank_input)``.
    - ``modes`` — fusion modes the facade supports; ``()`` for facades that
      do not use ``fusion_mode`` (profile).
    - ``rerank_prompt`` — default rerank prompt; ``""`` if the facade has no
      rerank stage.
    - ``item_type`` — label stamped on emitted ScoredItems when running the
      non-mrag (single-output) path. The mrag path always emits mixed
      ``"episode"`` + ``"atomic_fact"`` items regardless of this field.
    """

    arank: _AsyncRanker
    modes: tuple[FusionMode, ...]
    rerank_prompt: str
    item_type: Literal["case", "skill", "episode", "profile"]


_ALGO_REGISTRY: dict[str, _RankerSpec] = {}


async def _basic_arank(
    rank_input: RankInput,
    *,
    config: RankConfig = DEFAULT_RANK_CONFIG,
    llm: LLMClient | None = None,
    prompt: str | None = None,
    enable_rerank: bool = False,
) -> RankOutput:
    """Unified pipeline shared by case / skill / episodic facades.

    Looks up ``_ALGO_REGISTRY[rank_input.memory_type]`` for the
    facade-specific config (allowed modes, rerank prompt, item_type label),
    then branches on ``config.fusion_mode``:

    - ``"rrf"`` / ``"lr"`` — Phase 1 fusion only; emit single-``item_type``
      ScoredItems.
    - ``"mrag"`` — Full enterprise MRAG via ``fusion.expand`` (Phase 1 + 2-4);
      emit mixed ``"episode"`` + ``"atomic_fact"`` ScoredItems.

    Phase 5 LLM rerank is the shared final step in either branch.

    Aligned with enterprise / opensource现状: no business-field weighting,
    business metadata is pass-through (see facade docstrings).

    Raises
    ------
        KeyError: ``rank_input.memory_type`` not in ``_ALGO_REGISTRY``.
        ValueError: ``config.fusion_mode`` not in the facade's allowed modes.
    """
    spec = _ALGO_REGISTRY.get(rank_input.memory_type)
    if spec is None:
        raise KeyError(f"No ranker for memory_type={rank_input.memory_type!r}; registered: {sorted(_ALGO_REGISTRY)}")
    if config.fusion_mode not in spec.modes:
        raise ValueError(
            f"fusion_mode={config.fusion_mode!r} is not supported by the "
            f"{rank_input.memory_type!r} facade; allowed: {list(spec.modes)}"
        )

    item_type = spec.item_type
    rerank_prompt = spec.rerank_prompt
    sparse, dense = rank_input.sparse_candidates, rank_input.dense_candidates

    scored: list[ScoredItem]
    if config.fusion_mode == "mrag":
        episodes, facts, meta = fusion.expand(
            sparse,
            dense,
            rank_input.episode_to_facts,
            response_top_k=rank_input.top_k,
            config=config,
        )
        scored = [
            ScoredItem(id=ep.id, score=ep.score, item_type="episode", metadata=dict(ep.metadata)) for ep in episodes
        ]
        scored.extend(
            ScoredItem(
                id=f.id,
                score=f.score,
                item_type="atomic_fact",
                parent_episode_id=f.parent_episode_id,
                metadata=dict(f.metadata),
            )
            for f in facts
        )
        scored.sort(key=lambda it: it.score, reverse=True)
        meta = {"stage": item_type, "fusion_mode": "mrag", **meta}
    else:
        if not sparse and not dense:
            return RankOutput(items=[], metadata={"stage": item_type, "stop_reason": "no_candidates"})

        fused: list[Candidate]
        if not sparse:
            fused = list(dense)
        elif not dense:
            fused = list(sparse)
        elif config.fusion_mode == "lr":
            fused = fusion.lr(dense, sparse)
        else:  # rrf
            fused = fusion.rrf(dense, sparse, k=config.rrf_k)

        fused.sort(key=lambda c: c.score, reverse=True)
        fused = fused[: rank_input.top_k]

        scored = [ScoredItem(id=c.id, score=c.score, item_type=item_type, metadata=dict(c.metadata)) for c in fused]
        meta = {"stage": item_type, "fusion_mode": config.fusion_mode}

    if enable_rerank and scored:
        with_query = [
            Candidate(
                id=item.id,
                score=item.score,
                metadata={
                    **item.metadata,
                    "__rerank_query__": rank_input.query,
                    "item_type": item.item_type,
                    "parent_episode_id": item.parent_episode_id,
                },
            )
            for item in scored
        ]
        reranked = await arerank(with_query, prompt=prompt or rerank_prompt, top_k=rank_input.top_k, llm=llm)
        by_id = {it.id: it for it in scored}
        scored = [
            by_id[c.id].model_copy(update={"score": c.score, "metadata": {**by_id[c.id].metadata, **c.metadata}})
            for c in reranked
            if c.id in by_id
        ]
        meta = {**meta, "reranked": True}

    return RankOutput(items=scored, metadata=meta)


async def _arank(rank_input: RankInput, **kwargs: Any) -> RankOutput:
    """Async dispatch by ``memory_type`` — re-exported as ``arank`` from the package root.

    Private here (``_arank``) because the file's external public surface is
    the LLM rerank tool (``arerank`` / ``rerank``); the package re-exports
    this function as the top-level ``everalgo.rank.arank``.

    Raises
    ------
        KeyError: ``rank_input.memory_type`` is not in ``_ALGO_REGISTRY``.
    """
    spec = _ALGO_REGISTRY.get(rank_input.memory_type)
    if spec is None:
        raise KeyError(f"No ranker for memory_type={rank_input.memory_type!r}; registered: {sorted(_ALGO_REGISTRY)}")
    return await spec.arank(rank_input, **kwargs)


_rank = async_to_sync(_arank)
"""Sync bridge over ``_arank`` — re-exported as ``rank`` from the package root.

Only safe outside an event loop (CLI scripts, plain unit tests). For
FastAPI / asyncio code, ``await arank(...)`` instead.
"""

from everalgo.rank import fusion  # noqa: E402  (late import breaks the circular dependency with fusion)
