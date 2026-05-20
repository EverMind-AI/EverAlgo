"""Rerank module — LLM rerank tool + unified facade pipeline + dispatch."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

from asgiref.sync import async_to_sync
from pydantic import BaseModel, ConfigDict, Field

from everalgo.llm.types import ChatMessage
from everalgo.rank.fusion import AgenticConfig  # noqa: TC001  (runtime import required by pydantic field type)
from everalgo.types import Candidate, RankInput, RankOutput, ScoredItem

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient
    from everalgo.rank.fusion import RerankFn, RetrieveFn

__all__ = [
    "DEFAULT_RANK_CONFIG",
    "FusionMode",
    "RankConfig",
    "arerank",
    "rerank",
]


class RankedItem(BaseModel):
    """Single ranked candidate from rerank LLM."""

    id: str
    score: float


class RerankResponse(BaseModel):
    """LLM rerank output schema."""

    ranked: list[RankedItem] = Field(..., description="Ranked candidates with scores")


logger = logging.getLogger(__name__)

FusionMode = Literal["rrf", "lr", "mrag", "agentic"]
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

    agentic_config: AgenticConfig | None = None


DEFAULT_RANK_CONFIG = RankConfig()


async def arerank(
    items: Sequence[Candidate],
    *,
    prompt: str,
    top_k: int,
    llm: LLMClient,
) -> list[Candidate]:
    """LLM-driven rerank.

    Args:
        items: Candidates produced by fusion (+ optional weighting).
        prompt: Prompt template; must include ``{query}`` / ``{candidates_json}`` /
            ``{top_k}`` placeholders. Callers typically pass one of the modules
            in ``everalgo.rank.prompts.{en,zh}``.
        top_k: Maximum number of items to return.
        llm: LLM client (required — bound at the ranker instance level).

    Returns:
        Up to ``top_k`` Candidates, sorted descending by the LLM-assigned score.
        Each result has ``.score`` overwritten with the LLM score; the original
        fusion score is preserved in ``metadata["fusion_score"]``.

    Raises:
        KeyError: ``prompt`` is missing one of the required placeholders.
        LLMError: from the client's ``chat`` call.
    """
    if not items:
        return []

    client = llm

    query = ""
    for item in items:
        q = item.metadata.get("__rerank_query__")
        if isinstance(q, str):
            query = q
            break

    candidates_payload = [{"id": item.id, "score": item.score, **item.metadata} for item in items]
    rendered = prompt.format(
        query=query,
        candidates_json=json.dumps(candidates_payload, ensure_ascii=False, default=str),
        top_k=top_k,
    )

    response = await client.chat(
        messages=[ChatMessage(role="user", content=rendered)],
        response_format=RerankResponse,
    )

    parsed = cast("RerankResponse | None", response.parsed)
    if parsed is None:
        raise ValueError("LLM returned no parsed structured output")
    return _apply_rerank_scores(items, parsed, top_k)


rerank = async_to_sync(arerank)
"""Sync bridge for non-event-loop contexts (CLI / pytest). Per ADR 010."""


def _apply_rerank_scores(
    items: Sequence[Candidate],
    response: RerankResponse,
    top_k: int,
) -> list[Candidate]:
    """Apply LLM-assigned scores to candidates, sort, truncate."""
    by_id = {item.id: item for item in items}
    out: list[Candidate] = []
    for ranked_item in response.ranked:
        if ranked_item.id not in by_id:
            continue
        original = by_id[ranked_item.id]
        out.append(
            original.model_copy(
                update={
                    "score": ranked_item.score,
                    "metadata": {**original.metadata, "fusion_score": original.score},
                }
            )
        )

    out.sort(key=lambda c: c.score, reverse=True)
    return out[:top_k]


type _AsyncRanker = Callable[..., Awaitable[RankOutput]]


class _RankerSpec(NamedTuple):
    """Per-facade config entry in ``_ALGO_REGISTRY`` — arank target, supported modes, rerank prompt, item label."""

    arank: _AsyncRanker
    modes: tuple[FusionMode, ...]
    rerank_prompt: str
    item_type: Literal["case", "skill", "episode", "profile"]


_ALGO_REGISTRY: dict[str, _RankerSpec] = {}


async def _basic_arank(  # noqa: C901  # pyright: ignore[reportUnusedFunction]
    rank_input: RankInput,
    *,
    config: RankConfig = DEFAULT_RANK_CONFIG,
    llm: LLMClient | None = None,
    prompt: str | None = None,
    enable_rerank: bool = False,
    retrieve_fn: RetrieveFn | None = None,
    rerank_fn: RerankFn | None = None,
) -> RankOutput:
    """Unified pipeline dispatched to by all facade rankers.

    Branches on ``config.fusion_mode``: ``rrf``/``lr`` run Phase 1 only; ``mrag`` adds Phase 2-4 expansion;
    ``agentic`` runs LLM-guided multi-round. Phase 5 LLM rerank is an optional shared final step.

    Raises:
        KeyError: ``rank_input.memory_type`` not in ``_ALGO_REGISTRY``.
        ValueError: Unsupported ``fusion_mode`` for the facade, or ``agentic`` without ``rerank_fn`` / ``llm``.
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
    elif config.fusion_mode == "agentic":
        if rerank_fn is None:
            raise ValueError("fusion_mode='agentic' requires a `rerank_fn` callback (cross-encoder)")
        if not sparse and not dense:
            return RankOutput(items=[], metadata={"stage": item_type, "stop_reason": "no_candidates"})
        if llm is None:
            raise ValueError("fusion_mode='agentic' requires llm to be provided")
        agentic_cfg = config.agentic_config or fusion.DEFAULT_AGENTIC_CONFIG
        reranked = await fusion.aagentic_rank(
            rank_input.query,
            sparse,
            dense,
            rerank=rerank_fn,
            retrieve=retrieve_fn,
            top_k=rank_input.top_k,
            llm=llm,
            config=agentic_cfg,
        )
        scored = [ScoredItem(id=c.id, score=c.score, item_type=item_type, metadata=dict(c.metadata)) for c in reranked]
        meta = {"stage": item_type, "fusion_mode": "agentic", "round2": retrieve_fn is not None}
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
        if llm is None:
            raise ValueError("enable_rerank=True requires llm to be provided")
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
    """Dispatch by ``memory_type`` to the registered facade ranker.

    Raises:
        KeyError: ``rank_input.memory_type`` not in ``_ALGO_REGISTRY``.
    """
    spec = _ALGO_REGISTRY.get(rank_input.memory_type)
    if spec is None:
        raise KeyError(f"No ranker for memory_type={rank_input.memory_type!r}; registered: {sorted(_ALGO_REGISTRY)}")
    return await spec.arank(rank_input, **kwargs)


_rank = async_to_sync(_arank)
"""Sync bridge over ``_arank``; re-exported as ``rank`` from the package root. Only safe outside an event loop."""

from everalgo.rank import fusion  # noqa: E402  (late import breaks the circular dependency with fusion)
