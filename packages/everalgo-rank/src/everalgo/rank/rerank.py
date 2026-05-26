"""Rerank module — LLM rerank tool + unified facade pipeline + dispatch."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast

from asgiref.sync import async_to_sync
from pydantic import BaseModel, ConfigDict

from everalgo.llm.types import ChatMessage
from everalgo.types import Candidate, RankInput, RankOutput, ScoredItem

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

__all__ = [
    "DEFAULT_RANK_CONFIG",
    "FusionMode",
    "RankConfig",
    "_basic_arank",
    "arerank",
    "rerank",
]


logger = logging.getLogger(__name__)

FusionMode = Literal["rrf", "lr", "vector_anchored"]
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

    va_saturation_k: float = 5.0  # vector_anchored: saturation scale for BM25 scores
    va_alpha: float = 0.7  # vector_anchored: blend weight (1=vec-only, 0=bm25-only)


DEFAULT_RANK_CONFIG = RankConfig()


async def arerank(
    items: Sequence[Candidate],
    *,
    query: str,
    prompt: str,
    top_k: int,
    llm: LLMClient,
) -> list[Candidate]:
    """LLM-driven rerank.

    Args:
        items: Candidates produced by fusion (+ optional weighting).
        query: User query to rerank against. Substituted into the prompt's
            ``{query}`` placeholder.
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

    candidates_payload = [{"id": item.id, "score": item.score, **item.metadata} for item in items]
    rendered = prompt.format(
        query=query,
        candidates_json=json.dumps(candidates_payload, ensure_ascii=False, default=str),
        top_k=top_k,
    )

    ranked_items = await _call_llm_for_rerank(client, rendered)
    return _apply_rerank_scores(items, ranked_items, top_k)


rerank = async_to_sync(arerank)
"""Sync bridge for non-event-loop contexts (CLI / pytest). Per ADR 010."""


def _extract_json_object(text: str) -> str:
    """Find first balanced brace block in text (brace-balanced parser for nested JSON).

    Used for rerank responses which contain ``ranked: list[{id, score}]`` — a nested structure
    that a flat non-nested regex cannot handle.
    """
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON found: {text[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError(f"Unbalanced JSON: {text[:200]!r}")


async def _call_llm_for_rerank(llm: LLMClient, rendered: str) -> list[dict[str, Any]]:
    """Inner async function decorated with retry; separated so the decorator applies at definition time.

    Uses brace-balanced extractor because RerankResponse is nested (ranked: list[{id, score}]).
    Returns the raw ``ranked`` list as a list of dicts for score application.

    Raises:
        ValueError: If no JSON object found or the ``ranked`` field is absent/invalid.
    """
    response = await llm.chat(messages=[ChatMessage(role="user", content=rendered)])
    text = response.content
    json_str = _extract_json_object(text)
    data = json.loads(json_str)
    ranked = data.get("ranked")
    if not isinstance(ranked, list):
        raise ValueError(f"Missing or invalid 'ranked' field in rerank response: {data!r}")  # noqa: TRY004
    return cast("list[dict[str, Any]]", ranked)


def _apply_rerank_scores(
    items: Sequence[Candidate],
    ranked_items: list[dict[str, Any]],
    top_k: int,
) -> list[Candidate]:
    """Apply LLM-assigned scores to candidates, sort, truncate."""
    by_id = {item.id: item for item in items}
    out: list[Candidate] = []
    for ranked_item in ranked_items:
        item_id = ranked_item.get("id")
        item_score = ranked_item.get("score")
        if not isinstance(item_id, str) or item_id not in by_id:
            continue
        original = by_id[item_id]
        out.append(
            original.model_copy(
                update={
                    "score": float(item_score) if item_score is not None else 0.0,
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


async def _basic_arank(
    rank_input: RankInput,
    *,
    config: RankConfig = DEFAULT_RANK_CONFIG,
    llm: LLMClient | None = None,
    prompt: str | None = None,
    enable_rerank: bool = False,
    rerank_top_k: int | None = None,
) -> RankOutput:
    """Unified pipeline dispatched to by all facade rankers.

    Branches on ``config.fusion_mode``: ``rrf`` and ``lr`` are the two Phase-1 fusion strategies.
    Phase 5 LLM rerank is an optional shared final step.

    ``rerank_top_k`` lets the Phase-5 rerank narrow further than ``rank_input.top_k``:
    fusion produces a wider candidate pool (``top_k``), the LLM rescores all of them, and only the top
    ``rerank_top_k`` items survive. When ``None`` (default), rerank keeps ``rank_input.top_k`` items — the
    historical behaviour.

    Raises:
        KeyError: ``rank_input.memory_type`` not in ``_ALGO_REGISTRY``.
        ValueError: Unsupported ``fusion_mode`` for the facade.
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
    if not sparse and not dense:
        return RankOutput(items=[], metadata={"stage": item_type, "stop_reason": "no_candidates"})

    fused: list[Candidate]
    if not sparse:
        fused = list(dense)
    elif not dense:
        fused = list(sparse)
    elif config.fusion_mode == "lr":
        fused = fusion.lr(dense, sparse)
    elif config.fusion_mode == "vector_anchored":
        fused = fusion.vector_anchored(dense, sparse, saturation_k=config.va_saturation_k, alpha=config.va_alpha)
    else:  # rrf
        fused = fusion.rrf(dense, sparse, k=config.rrf_k)

    fused.sort(key=lambda c: c.score, reverse=True)
    fused = fused[: rank_input.top_k]

    scored = [ScoredItem(id=c.id, score=c.score, item_type=item_type, metadata=dict(c.metadata)) for c in fused]
    meta: dict[str, Any] = {"stage": item_type, "fusion_mode": config.fusion_mode}

    if enable_rerank and scored:
        if llm is None:
            raise ValueError("enable_rerank=True requires llm to be provided")
        with_query = [
            Candidate(
                id=item.id,
                score=item.score,
                metadata={
                    **item.metadata,
                    "item_type": item.item_type,
                    "parent_episode_id": item.parent_episode_id,
                },
            )
            for item in scored
        ]
        effective_rerank_top_k = rerank_top_k if rerank_top_k is not None else rank_input.top_k
        reranked = await arerank(
            with_query,
            query=rank_input.query,
            prompt=prompt or rerank_prompt,
            top_k=effective_rerank_top_k,
            llm=llm,
        )
        by_id = {it.id: it for it in scored}
        scored = [
            by_id[c.id].model_copy(update={"score": c.score, "metadata": {**by_id[c.id].metadata, **c.metadata}})
            for c in reranked
            if c.id in by_id
        ]
        meta = {**meta, "reranked": True, "rerank_top_k": effective_rerank_top_k}

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

from everalgo.rank import fusion  # noqa: E402  (late import breaks the circular dependency with rank/__init__.py)
