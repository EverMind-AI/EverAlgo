"""Agentic retrieval wrapper — LLM-guided sufficiency + multi/refined query expansion.

Wraps any caller-supplied ``base_retrieve: RetrieveFn`` with a sufficiency check
and optional Round 2 query expansion.

Control flow:
    1. Round 1: ``await base_retrieve(query, round1_top_n)``
    2. Optional rerank: ``await rerank_fn(query, round1)`` truncated to ``round1_rerank_top_n``
    3. Sufficiency check: LLM call (structured outputs) with ``SufficiencyCheckResponse``
    4. If sufficient: return ``reranked[:top_n]`` + ``AgenticDecision(is_multi_round=False, ...)``
    5. Round 2 (multi_query): LLM generates complementary queries; parallel round2_retrieve (or base_retrieve if unset) + RRF
    6. Round 2 (refined_query): LLM generates single refined query; single round2_retrieve (or base_retrieve if unset)
    7. Merge: Round 1 + Round 2 dedup by id, optional final rerank, truncate to ``top_n``
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from asgiref.sync import async_to_sync

from everalgo.llm.types import ChatMessage
from everalgo.rank.fusion import rrf
from everalgo.rank.prompts.en.multi_query import MULTI_QUERY_PROMPT_EN
from everalgo.rank.prompts.en.refined_query import REFINED_QUERY_PROMPT_EN
from everalgo.rank.prompts.en.sufficiency import SUFFICIENCY_CHECK_PROMPT_EN
from everalgo.rank.protocols import AgenticDecision, RerankFn, RetrieveFn

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everalgo.llm.protocols import LLMClient
    from everalgo.types import Candidate

__all__ = [
    "MultiQueryResponse",
    "RefinedQueryResponse",
    "SufficiencyCheckResponse",
    "aagentic_retrieve",
    "agentic_retrieve",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SufficiencyCheckResponse:
    """Parsed sufficiency check result."""

    is_sufficient: bool
    reasoning: str
    key_information_found: list[str]
    missing_information: list[str]


@dataclass(frozen=True)
class MultiQueryResponse:
    """Parsed multi-query generation result."""

    queries: list[str]
    reasoning: str


@dataclass(frozen=True)
class RefinedQueryResponse:
    """Parsed refined query generation result."""

    refined_query: str
    reasoning: str


async def aagentic_retrieve(
    query: str,
    *,
    base_retrieve: RetrieveFn,
    llm: LLMClient,
    rerank_fn: RerankFn | None = None,
    round2_retrieve: RetrieveFn | None = None,
    round2_cap: int | None = None,
    top_n: int = 20,
    round1_top_n: int = 20,
    round1_rerank_top_n: int = 10,
    refinement_strategy: Literal["multi_query", "refined_query"] = "multi_query",
    multi_query_count: int = 3,
    rrf_k: int = 60,
    sufficiency_prompt: str | None = None,
    multi_query_prompt: str | None = None,
    refined_query_prompt: str | None = None,
) -> tuple[list[Candidate], AgenticDecision]:
    """LLM-guided agentic retrieval wrapper.

    Wraps ``base_retrieve`` with a sufficiency check + optional Round 2 expansion.
    The caller composes hybrid / single-route search as ``base_retrieve``;
    this function only adds the LLM-guided sufficiency-then-expand loop.

    Args:
        query: User query string.
        base_retrieve: ``(query, top_k) -> list[Candidate]`` — async callable the caller provides.
        llm: LLM client for structured outputs calls.
        rerank_fn: Optional ``(query, candidates) -> list[Candidate]`` cross-encoder. Applied to
            Round 1 before the sufficiency check; result is truncated to ``round1_rerank_top_n``.
        round2_retrieve: Optional separate ``RetrieveFn`` for Round 2 (multi-query / refined-query
            sub-retrievals). When ``None``, falls back to ``base_retrieve``. Use this when R1 and R2
            should target different scopes (e.g. R1 cluster-scoped, R2 full corpus).
        round2_cap: Optional cap on total merged R1+R2-unique candidates before final rerank.
            When set, R2-unique is truncated so ``len(R1) + len(r2_unique_kept) <= round2_cap``.
            ``None`` (default) keeps every R2 item not already in R1.
        top_n: Maximum results to return.
        round1_top_n: Top-k passed to ``base_retrieve`` in Round 1 (and per Round 2 sub-query).
        round1_rerank_top_n: How many results to forward to the LLM sufficiency check (after rerank).
        refinement_strategy: ``"multi_query"`` generates parallel sub-queries; ``"refined_query"``
            generates a single refined query and runs one more ``base_retrieve``.
        multi_query_count: Maximum number of sub-queries to keep (``multi_query`` strategy only).
        rrf_k: RRF smoothing constant for fusing Round 2 sub-query results (``multi_query`` only).
        sufficiency_prompt: Override for ``SUFFICIENCY_CHECK_PROMPT_EN``. Must contain
            ``{query}`` and ``{retrieved_docs}`` placeholders.
        multi_query_prompt: Override for ``MULTI_QUERY_PROMPT_EN``. Must contain
            ``{original_query}``, ``{retrieved_docs}``, ``{missing_info}``, ``{key_info}`` placeholders.
        refined_query_prompt: Override for ``REFINED_QUERY_PROMPT_EN``. Must contain
            ``{original_query}``, ``{retrieved_docs}``, ``{missing_info}`` placeholders.

    Returns:
        ``(candidates, decision)`` where ``decision`` carries the LLM verdicts the caller
        cannot reconstruct externally (is_sufficient, refined_queries, query_strategy, reasoning).
        Token / latency / trace metrics stay caller-side — wrap ``base_retrieve`` / ``llm`` for accounting.

    Raises:
        ValueError: If the LLM returns no parsed structured output (fail-loud per algo responsibility split).
    """
    # ── Round 1 ──────────────────────────────────────────────────────────────
    round1 = await base_retrieve(query, round1_top_n)
    if not round1:
        return [], AgenticDecision(is_multi_round=False)

    # ── Round 1 optional rerank ───────────────────────────────────────────────
    if rerank_fn is not None:
        reranked = (await rerank_fn(query, round1))[:round1_rerank_top_n]
    else:
        reranked = round1[:round1_rerank_top_n]

    # ── Sufficiency check ─────────────────────────────────────────────────────
    sufficiency = await _check_sufficiency(
        query=query,
        candidates=reranked,
        llm=llm,
        prompt=sufficiency_prompt or SUFFICIENCY_CHECK_PROMPT_EN,
    )

    if sufficiency.is_sufficient:
        # Sufficient branch returns ``reranked[:top_n]`` (cross-encoder-ordered),
        # NOT ``round1[:top_n]`` (RRF-ordered). The cross-encoder ordering is what
        # downstream LLM sees, so this affects multi-hop answer extraction.
        return reranked[:top_n], AgenticDecision(
            is_multi_round=False,
            is_sufficient=True,
            reasoning=sufficiency.reasoning,
            key_information_found=list(sufficiency.key_information_found),
            missing_info=list(sufficiency.missing_information),
        )

    # ── Round 2 ───────────────────────────────────────────────────────────────
    if refinement_strategy == "multi_query":
        return await _multi_query_round2(
            query=query,
            reranked=reranked,
            sufficiency=sufficiency,
            llm=llm,
            base_retrieve=base_retrieve,
            round2_retrieve=round2_retrieve,
            round2_cap=round2_cap,
            rerank_fn=rerank_fn,
            top_n=top_n,
            round1_top_n=round1_top_n,
            multi_query_count=multi_query_count,
            rrf_k=rrf_k,
            prompt=multi_query_prompt or MULTI_QUERY_PROMPT_EN,
        )
    return await _refined_query_round2(
        query=query,
        reranked=reranked,
        sufficiency=sufficiency,
        llm=llm,
        base_retrieve=base_retrieve,
        round2_retrieve=round2_retrieve,
        round2_cap=round2_cap,
        rerank_fn=rerank_fn,
        top_n=top_n,
        round1_top_n=round1_top_n,
        prompt=refined_query_prompt or REFINED_QUERY_PROMPT_EN,
    )


agentic_retrieve = async_to_sync(aagentic_retrieve)
"""Sync bridge for non-event-loop contexts (CLI / pytest). Per ADR 010."""


async def _multi_query_round2(
    *,
    query: str,
    reranked: list[Candidate],
    sufficiency: SufficiencyCheckResponse,
    llm: LLMClient,
    base_retrieve: RetrieveFn,
    round2_retrieve: RetrieveFn | None,
    round2_cap: int | None,
    rerank_fn: RerankFn | None,
    top_n: int,
    round1_top_n: int,
    multi_query_count: int,
    rrf_k: int,
    prompt: str,
) -> tuple[list[Candidate], AgenticDecision]:
    """Generate complementary queries via LLM, run parallel Round 2, RRF-fuse, and merge with reranked."""
    multi = await _generate_multi_queries(
        original_query=query,
        candidates=reranked,
        missing_info=list(sufficiency.missing_information),
        key_info=list(sufficiency.key_information_found),
        llm=llm,
        prompt=prompt,
        count=multi_query_count,
    )
    r2 = round2_retrieve or base_retrieve
    round2_lists = await asyncio.gather(*(r2(q, round1_top_n) for q in multi.queries))
    if len(round2_lists) == 1:
        fused: list[Candidate] = list(round2_lists[0])
    else:
        fused = rrf(*round2_lists, k=rrf_k)
    if round2_cap is not None:
        fused = fused[:round2_cap]
    return await _merge_truncate_rerank(
        query=query,
        reranked=reranked,
        round2=fused,
        rerank_fn=rerank_fn,
        top_n=top_n,
        round2_cap=round2_cap,
        sufficiency=sufficiency,
        refined_queries=list(multi.queries),
        query_strategy="multi_query",
    )


async def _refined_query_round2(
    *,
    query: str,
    reranked: list[Candidate],
    sufficiency: SufficiencyCheckResponse,
    llm: LLMClient,
    base_retrieve: RetrieveFn,
    round2_retrieve: RetrieveFn | None,
    round2_cap: int | None,
    rerank_fn: RerankFn | None,
    top_n: int,
    round1_top_n: int,
    prompt: str,
) -> tuple[list[Candidate], AgenticDecision]:
    """Generate a single refined query via LLM, run Round 2, and merge with reranked."""
    refined = await _generate_refined_query(
        original_query=query,
        candidates=reranked,
        missing_info=list(sufficiency.missing_information),
        llm=llm,
        prompt=prompt,
    )
    r2 = round2_retrieve or base_retrieve
    round2 = await r2(refined.refined_query, round1_top_n)
    return await _merge_truncate_rerank(
        query=query,
        reranked=reranked,
        round2=round2,
        rerank_fn=rerank_fn,
        top_n=top_n,
        round2_cap=round2_cap,
        sufficiency=sufficiency,
        refined_queries=[refined.refined_query],
        query_strategy="refined_query",
    )


async def _merge_truncate_rerank(
    *,
    query: str,
    reranked: list[Candidate],
    round2: Sequence[Candidate],
    rerank_fn: RerankFn | None,
    top_n: int,
    round2_cap: int | None,
    sufficiency: SufficiencyCheckResponse,
    refined_queries: list[str],
    query_strategy: Literal["multi_query", "refined_query"],
) -> tuple[list[Candidate], AgenticDecision]:
    """Dedup Round 2 against the reranked Round 1 set by id; optionally cap; final rerank; truncate to top_n.

    The merge basis is the reranked top-N (cross-encoder-ordered), NOT the raw RRF round-1 set.

    When ``round2_cap`` is set, R2-unique items are kept until ``len(reranked) + kept <= cap``.
    When ``round2_cap is None`` (default), all R2-unique items are kept.
    """
    seen = {c.id for c in reranked}
    r2_unique = [c for c in round2 if c.id not in seen]
    if round2_cap is not None:
        keep = max(0, round2_cap - len(reranked))
        r2_unique = r2_unique[:keep]
    merged = list(reranked) + r2_unique
    if rerank_fn is not None:
        merged = await rerank_fn(query, merged)
    final = merged[:top_n]
    decision = AgenticDecision(
        is_multi_round=True,
        is_sufficient=False,
        reasoning=sufficiency.reasoning,
        key_information_found=list(sufficiency.key_information_found),
        missing_info=list(sufficiency.missing_information),
        refined_queries=refined_queries,
        query_strategy=query_strategy,
    )
    return final, decision


def _extract_outermost_json_object(text: str) -> str:
    """Extract the outermost ``{...}`` block from ``text`` via greedy slice (``find("{")..rfind("}")+1``).

    More forgiving than a non-greedy ``{[^{}]*}`` regex: tolerates nested objects
    (prompt asks for flat JSON but LLMs occasionally inline e.g. ``{"key": {"inner": ...}}``)
    and surrounding prose. Raises ``ValueError`` when no ``{...}`` is found.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]!r}")
    return text[start : end + 1]


async def _call_llm_for_sufficiency(llm: LLMClient, rendered: str) -> SufficiencyCheckResponse:
    """Call the LLM with the rendered sufficiency prompt and parse the JSON reply.

    Uses greedy outermost-``{...}`` slice for JSON extraction. Retry / fallback is the caller's
    responsibility — this function fails loud on parse errors.
    """
    response = await llm.chat(messages=[ChatMessage(role="user", content=rendered)], temperature=0.0)
    data = json.loads(_extract_outermost_json_object(response.content))
    return SufficiencyCheckResponse(
        is_sufficient=bool(data.get("is_sufficient", False)),
        reasoning=str(data.get("reasoning", "")),
        key_information_found=list(data.get("key_information_found") or []),
        missing_information=list(data.get("missing_information") or []),
    )


async def _call_llm_for_multi_queries(llm: LLMClient, rendered: str) -> MultiQueryResponse:
    """Call the LLM with the rendered multi-query prompt and parse the JSON reply.

    Uses greedy outermost-``{...}`` slice for JSON extraction. Retry / fallback is the caller's
    responsibility — this function fails loud on parse errors.
    """
    response = await llm.chat(messages=[ChatMessage(role="user", content=rendered)], temperature=0.4)
    data = json.loads(_extract_outermost_json_object(response.content))
    return MultiQueryResponse(
        queries=list(data.get("queries") or []),
        reasoning=str(data.get("reasoning", "")),
    )


async def _call_llm_for_refined_query(
    llm: LLMClient,
    rendered: str,
    original_query: str,
) -> RefinedQueryResponse:
    """Call the LLM with the rendered refined-query prompt and parse the plain-text reply.

    Strips known prefixes (``Refined Query:`` / ``Output:`` / ``Answer:`` / ``Query:``), validates
    length 5-300, falls back to ``original_query`` when the LLM returns garbage or echoes the original
    (case-insensitive).
    """
    response = await llm.chat(messages=[ChatMessage(role="user", content=rendered)], temperature=0.3)
    refined = response.content.strip()
    for prefix in ("Refined Query:", "Output:", "Answer:", "Query:"):
        if refined.startswith(prefix):
            refined = refined[len(prefix) :].strip()
    if not (5 <= len(refined) <= 300) or refined.lower() == original_query.lower():
        refined = original_query
    return RefinedQueryResponse(refined_query=refined, reasoning="")


async def _check_sufficiency(
    *,
    query: str,
    candidates: Sequence[Candidate],
    llm: LLMClient,
    prompt: str,
) -> SufficiencyCheckResponse:
    """LLM call: are these candidates sufficient to answer the query?

    Raises:
        ValueError: If the LLM fails to return a parseable JSON object (fail-loud; no retry).
    """
    rendered = prompt.format(query=query, retrieved_docs=_format_docs(candidates))
    return await _call_llm_for_sufficiency(llm, rendered)


async def _generate_multi_queries(
    *,
    original_query: str,
    candidates: Sequence[Candidate],
    missing_info: list[str],
    key_info: list[str],
    llm: LLMClient,
    prompt: str,
    count: int,
) -> MultiQueryResponse:
    """LLM call: produce complementary queries focused on missing_info.

    Raises:
        ValueError: If the LLM fails to return a parseable JSON object (fail-loud; no retry).
    """
    rendered = prompt.format(
        original_query=original_query,
        retrieved_docs=_format_docs(candidates),
        missing_info=", ".join(missing_info) if missing_info else "N/A",
        key_info=", ".join(key_info) if key_info else "N/A",
    )
    result = await _call_llm_for_multi_queries(llm, rendered)
    validated = [q for q in result.queries if 5 <= len(q) <= 300 and q.lower() != original_query.lower()]
    if count > 0:
        validated = validated[:count]
    return MultiQueryResponse(queries=validated, reasoning=result.reasoning)


async def _generate_refined_query(
    *,
    original_query: str,
    candidates: Sequence[Candidate],
    missing_info: list[str],
    llm: LLMClient,
    prompt: str,
) -> RefinedQueryResponse:
    """LLM call: produce a single refined query targeting missing_info.

    Issues one LLM call with the rendered prompt. The parser strips known plain-text prefixes
    (``Refined Query:`` / ``Output:`` / ``Answer:`` / ``Query:``), validates length 5-300, and
    falls back to ``original_query`` when the LLM output is invalid or echoes the original.
    """
    rendered = prompt.format(
        original_query=original_query,
        retrieved_docs=_format_docs(candidates),
        missing_info=", ".join(missing_info) if missing_info else "N/A",
    )
    return await _call_llm_for_refined_query(llm, rendered, original_query=original_query)


def _format_docs(candidates: Sequence[Candidate]) -> str:
    """Format candidates as multi-line blocks for LLM prompt rendering.

    Four lines per doc (``Document {i}:`` / Title / Date / Content). Body truncates at 500 chars.
    ``Date`` renders the doc's ms-epoch ``timestamp`` as ISO ``YYYY-MM-DDTHH:MM:SSZ`` to align
    with the ``Date`` field referenced by sufficiency/multi-query prompts.

    Raises:
        ValueError: If any candidate lacks ``episode.content`` — fail-loud, no fallback.
    """
    if not candidates:
        return "No retrieval results"
    lines: list[str] = []
    for i, c in enumerate(candidates, 1):
        episode_raw = c.metadata.get("episode")
        if not isinstance(episode_raw, dict):
            raise TypeError(f"Candidate {c.id} has no episode dict in metadata")
        episode_dict: dict[str, Any] = cast("dict[str, Any]", episode_raw)
        subject = str(episode_dict.get("subject") or "N/A")
        body = str(episode_dict.get("content") or "")
        if not body:
            raise ValueError(f"Candidate {c.id} has no episode.content")
        if len(body) > 500:
            body = body[:500] + "..."
        timestamp_str = _format_doc_timestamp(c.metadata.get("timestamp"))
        lines.append(f"Document {i}:\n  Title: {subject}\n  Date: {timestamp_str}\n  Content: {body}\n")
    return "\n".join(lines)


def _format_doc_timestamp(raw: Any) -> str:
    """Render fork's stage-1 ms epoch as ISO ``YYYY-MM-DDTHH:MM:SSZ``; ``N/A`` on missing or invalid."""
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw) or raw <= 0:
        return "N/A"
    return datetime.fromtimestamp(raw / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
