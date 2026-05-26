"""aagentic_retrieve — LLM-guided sufficiency check around a base retriever.

Demonstrates the 4 components the caller composes:

1. ``base_retrieve`` (RetrieveFn) — the caller's own search function (can be
   ``ahybrid_retrieve`` partially applied, a vector-DB call, anything).
2. ``rerank_fn`` (RerankFn, optional) — a cross-encoder applied to Round 1
   results before the sufficiency check.
3. ``llm`` (LLMClient) — judges sufficiency + emits multi-query refinement
   when results are insufficient. Here a ``FakeLLMClient`` returns scripted
   JSON; a real caller plugs in an ``OpenAIClient`` / ``AnthropicClient``.
4. The returned ``AgenticDecision`` carries the LLM verdicts the caller
   cannot reconstruct externally (is_sufficient, refined_queries, ...).

This example walks the **happy path**: the sufficiency check returns
``is_sufficient=True`` so Round 2 is skipped. See ``test_agentic.py`` for
multi-round + refined-query / multi-query expansion scenarios.

Run:
    uv run python examples/09_retrieval_agentic.py
"""

from __future__ import annotations

import asyncio
import json

from everalgo.retrieval import RerankFn, RetrieveFn, aagentic_retrieve
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate

# ---------------------------------------------------------------------------
# Caller-side corpus + base retriever + rerank function.
# ---------------------------------------------------------------------------

_CORPUS: list[dict[str, str | float]] = [
    {"id": "doc_a", "text": "Tenacity: async retries with exponential backoff in Python", "score": 0.92},
    {"id": "doc_b", "text": "Rate-limit handling with sliding-window counters", "score": 0.50},
    {"id": "doc_c", "text": "Exponential backoff fundamentals (formulas + jitter)", "score": 0.75},
]


def _make_base_retrieve() -> RetrieveFn:
    """Mock vector retriever — sorts by pre-baked score."""

    async def base_retrieve(query: str, k: int) -> list[Candidate]:
        ranked = sorted(_CORPUS, key=lambda d: float(d["score"]), reverse=True)
        return [
            Candidate(id=str(d["id"]), score=float(d["score"]), source="vector", metadata={"text": d["text"]})
            for d in ranked[:k]
        ]

    return base_retrieve


def _make_rerank() -> RerankFn:
    """Mock cross-encoder rerank — real caller calls Cohere / Voyage / a local model."""

    async def rerank(query: str, candidates: list[Candidate]) -> list[Candidate]:
        # Toy demo: keep order, multiply score by 1.05 to show rerank touched them.
        return [c.model_copy(update={"score": c.score * 1.05}) for c in candidates]

    return rerank


# ---------------------------------------------------------------------------
# Scripted LLM response — judges Round 1 results sufficient, so Round 2 skips.
# Schema mirrors ``SUFFICIENCY_CHECK_PROMPT_EN`` output contract.
# ---------------------------------------------------------------------------

_SUFFICIENCY_JSON_SUFFICIENT = json.dumps(
    {
        "is_sufficient": True,
        "reasoning": "All three documents directly address async retry patterns in Python.",
        "missing_info": [],
        "key_information_found": ["tenacity library", "exponential backoff", "jitter"],
    }
)


async def main() -> None:
    """Run aagentic_retrieve end-to-end and print results + decision."""
    base = _make_base_retrieve()
    rerank = _make_rerank()
    llm = FakeLLMClient(responses=[_SUFFICIENCY_JSON_SUFFICIENT])

    results, decision = await aagentic_retrieve(
        query="how do I retry async calls in Python?",
        base_retrieve=base,
        llm=llm,
        rerank_fn=rerank,
        top_n=3,
        round1_top_n=3,
        round1_rerank_top_n=3,
    )

    print("Top results:")
    for rank, c in enumerate(results, start=1):
        text = c.metadata.get("text", "")
        print(f"  #{rank}  id={c.id!r}  score={c.score:.3f}  text={text!r}")
    print()
    print("AgenticDecision:")
    print(f"  is_multi_round  = {decision.is_multi_round}")
    print(f"  is_sufficient   = {decision.is_sufficient}")
    print(f"  reasoning       = {decision.reasoning!r}")
    print(f"  refined_queries = {decision.refined_queries}")
    print(f"  LLM calls made  = {llm.call_count}")


if __name__ == "__main__":
    asyncio.run(main())
