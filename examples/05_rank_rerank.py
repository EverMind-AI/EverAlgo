"""arerank — LLM-based reranking of recall candidates.

Builds 4 ``Candidate`` objects with initial fusion scores (0.9 / 0.7 / 0.5 / 0.3),
then runs them through ``arerank(query=..., items=candidates, top_k=2, llm=fake)``.
The ``FakeLLMClient`` returns a ``{"ranked": [...]}`` payload that reorders and
re-scores the candidates; only the top-2 survive the ``top_k`` cut.

Run:
    uv run python examples/05_rank_rerank.py
"""

from __future__ import annotations

import asyncio
import json

from everalgo.rank.prompts.en.episodic import EPISODIC_RERANK_PROMPT_EN
from everalgo.rank.rerank import arerank
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate

# ---------------------------------------------------------------------------
# Scripted LLM response — returns a reordered subset; LLM bumps "ep_c" to top.
# ---------------------------------------------------------------------------

_RERANK_JSON = json.dumps(
    {
        "ranked": [
            {"id": "ep_c", "score": 0.96},
            {"id": "ep_a", "score": 0.88},
            {"id": "ep_b", "score": 0.41},
            {"id": "ep_d", "score": 0.12},
        ]
    }
)


def _make_candidates() -> list[Candidate]:
    """Four episodic candidates with descending fusion scores."""
    return [
        Candidate(
            id="ep_a",
            score=0.9,
            metadata={"episode": "Alice asked about retries."},
        ),
        Candidate(
            id="ep_b",
            score=0.7,
            metadata={"episode": "Bob set up a CI pipeline."},
        ),
        Candidate(
            id="ep_c",
            score=0.5,
            metadata={"episode": "Alice explored tenacity library for async back-off."},
        ),
        Candidate(
            id="ep_d",
            score=0.3,
            metadata={"episode": "Team discussed lunch options."},
        ),
    ]


async def main() -> None:
    """Rerank 4 candidates with a scripted LLM response and print the top-2 results."""
    fake = FakeLLMClient(responses=[_RERANK_JSON])
    candidates = _make_candidates()

    reranked: list[Candidate] = await arerank(
        candidates,
        query="Python async retry patterns",
        prompt=EPISODIC_RERANK_PROMPT_EN,
        top_k=2,
        llm=fake,
    )

    print("top_k=2 results (sorted by LLM score, descending):")
    for rank, c in enumerate(reranked, start=1):
        fusion_score = c.metadata.get("fusion_score", "n/a")
        print(f"  #{rank}  id={c.id!r}  llm_score={c.score:.2f}  fusion_score={fusion_score}")


if __name__ == "__main__":
    asyncio.run(main())
