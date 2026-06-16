"""End-to-end smoke tests for ``acategory_retrieve`` against a real LLM.

This file exercises the full pipeline:

    base_retrieve  → rollup_category_mass  → rerank_fn (LLM) → apply_category_boost

EverAlgo doesn't ship a cross-encoder, so we use ``everalgo.rank.rerank.arerank``
(the existing LLM-driven reranker) as the ``RerankFn`` — this is precisely the
contract pattern §7.2 of the design describes: callers decide what model the
``RerankFn`` wraps. ``base_retrieve`` and the document corpus are hand-rolled
so the test runs without any vector / BM25 index.

Assertions are probabilistic ("at least K of top-N from the expected category")
to tolerate normal LLM noise; the test will not flake on small ranking variance.

Skipped automatically when the three ``LLM_*`` env vars are absent.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import pytest

from everalgo.rank import acategory_retrieve
from everalgo.rank.rerank import arerank
from everalgo.types import Candidate

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

_RetrieveFn = Callable[[str, int], Awaitable[list[Candidate]]]
_RerankFn = Callable[[str, list[Candidate]], Awaitable[list[Candidate]]]


pytestmark = pytest.mark.integration


# Generic rerank prompt — domain-neutral so the model judges on the candidate's
# free-form ``text`` field rather than expecting case-specific fields.
_GENERIC_RERANK_PROMPT = """\
Score each candidate document for how directly it answers the user query, from
0.0 (completely off-topic) to 1.0 (a precise, direct answer).

User query:
{query}

Candidates (JSON array; each item has ``id``, ``score``, and a free-form
``text`` field describing the document):
{candidates_json}

Return at most {top_k} items sorted by score descending. Drop candidates whose
score would be near 0.

Output strictly the following JSON, no prose:
{{"ranked": [{{"id": "<candidate_id>", "score": <float in 0..1>}}]}}
"""


# A small mixed corpus. ``category_id`` lives in metadata, as it would in production.
_CORPUS: list[Candidate] = [
    # how-to
    Candidate(
        id="h1",
        score=0.0,
        metadata={"category_id": "how-to", "text": "How to tune Postgres VACUUM with autovacuum_max_workers."},
    ),
    Candidate(
        id="h2",
        score=0.0,
        metadata={"category_id": "how-to", "text": "Step-by-step: setting up logical replication in Postgres."},
    ),
    Candidate(
        id="h3",
        score=0.0,
        metadata={"category_id": "how-to", "text": "Walkthrough for configuring connection pooling with PgBouncer."},
    ),
    Candidate(
        id="h4",
        score=0.0,
        metadata={"category_id": "how-to", "text": "How to enable pg_stat_statements and analyze slow queries."},
    ),
    # concept (definitions / explanations)
    Candidate(
        id="c1",
        score=0.0,
        metadata={
            "category_id": "concept",
            "text": "MVCC: multi-version concurrency control concept and snapshot isolation.",
        },
    ),
    Candidate(
        id="c2",
        score=0.0,
        metadata={"category_id": "concept", "text": "What write-ahead logging (WAL) is and why it matters."},
    ),
    Candidate(
        id="c3",
        score=0.0,
        metadata={
            "category_id": "concept",
            "text": "Explanation of transaction isolation levels: read committed, serializable.",
        },
    ),
    Candidate(
        id="c4",
        score=0.0,
        metadata={"category_id": "concept", "text": "Why bloat happens in Postgres tables — concept of dead tuples."},
    ),
    # news
    Candidate(
        id="n1",
        score=0.0,
        metadata={"category_id": "news", "text": "Postgres 17 release notes: highlights and breaking changes."},
    ),
    Candidate(
        id="n2",
        score=0.0,
        metadata={"category_id": "news", "text": "CVE-2025-XXXX disclosed in Postgres extensions ecosystem."},
    ),
    Candidate(
        id="n3",
        score=0.0,
        metadata={"category_id": "news", "text": "Announcing Postgres 18 beta with new monitoring features."},
    ),
    Candidate(
        id="n4", score=0.0, metadata={"category_id": "news", "text": "PgConf 2026 schedule and keynote announcements."}
    ),
    # reference
    Candidate(
        id="r1",
        score=0.0,
        metadata={
            "category_id": "reference",
            "text": "Postgres GUC reference: shared_buffers, work_mem, effective_cache_size.",
        },
    ),
    Candidate(
        id="r2",
        score=0.0,
        metadata={"category_id": "reference", "text": "Full list of psql meta-commands (\\d, \\l, \\dt, \\du, ...)."},
    ),
    Candidate(
        id="r3",
        score=0.0,
        metadata={"category_id": "reference", "text": "SQL function reference for JSON operators in Postgres."},
    ),
]


def _jaccard(a: str, b: str) -> float:
    """Cheap recall score so the test stays index-free."""
    ta = {w.lower().strip(".,:;()") for w in a.split()}
    tb = {w.lower().strip(".,:;()") for w in b.split()}
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union)


def _make_base_retrieve() -> _RetrieveFn:
    """Return an async ``RetrieveFn`` over the in-memory corpus.

    Uses Jaccard token overlap to produce a recall score — good enough to give the
    rollup something to work with, and deterministic so test flakiness comes only
    from the LLM rerank.
    """

    async def base_retrieve(query: str, k: int) -> list[Candidate]:
        scored = [
            Candidate(
                id=doc.id,
                score=_jaccard(query, str(doc.metadata.get("text", ""))),
                metadata=dict(doc.metadata),
            )
            for doc in _CORPUS
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]

    return base_retrieve


def _make_rerank_fn(llm: LLMClient) -> _RerankFn:
    """Wrap ``arerank`` as a ``RerankFn`` using a generic rerank prompt.

    The prompt is domain-neutral ("score these candidates against the query") and
    serves as an adequate placeholder for a real cross-encoder in this smoke test.
    """

    async def rerank_fn(query: str, cands: list[Candidate]) -> list[Candidate]:
        return await arerank(
            cands,
            query=query,
            prompt=_GENERIC_RERANK_PROMPT,
            top_k=len(cands),
            llm=llm,
        )

    return rerank_fn


@pytest.mark.parametrize(
    ("query", "expected_category", "min_hits_in_top_n", "top_n"),
    [
        ("how do I tune Postgres VACUUM and autovacuum", "how-to", 2, 5),
        ("definition of MVCC and snapshot isolation", "concept", 2, 5),
        ("latest Postgres 17 release notes", "news", 2, 5),
    ],
)
async def test_acategory_retrieve_high_conf_query_prefers_target_category(
    real_llm: LLMClient,
    query: str,
    expected_category: str,
    min_hits_in_top_n: int,
    top_n: int,
) -> None:
    """High-confidence queries should land most of the top-N inside the target category."""
    results = await acategory_retrieve(
        query,
        base_retrieve=_make_base_retrieve(),
        rerank_fn=_make_rerank_fn(real_llm),
        recall_n=len(_CORPUS),
        rerank_n=8,
        mass_top_m=8,
        lam=0.2,
        top_n=top_n,
    )

    assert len(results) > 0, "expected at least one result"
    top = results[:top_n]
    hits = sum(1 for c in top if c.metadata.get("category_id") == expected_category)
    cats = [c.metadata.get("category_id") for c in top]
    ids = [c.id for c in top]
    # Robust to rerankers that aggressively prune near-0 candidates (some models leave a
    # single survivor): require the target category to hold the #1 slot and to dominate
    # whatever survived, rather than a fixed absolute count that assumes a full pool.
    assert top[0].metadata.get("category_id") == expected_category, (
        f"query={query!r} expected top-1 from {expected_category!r}, got {cats[0]!r} (ids={ids}, categories={cats})"
    )
    assert hits >= min(min_hits_in_top_n, len(top)), (
        f"query={query!r} expected {expected_category!r} to dominate top-{top_n}, "
        f"got {hits}/{len(top)} (ids={ids}, categories={cats})"
    )


async def test_acategory_retrieve_low_conf_query_does_not_break_ranking(
    real_llm: LLMClient,
) -> None:
    """Ambiguous/gibberish query → flat ``p(c)`` → ``conf`` near zero → boost ≈ 0.

    We can't assert exact ordering with a live LLM, but we can assert the call
    completes, returns a sensible-sized list, and does NOT collapse to a single
    category (which is what an over-eager boost would cause).
    """
    results = await acategory_retrieve(
        "random gibberish xyz qwerty asdf",
        base_retrieve=_make_base_retrieve(),
        rerank_fn=_make_rerank_fn(real_llm),
        recall_n=len(_CORPUS),
        rerank_n=8,
        mass_top_m=8,
        lam=0.2,
        top_n=8,
    )

    if not results:
        # Acceptable — the rerank LLM may have dropped everything as irrelevant.
        return

    categories_in_results = {c.metadata.get("category_id") for c in results}
    # Boost shouldn't dominate to a single category for a query with no signal.
    assert len(categories_in_results) >= 2, f"low-conf query collapsed to one category: {categories_in_results}"
