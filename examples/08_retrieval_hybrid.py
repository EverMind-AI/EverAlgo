"""ahybrid_retrieve — dual-route RRF fusion (dense + sparse).

Demonstrates the caller-vs-algo boundary that defines ``everalgo-retrieval``:
the caller binds storage / embeddings / index inside two ``RetrieveFn``
callables; the algo only sees the abstract ``(query, k) -> list[Candidate]``
surface and fuses the two ranked lists via Reciprocal Rank Fusion.

Run:
    uv run python examples/08_retrieval_hybrid.py
"""

from __future__ import annotations

import asyncio

from everalgo.retrieval import RetrieveFn, ahybrid_retrieve
from everalgo.types import Candidate

# ---------------------------------------------------------------------------
# Caller-side: a tiny in-memory corpus + two mock indexes.
#
# A real caller would bind a vector DB (Milvus / pgvector / ...) inside the
# dense ``RetrieveFn`` and a sparse index (Elasticsearch / tantivy / BM25)
# inside the sparse ``RetrieveFn``. The algo never sees the storage layer.
# ---------------------------------------------------------------------------

_CORPUS: list[dict[str, str | float]] = [
    {
        "id": "doc_a",
        "text": "Tenacity: async retries with exponential backoff in Python",
        "dense": 0.92,
        "sparse": 0.45,
    },
    {"id": "doc_b", "text": "Setting up GitHub Actions CI for Python monorepos", "dense": 0.30, "sparse": 0.88},
    {"id": "doc_c", "text": "Async backoff strategies — jitter, decorrelation, capping", "dense": 0.85, "sparse": 0.65},
    {"id": "doc_d", "text": "Team lunch menu options for next Friday", "dense": 0.10, "sparse": 0.20},
    {"id": "doc_e", "text": "Deep-dive: the tenacity library design", "dense": 0.75, "sparse": 0.55},
]


def _make_dense_retrieve() -> RetrieveFn:
    """Mock vector-DB retriever ranking by pre-baked cosine-like scores."""

    async def dense_retrieve(query: str, k: int) -> list[Candidate]:
        # Real caller: encode ``query`` and cosine-search a vector index.
        ranked = sorted(_CORPUS, key=lambda d: float(d["dense"]), reverse=True)
        return [
            Candidate(id=str(d["id"]), score=float(d["dense"]), source="vector", metadata={"text": d["text"]})
            for d in ranked[:k]
        ]

    return dense_retrieve


def _make_sparse_retrieve() -> RetrieveFn:
    """Mock BM25 retriever ranking by pre-baked keyword-overlap scores."""

    async def sparse_retrieve(query: str, k: int) -> list[Candidate]:
        # Real caller: tokenise ``query`` and BM25-search an inverted index.
        ranked = sorted(_CORPUS, key=lambda d: float(d["sparse"]), reverse=True)
        return [
            Candidate(id=str(d["id"]), score=float(d["sparse"]), source="keyword", metadata={"text": d["text"]})
            for d in ranked[:k]
        ]

    return sparse_retrieve


async def main() -> None:
    """Fuse dense + sparse rankings via RRF and print the top-3 result."""
    dense = _make_dense_retrieve()
    sparse = _make_sparse_retrieve()

    results = await ahybrid_retrieve(
        query="how do I retry async calls in Python?",
        dense_retrieve=dense,
        sparse_retrieve=sparse,
        top_n=3,
        dense_candidates=5,
        sparse_candidates=5,
        rrf_k=60,
    )

    print("Top-3 fused (RRF) results:")
    for rank, c in enumerate(results, start=1):
        text = c.metadata.get("text", "")
        print(f"  #{rank}  id={c.id!r}  rrf_score={c.score:.4f}  text={text!r}")


if __name__ == "__main__":
    asyncio.run(main())
