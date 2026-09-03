# Stage 5 Agentic Retrieval Flow

Stage 5 runs retrieval for every eligible question in each conversation. It consumes the BM25, embedding, and cluster indexes produced by Stage 4, composes EverAlgo's MaxSim, hybrid, cluster, and agentic retrieval operators, and writes `search_results.json`. The canonical stage ordering is `_STAGE_RUNNERS` in `benchmarks/common/runner.py`.

## Flow

```text
question
   │
   ├── dense route: AtomicFact + subject embeddings ── MaxSim ──┐
   │                                                            │
   └── sparse route: facts + subject + Episode body head ─ MaxSim│
                                                                ▼
                                                        hybrid RRF
                                                                │
                                                  first 10 clusters
                                                                │
                                               expand full membership
                                                                │
                                                    rerank → top 10
                                                                │
                                                sufficiency LLM check
                                           ┌────────────────────┴───────────────────┐
                                           │ sufficient                             │ insufficient
                                           ▼                                        ▼
                                      final top 10                      generate up to 3 queries
                                                                                    │
                                                                       full-corpus hybrid retrieval
                                                                                    │
                                                                        multi-query RRF + dedupe
                                                                                    │
                                                                         cap 40 → rerank → top 10
```

## 1. Stage 4 index contracts

Stage 5 loads three trusted local pickle artifacts per conversation. These files must never come from an untrusted or network-shared path because Python pickle can execute code while loading.

| Artifact | Searchable representation |
|---|---|
| `emb_conv_<i>.pkl` | Per-Episode AtomicFact vectors, subject vector, and a stored Episode vector |
| `bm25_conv_<i>.pkl` | One row per AtomicFact plus subject and the first 200 characters of the Episode body; each row maps back to its parent Episode |
| `cluster_index_conv_<i>.pkl` | Typed geometric clusters whose members are Episode IDs |

Although the embedding artifact also carries the Episode body vector, the current dense search path deliberately scores only AtomicFact vectors plus the subject vector. It has no summary or Episode-body fallback.

## 2. Dense and sparse MaxSim routes

The dense route embeds the question, computes cosine similarity against every AtomicFact and subject vector for an Episode, and keeps that Episode's maximum child score through `amaxsim_retrieve`.

The sparse route tokenizes the question, scores every BM25 row, maps rows back to their parent Episode, and likewise keeps the maximum row score per Episode.

The two routes each request 50 candidates by default. `ahybrid_retrieve` runs them concurrently and fuses their Episode rankings with reciprocal-rank fusion using `k=40`.

## 3. Cluster-scoped Round 1

`acluster_retrieve` scans the hybrid results in rank order until it has seen 10 distinct clusters by default. It then expands every member of those clusters from the full Episode list.

The expansion is deliberately unranked and unscored: it preserves the caller's `all_docs` order. `aagentic_retrieve` then applies the configured cross-encoder reranker and keeps the first 10 candidates for the LLM sufficiency check.

## 4. Sufficiency and Round 2

If the LLM says the reranked Round 1 context is sufficient, Stage 5 returns those candidates, truncated to the response limit of 10.

If it is insufficient, the current benchmark requests up to three complementary queries. Each query uses the full-corpus hybrid retriever rather than the cluster-scoped retriever, allowing Round 2 to escape the initial clusters. Multiple result lists are fused with RRF, deduplicated against the reranked Round 1 set, capped so the merged pool contains at most 40 candidates, reranked again against the original question, and truncated to the final top 10.

The reusable `aagentic_retrieve` operator also supports a single refined-query strategy, but the benchmark currently selects `refinement_strategy="multi_query"`.

## 5. Output and failure behavior

Each successful QA result stores the selected Episode IDs under `members`, the original QA record, and `retrieval_metadata` containing the sufficiency decision, missing and found information, generated queries, final count, token totals, and a compact trace.

Filtered categories return no result. Missing BM25 or embedding files soft-skip a conversation; a missing cluster index raises because the agentic benchmark path requires it. Per-question retrieval failures are retried and then written to `search_<question_id>.error.txt` before the stage fails loudly.

## Source of truth

- Stage registry: `benchmarks/common/runner.py::_STAGE_RUNNERS`
- Index contents: `benchmarks/common/stages/index.py::extract_searchable_units`, `_build_emb_index`, `_build_cluster_index`
- Retrieval composition: `benchmarks/common/stages/search.py::_build_retrieval_closures`, `_build_cluster_closures`, `_run_agentic_retrieval`
- Dense scoring contract: `benchmarks/common/stages/search.py::_score_emb_item`, `_emb_doc_to_children`
- Generic algorithms: `everalgo.rank.maxsim.amaxsim_retrieve`, `everalgo.rank.hybrid.ahybrid_retrieve`, `everalgo.rank.cluster.acluster_retrieve`, `everalgo.rank.agentic.aagentic_retrieve`
