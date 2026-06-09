# Agentic Retrieval Flow

The complete agentic retrieval flow in benchmark stage 3. Built on the `aagentic_retrieve` operator, combining hybrid / cluster / rerank, aligned with the behavior of `scene_retrieval.py` in the `evercore-locomo-benchmark` (the original reproduction code in the open-source repo).

Scope: per-conversation. Stage 2 builds independent embedding / BM25 / cluster indexes for each conversation; stage 3 processes one conversation at a time.

---

## Step 1: Embedding Retrieval

**Input**: the fact-level embedding index for all MemCells in the current conversation (pre-built in stage 2 as `emb_conv_<i>.pkl`)

**Process** (`search.py:1016-1025` -> `amaxsim_retrieve`):

1. Embed the query (`search.py:711`, calling `embedding_client`: `Qwen3-Embedding-4B`, 1024-dim Matryoshka truncation, `config.py:108,113`)
2. For each MemCell in the current conversation, select the embeddings to use (`_emb_doc_to_children`, `search.py:654-696`):
   - If the MemCell's `atomic_facts` embedding list is **non-empty** -> **use only the atomic_facts embeddings**; subject/summary do not participate
   - If `atomic_facts` is empty -> fallback: in order `episode -> subject -> summary`, use the first field that has an embedding
3. Compute cosine similarity between each selected fact embedding and the query embedding (`search.py:691-692`, vectorized computation)
4. Group facts back to their parent MemCell by `parent_id`; for each MemCell take the highest score among its facts (MaxSim max-pool, performed by `amaxsim_retrieve`)
5. Sort MemCells by score descending, take top 50 (`dense_candidates=50`, `config.py:27`)

**Output**: 50 MemCells, each with a cosine score

---

## Step 2: BM25 Retrieval

**Input**: the fact-level BM25 index for all MemCells in the current conversation (pre-built in stage 2 as `bm25_conv_<i>.pkl`)

The BM25 index is built as follows: stage 2 adds every **searchable unit** of each MemCell (each atomic fact, subject, summary) as an independent row in the BM25 corpus, with `fact_to_doc_idx` recording which MemCell each row belongs to (`index.py:5-9`).

**Process** (`search.py:1027-1036` -> `amaxsim_retrieve`):

1. Tokenize the query for BM25 (Porter stemming + stopword removal, `search.py:623-626`)
2. `bm25.get_scores(tokenized_query)` scores **every row** in the corpus (`search.py:629`) — atomic facts, subject, and summary all participate; there is no exclusion logic
3. Map each row's score back to its parent MemCell via `fact_to_doc_idx` (`search.py:635`); for each MemCell take the highest score among its rows (MaxSim max-pool, performed by `amaxsim_retrieve`)
4. Sort MemCells by score descending, take top 50 (`sparse_candidates=50`, `config.py:28`)

**Output**: 50 MemCells, each with a BM25 score

> **Key difference from Step 1**: embedding retrieval **uses only atomic facts** when they are non-empty (exclusive); BM25 retrieval **always uses all searchable units** (atomic facts + subject + summary all participate in scoring).

---

## Step 3: RRF Fusion

**Input**: 50 candidates from Step 1 + 50 candidates from Step 2

**Process** (`hybrid.py:74-91` -> `fusion.py:27-43`):

1. Step 1 and Step 2 execute in parallel via `asyncio.gather` (`hybrid.py:74-77`)
2. If both routes have results: `rrf(dense, sparse, k=40)` fusion (`hybrid.py:87`, `config.py:33`)
   - Each list has its own ranking (1-based); each MemCell's RRF score = `sum(1/(40 + rank_i))` (`fusion.py:40`)
   - A MemCell appearing in both routes: contributions from both are summed
   - A MemCell appearing in only one route: only one contribution
   - Candidates with empty IDs are skipped (`fusion.py:37-38`)
   - When the same ID appears multiple times, the first-encountered Candidate object is kept (`fusion.py:39` `doc_map.setdefault`)
3. If only one route has results: use it directly, no fusion (`hybrid.py:80-83`)
4. Sort by RRF score descending (`fusion.py:42`)
5. The RRF function itself does not truncate; it returns all fused results. `ahybrid_retrieve`'s `top_n` is passed by the caller; the cluster path passes `len(all_docs)` (= corpus size, i.e. no truncation, `cluster.py:75`)

**Output**: 70-90 de-duplicated MemCells (depending on overlap between the two routes), with RRF scores, not truncated

---

## Step 4: Cluster Selection

**Input**: all RRF results from Step 3 + pre-built cluster list (stage 1 artifact `clusters_conv_<i>.json`) + full-corpus all_docs (`search.py:1054`, all MemCells in the current conversation, score=0.0)

**Process** (`cluster.py:75-102`, `search.py:1056-1068`):

1. Build a reverse mapping `memcell_id -> cluster_id` (`cluster.py:80-82`)
2. Scan Step 3 results from highest to lowest RRF score (`cluster.py:84-91`):
   - For each MemCell, look up which cluster it belongs to; skip if it does not belong to any cluster
   - Add the cluster_id to the `selected_cluster_ids` set
   - Stop scanning when the set reaches size 10 (`cluster_top_k=10`, `config.py:87`)
3. Collect all member IDs from the selected clusters (`cluster.py:97-100`)
4. Filter `all_docs` to keep only MemCells whose IDs are in the member set (`cluster.py:102`) — this includes all MemCells in those clusters, not just the ones that appeared in Step 3

**Output**: all MemCells from 10 clusters (typically 30-60), in `all_docs` original order, with no score sorting and no truncation

---

## Step 5: R1 Rerank

**Input**: all expanded members from Step 4 (30-60 MemCells)

**Process** (`agentic.py:142-143` -> `search.py:431-585`):

1. All expanded members from Step 4 enter `aagentic_retrieve` as `round1` (`agentic.py:137`). The `cluster_scoped` closure ignores the `round1_top_n=50` parameter (`search.py:1056-1060`), returning all expanded members without truncation
2. For each MemCell, take `episode.content` as the reranker input text (`_format_doc_for_rerank`, `search.py:358-372`). No fallback; empty `episode.content` raises `ValueError`
3. Send to Qwen3-Reranker-4B cross-encoder (`config.py:114`), instruction = `"Determine if the passage contains specific facts, entities (names, dates, locations), or details that directly answer the question."` (`config.py:50-53`)
4. Process in batches: `batch_size=32`, `concurrent_batches=2` (`config.py:44-45`), each batch retries up to 3 times with exponential backoff (`config.py:46-47`)
5. If batch success rate falls below 30% (`fallback_threshold=0.3`, `config.py:49`) -> raise `RuntimeError`, caught by the outer retry loop (up to 5 attempts). Does not fall back to original ordering
6. Sort by reranker score descending, truncate to top 10 (`round1_rerank_top_n=10`, `config.py:41`, `agentic.py:143`)

**Output**: 10 MemCells, sorted by cross-encoder score (referred to as `reranked`; Steps 6-12 all build on this)

---

## Step 6: LLM Sufficiency Check

**Input**: 10 `reranked` MemCells from Step 5

**Process** (`agentic.py:148-153` -> `agentic.py:392-405`):

1. Format the 10 documents (`_format_docs()`, `agentic.py:458-484`):
   - Title: `Candidate.metadata["episode"]["subject"]`, or `"N/A"` if absent
   - Date: `Candidate.metadata["timestamp"]` (ms epoch -> ISO `YYYY-MM-DDTHH:MM:SSZ`), or `"N/A"` if invalid
   - Content: `Candidate.metadata["episode"]["content"]`, or raise `ValueError` if absent; truncated to 500 chars + `"..."` if longer
   - Missing episode dict raises `TypeError`
   - Each document formatted as: `Document {i}:\n  Title: ...\n  Date: ...\n  Content: ...\n`
   - Empty list returns `"No retrieval results"`
2. Render prompt: `SUFFICIENCY_CHECK_PROMPT_EN.format(query=query, retrieved_docs=formatted)` (`agentic.py:404`)
3. LLM call: `gpt-4.1-mini`, `temperature=0.0` (`agentic.py:347`)
4. JSON extraction: `find("{")` + `rfind("}")` (`agentic.py:334-335`); parse failure raises `ValueError`
5. Parse into `SufficiencyCheckResponse`: `{is_sufficient, reasoning, key_information_found, missing_information}` (`agentic.py:349-354`)

**Branch** (`agentic.py:155-165`):
- **Sufficient** -> return `reranked[:10]` (in cross-encoder order), flow ends
- **Not sufficient** -> proceed to Step 7

**Output**: `SufficiencyCheckResponse` (sufficient/not sufficient + found information + missing information)

---

## Step 7: Multi-Query Generation

**Input**: original query + 10 `reranked` MemCells from Step 5 (formatted the same way as Step 6) + `key_information_found` and `missing_information` from Step 6

**Process** (`agentic.py:220-228` -> `agentic.py:408-433`):

1. Format the 10 documents (same `_format_docs()` as Step 6)
2. Join `missing_info` as a comma-separated string; empty becomes `"N/A"` (`agentic.py:426`)
3. Same for `key_info` (`agentic.py:427`)
4. Render prompt: `MULTI_QUERY_PROMPT_EN.format(original_query=..., retrieved_docs=..., missing_info=..., key_info=...)` (`agentic.py:423-428`)
5. LLM call: `gpt-4.1-mini`, `temperature=0.4` (`agentic.py:363`)
6. JSON extraction: same as Step 6 (`find("{")` + `rfind("}")`); parse failure raises `ValueError`
7. Parse into `MultiQueryResponse`: `{queries: [str, ...], reasoning: str}` (`agentic.py:365-368`)
8. Validate each query: length 5-300 chars, not identical to the original query (case-insensitive); discard those that fail (`agentic.py:430-432`)
9. Truncate to `multi_query_count=3` (`config.py:34`, `agentic.py:431-432`)

**Output**: up to 3 validated complementary query strings

---

## Step 8: R2 Parallel Recall

**Input**: up to 3 queries from Step 7 x full corpus (no cluster restriction)

**Process** (`agentic.py:229-230`):

1. `r2 = round2_retrieve` (= `hybrid_full`, full-corpus Hybrid, `search.py:1071`) — R2 does not go through the cluster path; it retrieves from the full corpus directly
2. The 3 queries execute in parallel via `asyncio.gather` (`agentic.py:230`)
3. Each query goes through the full Step 1->2->3 flow: Emb top 50 + BM25 top 50 -> RRF(k=40)
4. Each query returns top 50 (`round1_top_n=50`, `search.py:1102` -> `agentic.py:230`)

**Output**: up to 3 lists, each with up to 50 MemCells

---

## Step 9: Multi-RRF Fusion + Truncation

**Input**: up to 3 lists from Step 8

**Process** (`agentic.py:231-236`):

1. If only 1 list: use it directly, no fusion (`agentic.py:231-232`)
2. Otherwise: `rrf(*round2_lists, k=40)` fusion (`rrf_k=40`, `config.py:33` -> `search.py:1106`, `agentic.py:234`)
3. Truncate to `round2_cap=40`: `fused = fused[:40]` (`agentic.py:235-236`)

**Output**: up to 40 MemCells, sorted by Multi-RRF score descending

---

## Step 10: Deduplication and Merge

**Input**: `reranked` from Step 5 (10 items) + `fused` from Step 9 (<=40 items)

**Process** (`agentic.py:306-311`):

1. Build the R1 ID set: `seen = {c.id for c in reranked}` (`agentic.py:306`)
2. Remove R1 duplicates from Step 9 results: `r2_unique = [c for c in fused if c.id not in seen]` (`agentic.py:307`)
3. Cap: `keep = max(0, 40 - 10) = 30`, `r2_unique = r2_unique[:30]` (`agentic.py:308-310`)
4. Merge: `merged = reranked(10) + r2_unique(<=30)` (`agentic.py:311`)

**Output**: up to 40 MemCells, R1 first, R2 supplements after

---

## Step 11: Final Rerank + Truncation

**Input**: `merged` from Step 10 (<=40 items)

**Process** (`agentic.py:312-314`):

1. Same Qwen3-Reranker-4B cross-encoder (shares `rerank_fn` with Step 5, `search.py:1076`)
2. Score all merged candidates, sort by score descending
3. The reranker does not truncate; it returns all re-ranked results (`_build_rerank_fn` passes `top_n=len(candidates)`, `search.py:574`)
4. Truncate: `final = merged[:top_n]` (`top_n=10`, `config.py:35`, `agentic.py:314`)

**Output**: final top 10 MemCells + `AgenticDecision` metadata (`agentic.py:315-324`)
