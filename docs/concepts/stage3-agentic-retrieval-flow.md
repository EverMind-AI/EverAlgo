# Agentic Retrieval Flow

Benchmark（stage 3）的完整 agentic 检索流程。基于 `aagentic_retrieve` 算子，组合 hybrid / cluster / rerank，对齐 `evercore-locomo-benchmark`（开源仓库的原始复现代码）中 `scene_retrieval.py` 的行为。

Scope：per-conversation。stage 2 为每个 conversation 建独立的 embedding / BM25 / cluster index，stage 3 逐 conversation 处理。

---

## Step 1：Embedding 检索

**操作对象**：当前 conversation 下所有 MemCell 的 fact-level embedding index（stage 2 预建的 `emb_conv_<i>.pkl`）

**过程**（`search.py:1045-1054` → `amaxsim_retrieve`）：

1. 将 query 做 embedding（`search.py:740`，调用 `embedding_client`：`Qwen3-Embedding-4B`，1024 维 Matryoshka 截断，`config.py:113,118`）
2. 遍历当前 conversation 的每个 MemCell，按以下规则选择参与计算的 embedding（`_emb_doc_to_children`，`search.py:699-710`）：
   - 如果该 MemCell 的 `atomic_facts` embedding 列表**非空** → **只用 atomic_facts 的 embedding**，subject/summary 不参与
   - 如果 `atomic_facts` 为空 → fallback：按 `episode → subject → summary` 顺序，取有 embedding 的字段
3. 对选中的每个 fact embedding 和 query embedding 算 cosine similarity（`search.py:720-721`，向量化计算）
4. 按 `parent_id` 将 fact 分组回所属 MemCell，每个 MemCell 取其所有 fact 中的最高分（MaxSim max-pool，由 `amaxsim_retrieve` 完成）
5. 按 MemCell 分数降序，取 top 50（`dense_candidates=50`，`config.py:27`）

**输出**：50 个 MemCell，每个带 cosine 分数

---

## Step 2：BM25 检索

**操作对象**：当前 conversation 下所有 MemCell 的 fact-level BM25 index（stage 2 预建的 `bm25_conv_<i>.pkl`）

BM25 index 的构建方式：stage 2 将每个 MemCell 的**所有 searchable unit**（每条 atomic fact、subject、summary）各自作为独立的一行加入 BM25 corpus，通过 `fact_to_doc_idx` 记录每行属于哪个 MemCell（`index.py:5-9`）。

**过程**（`search.py:1056-1065` → `amaxsim_retrieve`）：

1. 将 query 做 BM25 tokenization（Porter stemming + stopword removal，`search.py:653-656`）
2. `bm25.get_scores(tokenized_query)` 对 corpus 中**每一行**算 BM25 分数（`search.py:658`）——atomic facts、subject、summary 全部参与，无排他逻辑
3. 通过 `fact_to_doc_idx` 将每行的分数映射回所属 MemCell（`search.py:664`），每个 MemCell 取其所有行中的最高分（MaxSim max-pool，由 `amaxsim_retrieve` 完成）
4. 按 MemCell 分数降序，取 top 50（`sparse_candidates=50`，`config.py:28`）

**输出**：50 个 MemCell，每个带 BM25 分数

> **和 Step 1 的关键区别**：Embedding 检索在 atomic facts 非空时**只用 atomic facts**（排他）；BM25 检索**始终用全部 searchable unit**（atomic facts + subject + summary 都参与打分）。

---

## Step 3：RRF 融合

**操作对象**：Step 1 的 50 条 + Step 2 的 50 条

**过程**（`hybrid.py:74-91` → `fusion.py:27-43`）：

1. Step 1 和 Step 2 通过 `asyncio.gather` 并行执行（`hybrid.py:74-77`）
2. 如果两路都有结果：`rrf(dense, sparse, k=40)` 融合（`hybrid.py:87`，`config.py:33`）
   - 两个列表各自有排名（1-based），每个 MemCell 的 RRF 分数 = `Σ 1/(40 + rank_i)`（`fusion.py:40`）
   - 同一 MemCell 出现在两路中：两项贡献累加
   - 只出现在一路中：只有一项贡献
   - 空 ID 候选跳过（`fusion.py:37-38`）
   - 同一 ID 多次出现时，保留首次遇到的 Candidate 对象（`fusion.py:39` `doc_map.setdefault`）
3. 如果只有一路有结果：直接用该路，不融合（`hybrid.py:80-83`）
4. 按 RRF 分数降序排列（`fusion.py:42`）
5. RRF 函数本身无截断逻辑，返回全部融合结果；`ahybrid_retrieve` 的 `top_n` 由调用者传入，cluster 路径传 `len(all_docs)`（等于语料库大小，即不截断，`cluster.py:74`）

**输出**：去重后 70-90 条 MemCell（取决于两路重叠度），带 RRF 分数，不截断

---

## Step 4：Cluster 选择

**操作对象**：Step 3 的全部 RRF 结果 + 预建的 cluster 列表（stage 1 产物 `clusters_conv_<i>.json`）+ 全库 all_docs（`search.py:1083`，当前 conversation 的全部 MemCell，score=0.0）

**过程**（`cluster.py:74-97`，`search.py:1085-1097`）：

1. 建反向映射 `memcell_id → cluster_id`（`cluster.py:79-82`）
2. 按 RRF 分数从高到低扫描 Step 3 结果（`cluster.py:84-91`）：
   - 每条 MemCell 查它属于哪个 cluster，不属于任何 cluster 的跳过
   - 把 cluster_id 加入 `selected_cluster_ids` 集合
   - 集合大小达到 10 → 停止扫描（`cluster_top_k=10`，`config.py:87`）
3. 收集选中 cluster 的全部 member ID（`cluster.py:94-96`）
4. 从 `all_docs` 中过滤出 ID 在 member 集合中的 MemCell（`cluster.py:97`）——不只是 Step 3 命中的，是该 cluster 下所有 MemCell

**输出**：10 个 cluster 的全部 MemCell（通常 30-60 条），按 `all_docs` 原始顺序，无分数排序，无截断

---

## Step 5：R1 Rerank

**操作对象**：Step 4 的全部展开成员（30-60 条 MemCell）

**过程**（`agentic.py:142-143` → `search.py:460-571`）：

1. Step 4 的全部展开成员作为 `round1` 进入 `aagentic_retrieve`（`agentic.py:137`）。`cluster_scoped` 闭包忽略了 `round1_top_n=50` 参数（`search.py:1068-1069`），返回全部展开成员，不截断
2. 每条 MemCell 取 `episode.content` 作为 reranker 输入文本（`_format_doc_for_rerank`，`search.py:362-373`）。无 fallback，`episode.content` 为空则 raise `ValueError`
3. 送 Qwen3-Reranker-4B cross-encoder（`config.py:119`），instruction = `"Determine if the passage contains specific facts, entities (names, dates, locations), or details that directly answer the question."`（`config.py:50-53`）
4. 分 batch 处理：`batch_size=32`，`concurrent_batches=2`（`config.py:44-45`），每 batch 最多重试 3 次（指数退避，`config.py:46-47`）
5. 如果 batch 成功率低于 30%（`fallback_threshold=0.3`，`config.py:70`）→ raise `RuntimeError`，由外层重试循环捕获（最多 5 次）。不回退到原始排序
6. 按 reranker 分数降序排列，截断到 top 10（`round1_rerank_top_n=10`，`config.py:41`，`agentic.py:143`）

**输出**：10 条 MemCell，按 cross-encoder 分数排序（记为 `reranked`，后续 Step 6-12 都基于此）

---

## Step 6：LLM 充分性检查

**操作对象**：Step 5 的 10 条 `reranked`

**过程**（`agentic.py:148-153` → `agentic.py:390-403`）：

1. 格式化 10 条文档（`_format_docs()`，`agentic.py:457-481`）：
   - Title：`Candidate.metadata["episode"]["subject"]`，无则 `"N/A"`
   - Date：`Candidate.metadata["timestamp"]`（ms epoch → ISO `YYYY-MM-DDTHH:MM:SSZ`），无效则 `"N/A"`
   - Content：`Candidate.metadata["episode"]["content"]`，无则 raise `ValueError`；超 500 字符截断 + `"..."`
   - 无 episode dict 则 raise `TypeError`
   - 每条格式：`Document {i}:\n  Title: ...\n  Date: ...\n  Content: ...\n`
   - 空列表返回 `"No retrieval results"`
2. 渲染 prompt：`SUFFICIENCY_CHECK_PROMPT_EN.format(query=query, retrieved_docs=formatted)`（`agentic.py:402`）
3. LLM 调用：`gpt-4.1-mini`，`temperature=0.0`（`agentic.py:345`）
4. JSON 提取：`find("{")` + `rfind("}")`（`agentic.py:332-333`），解析失败 raise `ValueError`
5. 解析为 `SufficiencyCheckResponse`：`{is_sufficient, reasoning, key_information_found, missing_information}`（`agentic.py:347-352`）

**分支**（`agentic.py:155-165`）：
- **充分** → 返回 `reranked[:10]`（cross-encoder 顺序），流程结束
- **不充分** → 进入 Step 7

**输出**：`SufficiencyCheckResponse`（充分/不充分 + 已找到信息 + 缺失信息）

---

## Step 7：Multi-Query 生成

**操作对象**：原始 query + Step 5 的 10 条 `reranked`（同 Step 6 格式化方式） + Step 6 产出的 `key_information_found` 和 `missing_information`

**过程**（`agentic.py:220-228` → `agentic.py:408-434`）：

1. 格式化 10 条文档（同 Step 6 的 `_format_docs()`）
2. `missing_info` join 为逗号分隔字符串，空则 `"N/A"`（`agentic.py:424`）
3. `key_info` 同上（`agentic.py:425`）
4. 渲染 prompt：`MULTI_QUERY_PROMPT_EN.format(original_query=..., retrieved_docs=..., missing_info=..., key_info=...)`（`agentic.py:423-427`）
5. LLM 调用：`gpt-4.1-mini`，`temperature=0.4`（`agentic.py:361`）
6. JSON 提取：同 Step 6（`find("{")` + `rfind("}")`），解析失败 raise `ValueError`
7. 解析为 `MultiQueryResponse`：`{queries: [str, ...], reasoning: str}`（`agentic.py:363-366`）
8. 校验每个查询：长度 5-300 字符、不与原始 query 相同（case-insensitive），不通过的丢弃（`agentic.py:430-432`）
9. 截断到 `multi_query_count=3`（`config.py:34`，`agentic.py:433`）

**输出**：最多 3 个经过校验的互补查询字符串

---

## Step 8：R2 并行召回

**操作对象**：Step 7 的最多 3 个查询 × 全库（脱离 cluster 限制）

**过程**（`agentic.py:231-232`）：

1. `r2 = round2_retrieve`（= `hybrid_full`，全库 Hybrid，`search.py:1100`）——R2 不走 cluster 路径，直接全库检索
2. 3 个查询通过 `asyncio.gather` 并行执行（`agentic.py:232`）
3. 每个查询走完整的 Step 1→2→3 流程：Emb top 50 + BM25 top 50 → RRF(k=40)
4. 每个查询返回 top 50（`round1_top_n=50`，`search.py:1131` → `agentic.py:232`）

**输出**：最多 3 个列表，每个最多 50 条 MemCell

---

## Step 9：Multi-RRF 融合 + 截断

**操作对象**：Step 8 的最多 3 个列表

**过程**（`agentic.py:233-238`）：

1. 如果只有 1 个列表：直接用，不融合（`agentic.py:233-234`）
2. 否则：`rrf(*round2_lists, k=40)` 融合（`rrf_k=40`，`config.py:33` → `search.py:1135`，`agentic.py:236`）
3. 截断到 `round2_cap=40`：`fused = fused[:40]`（`agentic.py:237-238`）

**输出**：最多 40 条 MemCell，按 Multi-RRF 分数降序

---

## Step 10：去重合并

**操作对象**：Step 5 的 `reranked`（10 条）+ Step 9 的 `fused`（≤40 条）

**过程**（`agentic.py:306-311`）：

1. 建 R1 的 ID 集合：`seen = {c.id for c in reranked}`（`agentic.py:306`）
2. 从 Step 9 结果中去掉已在 R1 里的：`r2_unique = [c for c in fused if c.id not in seen]`（`agentic.py:307`）
3. cap：`keep = max(0, 40 - 10) = 30`，`r2_unique = r2_unique[:30]`（`agentic.py:308-310`）
4. 合并：`merged = reranked(10) + r2_unique(≤30)`（`agentic.py:311`）

**输出**：最多 40 条 MemCell，R1 在前，R2 补充在后

---

## Step 11：Final Rerank + 截断返回

**操作对象**：Step 10 的 `merged`（≤40 条）

**过程**（`agentic.py:312-314`）：

1. 同一个 Qwen3-Reranker-4B cross-encoder（和 Step 5 共用 `rerank_fn`，`search.py:1105`）
2. 对全部 merged 候选做打分，按分数降序排列
3. reranker 不截断，返回全部重排结果（`_build_rerank_fn` 传 `top_n=len(candidates)`，`search.py:575`）
4. 截断：`final = merged[:top_n]`（`top_n=10`，`config.py:35`，`agentic.py:314`）

**输出**：最终 top 10 条 MemCell + `AgenticDecision` 元数据（`agentic.py:315-324`）
