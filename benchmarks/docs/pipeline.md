# 基准测试流水线

`benchmarks/` 目录下 LoCoMo 基准测试的端到端流程：5 个串行阶段，每个阶段消费前一阶段
落盘的中间结果。中间产物以 Pickle / JSON 形式存放，所以任一阶段都可通过 `--stages`
单独重跑。

## 总览

```
LoCoMo JSON  ───┐
                ├─→ 阶段 1 抽取  ──→ MemCells (json)
                                       │
                                       ├─→ 阶段 2 建索引 ──→ BM25 + Emb (pkl)
                                       │                       │
                                       └──────────────────────┴──→ 阶段 3 检索 ──→ members
                                                                                     │
                                       ┌─────────────────────────────────────────────┘
                                       ↓
                                  阶段 4 回答 ──→ generated_answers
                                                       │
                                                       ↓
                                                 阶段 5 评估 ──→ accuracy
                                                                    │
                                                                    ↓
                                                              report.{txt,json}
```

| 阶段 | 输出目录 | 格式 | 使用的 EverAlgo 包 | 外部服务 |
|---|---|---|---|---|
| 1 抽取 | `stage1_extract/` | 每对话一个 JSON（+ `clusters_conv_<i>.json`） | `boundary`, `user-memory`, `clustering` | OpenRouter (`gpt-4.1-mini`) |
| 2 建索引 | `stage2_index/` | 每对话多个 Pickle（bm25 / emb / cluster_index） | `clustering`（仅 `Cluster` 类型） | DeepInfra (`Qwen3-Embedding-4B`) |
| 3 检索 | `stage3_search/` | 单个 JSON | `rank`（融合 + MaxSim）| OpenRouter + DeepInfra (`Qwen3-Reranker-4B`) |
| 4 回答 | `stage4_answer/` | 单个 JSON | — | OpenRouter (`gpt-4.1-mini`) |
| 5 评估 | `stage5_evaluate/` | 单个 JSON | — | OpenRouter (`gpt-4o-mini`) |

---

## 数据预处理（Loader）

`LocomoDataset.load_conversations()` 把 LoCoMo `locomo10.json` 转成统一的
`Conversation { id, speakers, messages }` 值类型，几个关键处理对齐 EverCore main：

- **message timestamp**：LoCoMo 只有 session 级 `session_<N>_date_time`，没有
  per-message timestamp。loader 给同一 session 内每条 message 派 `session_time +
  i*30s` 递增的毫秒 epoch（mirror main `stage1_memcells_extraction.py:114-123`）。
  让 BoundaryDetector 看到 monotonically advancing timestamps，避免同 session
  内全部 message 同 ts 让 LLM 误判为「并发说话」漏切。
- **sender_id 命名**：`f"{speaker.lower().replace(' ','_')}_{conv_id}"`，其中 `conv_id`
  是完整字符串 `locomo_exp_user_<i>`，所以实际形如 `caroline_locomo_exp_user_0`
  （`loader.py:77`）。同 conv 内 speaker 稳定区分；跨 conv 因 conv_id 不同天然唯一。
- **img_url 拼接**：LoCoMo 5882 条 message 里 910 条（15.5%）含 `img_url` +
  `blip_caption`。loader 把图片描述拼到 content 前：
  `"[<speaker> shared an image: <blip_caption>] <text>"`（mirror main
  `stage1_memcells_extraction.py:134-140`）。EvalQA 里 39.4% 的题 evidence 含图片
  消息 —— 不拼丢的是真信号。
- **cat 5 过滤**：LoCoMo 的 `category=5` 是 adversarial（设计上无答案的对抗题）。
  loader 在 `load_qa_pairs` 层直接跳过，下游 search / answer / judge 不浪费算力。

输出的 `Message` 字段：`id` (= raw `dia_id`) / `role` (= `"user"`，LoCoMo 无 system
区分) / `content` (str，含 image caption 前缀) / `timestamp` (ms epoch) / `sender_id`
/ `sender_name`。

---

## 阶段 1 —— 抽取（Extract）

**输入**：`LocomoDataset.load_conversations()` 返回的完整对话。LoCoMo 的 `conv_0`
含 419 条消息，跨多个会话日。

**三步流程（每对话一次）**：

```
原始消息
    ↓
[1] BoundaryDetector       ← LLM 判断切片位置
    ↓
list[MemCell]              ← 几十个语义连续的记忆单元
    ↓
[2] EpisodeExtractor       ← 每个 MemCell 调一次 LLM
    ↓
Episode { subject, content }
    ↓
[3] AtomicFactExtractor    ← 每个 MemCell 调一次 LLM
    ↓
list[AtomicFact { fact }]
```

1. **`BoundaryDetector`**（everalgo-user-memory）：把消息序列切成多个 MemCell，每个
   是 1~N 条消息的语义连续块。**逐消息增量检测**：调 `adetect_step` 一条条喂
   （`extract.py:_detect_all_boundaries`），caller 侧只负责 front-2-buffer（前 2 条
   不触发 LLM）和流末把残余 `tail` flush 成最后一个 cell；smart-mask 阈值门控、masking、
   force-split（`hard_token_limit=8192` / `hard_message_limit=50`）与 cut-and-bridge
   状态转移都封装在 `adetect_step` 内。注意 50 是内部 force-split 的硬上限，
   **不是「每批 50 条」的批大小**。
2. **`EpisodeExtractor`**：对每个 MemCell 生成叙述 —— `subject`（短标题）+
   `content`（长叙述）。
3. **`AtomicFactExtractor`**：从上一步生成的 **Episode 正文**（不是原始对话消息）抽取
   离散事实列表，调 `aextract_from_text(episode_body, timestamp=mc.timestamp,
   prompt=EVENT_LOG_PROMPT)`（`extract.py:173-175`）。输入是 episode body 字符串 +
   MemCell 时间戳；`EVENT_LOG_PROMPT` 用 `event_log` schema key（与算法默认的
   `ATOMIC_FACT_FROM_TEXT_PROMPT_EN` 的 `atomic_facts` key 等价，parser 两者都容忍）。

### Clustering pass（默认开启）

抽取完成后，若 `enable_clustering=True`（默认，`config.py:77`），对每个 MemCell 用其
episode body 的 embedding（即 `episode.content_embeddings`）增量调
`everalgo.clustering.cluster_by_geometry`（cosine + 时间窗，`cluster_similarity_threshold=0.70`
/ `cluster_max_time_gap_days=7.0`）分簇，落 `clusters_conv_<i>.json`
（`extract.py:_run_clustering_pass`）。这套簇被 stage2 转成 cluster index、stage3 用于
2-level 检索 —— **clustering 实际参与 LoCoMo 跑分**。

未运行的提取器：`ForesightExtractor` / `ProfileExtractor` 在 stage1 代码里**从未被调用**
（也不存在 `enable_foresight_extraction` / `enable_profile_extraction` 这两个开关）。

### 输出：`memcells_conv_<i>.json`

JSON 结构是 EverAlgo 原生类型的扁平超集：

```json
[
  {
    "id": "0",
    "timestamp": 1683525360000,
    "items": [
      {
        "id": "D1:1",
        "role": "user",
        "content": "...",
        "timestamp": 1683525360000,
        "sender_id": "caroline_locomo_exp_user_0",
        "sender_name": "Caroline"
      }
    ],
    "episode": {
      "subject": "Caroline reconnects with Melanie",
      "summary": "On May 8 2023, Caroline and Melanie reconnected... (content[:200]+'...')",
      "content": "On May 8 2023, Caroline and Melanie reconnected...",
      "content_embeddings": [0.01, -0.02]
    },
    "atomic_facts": {
      "time": "May 8, 2023(Monday) at 02:00 PM",
      "timestamp": 1683525360000,
      "atomic_fact": ["Caroline went to LGBTQ support group"],
      "fact_embeddings": [[0.03]]
    }
  }
]
```

注意几处与 EverAlgo 原生类型的差异（`extract.py:_serialize_memcell`）：

- `episode.summary` 无独立 LLM 摘要，恒为 `content[:200] + "..."`（尾部省略号是 load-bearing，下游 BM25 / embedding 会 tokenize 此字段）。
- `episode.content_embeddings` 只用于 stage1 clustering。
- `atomic_facts` 是 **dict**（`time` / `timestamp` / `atomic_fact` 字符串列表 / `fact_embeddings`），不是 `[{"fact": ...}]` 数组。
- clustering 开启时同目录额外写 `clusters_conv_<i>.json`（`{clusters: [...], memcell_to_cluster: {...}}`）。

会话 ID 由文件名 `memcells_conv_<i>.json` 隐式承载，无需在记录内重复。419 条消息 → 约 30~50 个 MemCell。

### 并发

- 跨对话：`asyncio.Semaphore(max_concurrent_convs=10)` 控制并发对话数
- 单对话内：跨 MemCell 用 `asyncio.gather` 并行，全局 `mc_sem = Semaphore(20)`（硬编码，`extract.py:586`）限制 MemCell LLM 并发上限

---

## 阶段 2 —— 建索引（Index）

对每个对话的 MemCell 集合分别建两套检索结构。

### BM25（关键词检索）

**fact 级索引**：每个 atomic_fact、`episode.subject`、`episode.summary` 各自 tokenize 成一条独立 BM25 document（不拼接、不加权重复），`fact_to_doc_idx` 把每行映射回父 MemCell。检索时取一个 doc 所有 fact 行的最高分（MaxSim 聚合）。

```python
# extract_searchable_units(mc)：每个 unit 一行 doc
units = list(atomic_facts)       # 每个 fact 一行
if subject: units.append(subject)
if summary: units.append(summary)
if not units and content:        # 仅当上面全空才 fallback 到 content
    units = [content]
# _tokenize：lower -> word_tokenize -> 保留 alpha 且 len>=2 且非停词 -> PorterStemmer
tokens = ["carolin", "reconnect", "melani", "lgbtq", "support"]
bm25 = BM25Okapi(fact_corpus)    # 所有 unit 跨所有 MemCell 摊平成一个语料
```

落盘 `bm25_conv_<i>.pkl`：`{"bm25", "docs": memcells, "fact_to_doc_idx", "index_type": "maxsim"}`，约 200 KB。纯本地计算。

### Embedding（语义检索）

**整个对话 flatten 后批量 embed**（不是「每个 MemCell 一次 API」）：把全对话所有待 embed 文本摊平成一个列表，按 `embedding_batch_size=256` 分批、最多 `embedding_concurrent_batches=5` 个批并发、组间 sleep 1s（`index.py:_embed_batched`）。stage1 已算过的 `fact_embeddings` 会被复用，跳过重复 embed。

字段选择（`_flatten_conv_texts`，`index.py`）—— **atomic_facts + subject + summary 三者都 embed，不是二选一**：

```python
if atomic_facts and not has_precomputed:
    texts += atomic_facts            # 每个 fact 一向量
if subject: texts.append(subject)
if summary: texts.append(clean_summary(summary))
if not atomic_facts and not has_precomputed and content:
    texts.append(content)            # content 仅在 atomic_facts 缺失时 fallback
vectors = await embedding.embed(texts, dimensions=1024)
```

每个 MemCell 存成 `{"doc": memcell, "embeddings": {...}}`，`embeddings` 含 `atomic_facts: [ndarray, ...]` + `subject: ndarray` + `summary: ndarray`（atomic_facts 缺失时退化为 `episode: ndarray`）。subject / summary 即使有 atomic_facts 也照样 embed，让 MaxSim 同时覆盖主题级信号。

落盘 `emb_conv_<i>.pkl`，约 7~10 MB（每个对话）。**这是磁盘占用的大头。**

### 为什么 dim=1024 不是 2560

Qwen3-Embedding-4B 是 Matryoshka 模型，DeepInfra 默认返回 2560 维；传
`dimensions=1024` 参数让服务端截断到 1024 维。对齐 EverCore main 的
`HybridVectorizeConfig.dimensions=1024`，确保 cosine 相似度与 RRF 排名跟 baseline
可逐字节对比。

### 为什么 atomic_facts 用一组向量、而非合并成一个

Stage 3 用 MaxSim 策略：对一个 query embedding，跟 MemCell 的所有 atomic_facts
向量逐个算 cosine 相似度，取**最大值**作为这个 MemCell 的整体得分。

直觉：「这条记忆里只要有任何一个事实和查询语义相关，整个记忆就值得检索出来」，比
合并平均更精准。

---

## 阶段 3 —— 检索（Search，最复杂）

Agentic 多轮检索，每题独立流程：

整个 stage3 委托 `everalgo.rank.aagentic_retrieve`，benchmark 只注入 BM25 / Embedding 检索闭包和 reranker；充分性检查、多查询改写的 prompt 全来自 `everalgo.rank.prompts`。

```
question
  ↓  (everalgo.rank.aagentic_retrieve)
[第 1 轮 base_retrieve]
  ├─ LoCoMo 默认 cluster-scoped (acluster_retrieve)：
  │    Level-1 hybrid (BM25 + Emb MaxSim, RRF k=40) 召回 cluster_base_candidates=100
  │    → 选 top cluster_top_k=10 个簇 → Level-2 展开簇内全部成员 (未重排)
  └─ 无 cluster index 时 fallback：hybrid_full (round1_top_n=50)
  ↓
[rerank] Qwen3-Reranker-4B 重排 → 取 round1_rerank_top_n=10
  ↓
[充分性检查] LLM (gpt-4.1-mini @ temp 0.0)「这 10 条能回答 question 吗？」
  ↓
  ├─ ✓ 充分    → 返回 top_n = response_top_k = 10
  │
  └─ ✗ 不充分  → 第 2 轮
       ↓
       [多查询改写] LLM 生成 multi_query_num=3 个 query (基于 missing_info + key_info)
       ↓
       3 个 query 并行 round2_retrieve=hybrid_full → 各 Top 50 → RRF(k=40) 融合
       ↓
       去重合并 (第 1 轮 rerank 后 10 条 + 第 2 轮)，round2_cap=40
       ↓
       reranker 再排 → 最终 top_n=10
```

### 充分性检查（Sufficiency Check）

LLM 接收 query + Top 10 docs，输出结构化 JSON：

- `is_sufficient: bool`
- `reasoning: str`（解释）
- `missing_info: list[str]`（如「需要知道事件发生的具体日期」）
- `key_information_found: list[str]`（如「已知 Caroline 是 counselor」）

充分 → 跳过第 2 轮省 LLM 调用；不充分 → 用 `missing_info` 指导第 2 轮 query 改写。

### 多查询改写（Multi-Query）

LLM 输入：原 query + missing_info + key_information_found，输出 3 个 refined
queries，每个聚焦不同的缺失维度。例如：

- 原 query：`When did Caroline meet her mentors?`
- 改写 1：`Caroline's first meeting with her counselor mentor in 2023`
- 改写 2：`Caroline's friends and family gatherings May 2023`
- 改写 3：`Specific dates Caroline saw her mentor figures`

### 输出：`search_results.json`

```json
{
  "locomo_exp_user_0": [
    {
      "question_id": "locomo_exp_user_0_qa0",
      "query": "When did Caroline go to the LGBTQ support group?",
      "members": ["3", "7", "12", "18", "21"],
      "original_qa": { "question_id": "...", "conv_id": "...", "question": "...", "golden_answer": "...", "category": "..." },
      "retrieval_metadata": {
        "is_multi_round": false,
        "is_sufficient": true,
        "reasoning": "...",
        "missing_info": [],
        "key_information_found": ["..."],
        "refined_queries": [],
        "query_strategy": "multi_query",
        "final_count": 10,
        "prompt_tokens": 1843,
        "completion_tokens": 122,
        "trace": { "final_top": [{ "id": "3", "score": 0.89 }] }
      }
    }
  ]
}
```

字段名是 **`members`**（不是 `memcell_ids`），值为会话内本地序号（如 `"0"`、`"1"` …），不是全局唯一 id。`retrieval_metadata` 的 `is_multi_round` / `reasoning` / `missing_info` / `query_strategy` 等直接透传自 `everalgo.rank` 的 `AgenticDecision`（`search.py:1134-1146`）；文档早期版本的 `round1_count` / `round1_reranked_count` / `total_latency_ms` 字段并不存在。

### 关键参数（对齐 EverCore）

| 参数 | 值 | 含义 |
|---|---|---|
| `hybrid_emb_candidates` | 50 | 每路 emb 检索召回数 |
| `hybrid_bm25_candidates` | 50 | 每路 BM25 召回数 |
| `hybrid_rrf_k` | 40 | RRF 融合常数（L1 hybrid 与 R2 multi-query 共用；库 `RankConfig.rrf_k` 默认 60 在此被 override） |
| `round1_rerank_top_n` | 10 | R1 rerank 后进充分性检查的窗口 |
| `response_top_k` | 10 | 最终返回 Top（拼 context 用） |
| `multi_query_num` | 3 | 第 2 轮改写数 |
| `max_concurrent_qa` | 30 | QA 并发数（stage 3/4/5）|

---

## 阶段 4 —— 回答（Answer）

**输入**：

- `stage3_search/search_results.json`（每题的 `members`，即检索出的 memcell 本地 id 列表）
- `stage1_extract/memcells_*.json`（按 member id 反查完整 MemCell）

### 流程（每题一次）

```python
# 1. 从 members 反查完整 memcell，取前 response_top_k=10
mc_ids = item["members"]
top_memcells = [memcells_map[conv_id][mc_id] for mc_id in mc_ids[:10] if mc_id in memcells_map[conv_id]]

# 2. 拼 context（对齐 EverCore 格式：每条 doc 间用 "\n---\n\n" 明确分块）
context = f"""Episodes memories for conversation between Caroline and Melanie:

Caroline reconnects with Melanie: On May 8 2023, Caroline...
---

Discussion about LGBTQ support: Caroline shared that...
---

..."""

# 3. 喂给 LLM（gpt-4.1-mini @ temperature=0.0 覆盖 config 0.3；max_tokens=32768 覆盖 config 16384）
prompt = ANSWER_PROMPT.format(context=context, question=question)
# ANSWER_PROMPT 是从 EverCore 移植的 CoT 模板，要求 LLM 走 7 步推理

# 4. LLM 输出长 CoT，STEP 7 节是 "## STEP 7: FINAL ANSWER"
#    空结果 / 异常都重试：_ANSWER_RETRIES=5 次，退避 1.0 * 2**attempt
for attempt in range(5):
    response = await llm.chat([...], temperature=0.0, max_tokens=32768)
    answer = _extract_final_answer(response.content)
    if answer:
        break

# 5. 3 级 marker fallback + rsplit（取最后一个 marker，容忍 reasoning 里提前出现 marker）
def _extract_final_answer(raw: str) -> str:
    for marker in ("## STEP 7: FINAL ANSWER", "FINAL ANSWER:", "FINAL ANSWER"):
        if marker in raw:
            return raw.rsplit(marker, 1)[1].lstrip(":").strip()
    return raw.strip()
```

Stage 3 的 `aagentic_retrieve` 以 `top_n=response_top_k=10` 返回，`members` 通常已是 10 条；Stage 4 再按 `response_top_k=10` 截取拼 context。Context 中**有意不渲染原始毫秒 timestamp**（LLM 无法 parse 毫秒 epoch；
时间线索靠 `episode.content` 内的自然语言时间词，如「May 8, 2023」）。

### 输出：`answers.json`

扁平 list，所有对话的题混在一起：

```json
[
  {
    "question_id": "locomo_exp_user_0_qa0",
    "question": "When did Caroline go to the LGBTQ support group?",
    "answer": "May 8, 2023",
    "golden_answer": "May 7, 2023",
    "category": "2",
    "conversation_id": "locomo_exp_user_0",
    "formatted_context": "Episodes memories for...",
    "raw_response": "## STEP 1: ...## FINAL ANSWER: ...",
    "prompt_tokens": 1843,
    "completion_tokens": 412
  }
]
```

---

## 阶段 5 —— 评估（Evaluate）

LLM-as-judge 打分，每题 3 次独立投票、取多数派。

### 流程（每题一次）

```python
# 1. 过滤 adversarial 题（cat 5）
scored = [a for a in answers if str(a["category"]) not in dataset.filter_categories()]  # {"5"}

# 2. 调 everalgo.testing.allm_judge，每题 judge_runs=3 次（gpt-4o-mini @ temp 0.0）
result = await allm_judge(
    question=..., golden_answer=..., generated_answer=...,
    judge_prompt=JUDGE_PROMPT, judge_system_prompt=JUDGE_SYSTEM_PROMPT,
    llm=services.llm, num_runs=3, judge_model="gpt-4o-mini", judge_temperature=0.0,
)
# JUDGE_PROMPT 要求 LLM 输出 {"label": "CORRECT" | "WRONG"}
# per-question is_correct = 多数投票 sum(runs) > num_runs/2，存入 detailed_results

# 3. headline accuracy = mean-of-runs（不是 per-question 多数投票后再算比例）：
#    每个 judge run 独立算一遍全体 accuracy，再取 N 个 run 的平均
run_accs = [acc_of_run(i) for i in range(num_runs)]
accuracy = mean(run_accs); std_accuracy = std(run_accs)
correct = round(accuracy * total)
```

### 为什么 judge 用 gpt-4o-mini 不是 gpt-4.1-mini

- **答题模型**（gpt-4.1-mini）：能力强，专心生成
- **裁判模型**（gpt-4o-mini）：判断对错够用，价格约为答题模型的 1/10，速度更快
- 3 次投票降低随机性

### 输出：`eval_results.json`

```json
{
  "total_questions": 1540,
  "correct": 1391,
  "accuracy": 0.9032,
  "std_accuracy": 0.0041,
  "run_accuracies": [0.9039, 0.9006, 0.9052],
  "per_category": {
    "1": { "label": "single-hop",  "correct": 271, "total": 282, "accuracy": 0.9610, "run_accuracies": [] },
    "2": { "label": "temporal",    "correct": 280, "total": 321, "accuracy": 0.8723, "run_accuracies": [] },
    "3": { "label": "open-domain", "correct":  68, "total":  96, "accuracy": 0.7083, "run_accuracies": [] },
    "4": { "label": "multi-hop",   "correct": 772, "total": 841, "accuracy": 0.9180, "run_accuracies": [] }
  },
  "detailed_results": [
    {
      "...full answer record...": "...",
      "is_correct": true,
      "judge_runs": [true, true, false]
    }
  ]
}
```

`accuracy` 是 mean-of-runs（N 个 judge run 各自 accuracy 的平均）；`std_accuracy` / `run_accuracies` 量化 judge 非确定性。`detailed_results` 每题保留 `is_correct`（多数投票）+ `judge_runs`（每次投票），逐题可定位对错。

---

## 配置参数

所有可调参数在 `benchmarks/common/config.py::BenchmarkConfig`，默认值严格对齐
EverCore 评估框架：

| 类别 | 字段 | 默认值 |
|---|---|---|
| Retrieval | `retrieval_mode` | `"agentic"` |
| Retrieval | `use_hybrid_search` | `True` |
| Retrieval | `use_reranker` | `True` |
| Retrieval | `use_multi_query` | `True` |
| Top-N | `emb_recall_top_n` | 40 |
| Top-N | `reranker_top_n` | 20 |
| Top-N | `hybrid_emb_candidates` | 50 |
| Top-N | `hybrid_bm25_candidates` | 50 |
| Top-N | `hybrid_rrf_k` | 40 |
| Top-N | `multi_query_num` | 3 |
| Top-N | `response_top_k` | 10 |
| LLM | `llm_model` | `openai/gpt-4.1-mini` |
| LLM | `llm_temperature` | 0.3（Stage 4 答题时显式覆盖为 0.0）|
| LLM | `llm_max_tokens` | 16384（config 默认；stage4 答题时显式覆盖为 32768）|
| Judge | `judge_model` | `openai/gpt-4o-mini` |
| Judge | `judge_temperature` | 0.0 |
| Judge | `judge_runs` | 3 |
| Models | `embedding_model` | `Qwen/Qwen3-Embedding-4B` |
| Models | `embedding_dimensions` | 1024（Matryoshka 截断；对齐 EverCore）|
| Models | `reranker_model` | `Qwen/Qwen3-Reranker-4B` |
| Concurrency | `max_concurrent_convs` | 10（stage 1/2 跨对话并发上限）|
| Concurrency | `max_concurrent_qa` | 30（stage 3/4/5 QA 并发；main 用 50）|

---

## Smoke 模式

指定 `--smoke` 时各阶段的行为：

| 阶段 | 行为 |
|---|---|
| 1 抽取 | 截取 `conversations[:3]` —— 只对前 3 个对话做抽取 |
| 2 建索引 | 不主动截取，glob 自动只看到 Stage 1 写出的 3 个对话的 pkl |
| 3 检索 | 每个对话取 `qa_pairs[:10]` —— 共 30 题 |
| 4 回答 | 读 `search_results.json`，已只剩 30 条 |
| 5 评估 | 读 `answers.json`，过滤 cat 5 后再做 judge |

总规模 = 3 × 10 = **30 题**。耗时约 5~10 分钟，成本 < $1。

Smoke 的目的是验证 pipeline 不崩，**不是验证分数**（N=30 置信区间太宽，±17pp）。

---

## 不在 baseline 路径上的 EverAlgo 包

LoCoMo 会触发：`everalgo-boundary` / `everalgo-user-memory`（episode + atomic_fact）/ `everalgo-clustering`（stage1 `cluster_by_geometry` 分簇 + stage3 cluster-scoped 检索）/ `everalgo-rank`（agentic + hybrid + maxsim + cluster facade）。

不会触发：

- `everalgo-agent-memory` —— agent 轨迹场景，与对话场景正交
- `everalgo-parser` —— 多模态输入（实验性）
- `everalgo-knowledge` —— knowledge memory（实验性）

如果未来扩展数据集（如 LongMemEval、PersonaMem），在 `benchmarks/datasets/<name>/`
下放新的 `Dataset` 实现即可，不需要改 `common/`。
