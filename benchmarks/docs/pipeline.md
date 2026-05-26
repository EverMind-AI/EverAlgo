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
                                       └──────────────────────┴──→ 阶段 3 检索 ──→ memcell_ids
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
| 1 抽取 | `stage1_extract/` | 每对话一个 JSON | `boundary`, `user-memory` | OpenRouter (`gpt-4.1-mini`) |
| 2 建索引 | `stage2_index/` | 每对话一个 Pickle | — | DeepInfra (`Qwen3-Embedding-4B`) |
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
- **sender_id 命名**：`<name>_<conv_idx>`（如 `caroline_0`、`melanie_0`），mirror
  main `unique_id = f"{name.lower().replace(' ','_')}_{con_id}"`。同 conv 内的
  speaker 有稳定 disambiguation；跨 conv 重复无害（每 conv 单独建索引）。
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
   是 1~N 条消息的语义连续块。**Streaming batching**：每批 50 条新 msg + 上批未切
   `tail` 一起喂 LLM，LLM 返回 `(cells, tail)`；未切的 `tail` 留到下批，**最后
   一批传 `is_final=True` 强制 flush** tail 到最后一个 cell（mirror main
   `ConvMemCellExtractor` batch_size=50）。这样长 conv（500+ msg）不会让 LLM 单次
   认知过载漏切，避免出 154-msg outlier mega-cell。
2. **`EpisodeExtractor`**：对每个 MemCell 生成叙述 —— `subject`（短标题）+
   `content`（长叙述）。
3. **`AtomicFactExtractor`**：抽取该 MemCell 里的离散事实列表。**输入文本格式**
   `[<ts>] <speaker>: <content>`（mirror main `atomic_fact_extractor.py:255-262`）
   —— 每行带 timestamp，让 LLM 抽出的 fact 含时间锚定，例如
   `Caroline said she went to an LGBTQ support group yesterday (May 7, 2023)`。

### 跳过的步骤（与 EverCore baseline 对齐）

- `ForesightExtractor` —— `enable_foresight_extraction=False`
- `ProfileExtractor` —— `enable_profile_extraction=False`
- `Clustering` —— EverCore 默认开启，但关闭 Profile 后无下游消费者，省略

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
        "sender_id": "u_caroline",
        "sender_name": "Caroline"
      }
    ],
    "episode": {
      "subject": "Caroline reconnects with Melanie",
      "content": "On May 8 2023, Caroline and Melanie reconnected..."
    },
    "atomic_facts": [
      { "fact": "Caroline went to LGBTQ support group" }
    ]
  }
]
```

会话 ID 由文件名 `memcells_conv_<i>.json` 隐式承载，无需在记录内重复。

419 条消息 → 约 30~50 个 MemCell。

### 并发

- 跨对话：`asyncio.Semaphore(max_concurrent_convs=10)` 控制并发对话数
- 单对话内：跨 MemCell 用 `asyncio.gather` 并行，全局 `Semaphore(30)` 限制 LLM 并发上限

---

## 阶段 2 —— 建索引（Index）

对每个对话的 MemCell 集合分别建两套检索结构。

### BM25（关键词检索）

```python
# 文本选择走 short-circuit 路径，对齐 EverCore main：
#   atomic_facts 非空 → 只拼 facts（更精准，避免 episode 长文稀释关键词）
#   否则 fallback   → subject*3 + summary*2 + content（按重要性加权重复）
if atomic_facts:
    text = " ".join(af["fact"] for af in atomic_facts)
else:
    text = " ".join([subject]*3 + [summary]*2 + [content])
# 分词：lowercase → 去停词 → Porter stemmer
tokens = ["carolin", "reconnect", "melani", "lgbtq", "support", ...]
# 喂给 rank_bm25
bm25 = BM25Okapi(all_tokenized_docs)
```

落盘 `bm25_conv_<i>.pkl`，约 200 KB。纯本地计算。

### Embedding（语义检索）

对每个 MemCell 用**一次 API 调用**批量 embed。**字段选择走 short-circuit，对齐
EverCore main**：

```python
if atomic_facts:
    # 主路径：每个 fact 独立 embed（Stage 3 MaxSim 用），不混入 episode 长文
    texts_to_embed = [af["fact"] for af in atomic_facts]
else:
    # fallback：atomic_facts 为空时才 embed 三字段
    texts_to_embed = [subject, summary, content]   # 各自非空字段
vectors = await deepinfra.embed(texts_to_embed, dimensions=1024)
```

每个 MemCell 存成（**两种 shape 之一**，取决于走主路径还是 fallback 路径）：

```python
# 主路径（占 99.6%）：
{
  "doc": {完整 memcell 字典},
  "embeddings": {
    "atomic_facts": [np.ndarray(1024), np.ndarray(1024), ...]  # 每个事实一个向量
  }
}

# fallback 路径（atomic_facts 为空时，约占 0.4%）：
{
  "doc": {完整 memcell 字典},
  "embeddings": {
    "subject": np.ndarray(1024),
    "summary": np.ndarray(1024),
    "content": np.ndarray(1024),
  }
}
```

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

```
question
  ↓
[第 1 轮] hybrid_search_with_rrf
  ├─ BM25 检索（关键词匹配）       → Top 50
  ├─ Embedding MaxSim 检索（语义） → Top 50
  └─ RRF 融合（k=60）              → Top 20
  ↓
reranker_search
  └─ Qwen3-Reranker-4B 重排 Top 20 → Top 10
  ↓
[充分性检查] LLM (gpt-4.1-mini)
  「看这 10 条 doc，能回答 question 吗？」
  ↓
  ├─ ✓ 充分    → 返回 Top 10
  │
  └─ ✗ 不充分  → 进入第 2 轮
       ↓
       [多查询改写] LLM 生成 3 个 refined queries
       （基于 missing_info + key_info_found）
       ↓
       [第 2 轮] 3 个 query 并行 hybrid search
         每个产出 Top 50
       ↓
       multi_rrf_fusion 融合 3 路结果 → Top 40
       ↓
       去重合并第 1 轮 Top 20 + 第 2 轮 → 40 个 doc
       ↓
       reranker_search 再排 → 最终 Top 20
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
      "memcell_ids": ["3", "7", "12", "18", "21"],
      "original_qa": { /* 完整 QA dict */ },
      "retrieval_metadata": {
        "is_multi_round": false,
        "is_sufficient": true,
        "round1_count": 20,
        "round1_reranked_count": 10,
        "total_latency_ms": 5234,
        "prompt_tokens": 1843,
        "completion_tokens": 122
      }
    }
  ]
}
```

`memcell_ids` 是会话内的本地序号（如 `"0"`、`"1"` …），不是全局唯一 id。

### 关键参数（对齐 EverCore）

| 参数 | 值 | 含义 |
|---|---|---|
| `hybrid_emb_candidates` | 50 | 每路 emb 检索召回数 |
| `hybrid_bm25_candidates` | 50 | 每路 BM25 召回数 |
| `hybrid_rrf_k` | 40 | RRF 融合常数 |
| `reranker_top_n` | 20 | 最终 Top |
| `multi_query_num` | 3 | 第 2 轮改写数 |
| `max_concurrent_qa` | 30 | QA 并发数（stage 3/4/5）|

---

## 阶段 4 —— 回答（Answer）

**输入**：

- `stage3_search/search_results.json`（每题 Top 20 memcell_ids）
- `stage1_extract/memcells_*.json`（按 memcell_id 反查完整 MemCell）

### 流程（每题一次）

```python
# 1. 从 memcell_ids 反查完整 memcell
top_memcells = [memcells_map[conv_id][mc_id] for mc_id in memcell_ids[:10]]

# 2. 拼 context（对齐 EverCore 格式：每条 doc 间用 "\n---\n\n" 明确分块）
context = f"""Episodes memories for conversation between Caroline and Melanie:

Caroline reconnects with Melanie: On May 8 2023, Caroline...
---

Discussion about LGBTQ support: Caroline shared that...
---

..."""

# 3. 喂给 LLM（gpt-4.1-mini @ temperature=0.0，覆盖 config 默认 0.3）
prompt = ANSWER_PROMPT.format(context=context, question=question)
# ANSWER_PROMPT 是从 EverCore 移植的 CoT 模板，要求 LLM 走 7 步推理

# 4. LLM 输出长 CoT，最后是 "## FINAL ANSWER: ..."
#    空结果重试（对齐 EverCore）：偶发 OpenRouter 空响应时最多重试 3 次
for _ in range(3):
    response = await llm.chat([...], temperature=0.0)
    answer = extract_final_answer(response.content)
    if answer:
        break

# 5. 直接 split 提取 FINAL ANSWER 后内容（对齐 EverCore，不要求 ## 前缀，不截断换行）
def extract_final_answer(raw: str) -> str:
    if "FINAL ANSWER:" in raw:
        return raw.split("FINAL ANSWER:")[1].strip()
    return raw.strip()
```

Stage 3 返回 Top 20，Stage 4 只取前 `response_top_k=10` 拼 context（对齐 EverCore
默认）。Context 中**有意不渲染原始毫秒 timestamp**（LLM 无法 parse 毫秒 epoch；
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
if qa.category in dataset.filter_categories():  # = {"5"}
    skip

# 2. 调 LLM judge，3 次并发
prompt = JUDGE_PROMPT.format(
    question=question,
    gold_answer=golden_answer,
    response=generated_answer,
)
# JUDGE_PROMPT 要求 LLM 输出 {"label": "CORRECT" | "WRONG"}

results = await asyncio.gather(*[
    llm.chat(prompt, model="gpt-4o-mini", temperature=0)
    for _ in range(3)
])

# 3. 解析每次的 label，多数投票（≥2 票为 correct 即视为 correct）
votes = [parse_label(r) for r in results]
is_correct = sum(votes) > 1.5
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
  "per_category": {
    "1": { "label": "single-hop",  "correct": 271, "total": 282, "accuracy": 0.9610 },
    "2": { "label": "temporal",    "correct": 280, "total": 321, "accuracy": 0.8723 },
    "3": { "label": "open-domain", "correct":  68, "total":  96, "accuracy": 0.7083 },
    "4": { "label": "multi-hop",   "correct": 772, "total": 841, "accuracy": 0.9180 }
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

`detailed_results` 保留每题 3 次投票，逐题查看可定位为什么某题算对/错。

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
| LLM | `llm_max_tokens` | 32768（对齐 EverCore main，给长 CoT 留余量）|
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

LoCoMo 不会触发：

- `everalgo-clustering` —— 需要 Profile 输出作下游消费者
- `everalgo-agent-memory` —— agent 轨迹场景，与对话场景正交
- `everalgo-parser` —— 多模态输入（实验性）
- `everalgo-knowledge` —— knowledge memory（实验性）

如果未来扩展数据集（如 LongMemEval、PersonaMem），在 `benchmarks/datasets/<name>/`
下放新的 `Dataset` 实现即可，不需要改 `common/`。
