# 基准测试流水线

`benchmarks/` 目录下 LoCoMo 基准测试的端到端流程：7 个串行阶段，每个阶段消费前一阶段落盘的中间结果。中间产物以 Pickle / JSON 形式存放，所以任一阶段都可通过 `--stages` 单独重跑。

## 总览

```
LoCoMo JSON  ───┐
                ├─→ 阶段 1 抽取基础  ──→ MemCells + Episodes + Clusters (json)
                                          │
                                          ├─→ 阶段 2 反思  ──→ 合并后的 Episodes (json, 可选)
                                          │
                                          ├─→ 阶段 3 充实  ──→ AtomicFacts + embeddings (json)
                                          │
                                          ├─→ 阶段 4 建索引 ──→ BM25 + Emb (pkl)
                                          │                       │
                                          └──────────────────────┴──→ 阶段 5 检索 ──→ members
                                                                                        │
                                          ┌────────────────────────────────────────────┘
                                          ↓
                                     阶段 6 回答 ──→ generated_answers
                                                          │
                                                          ↓
                                                    阶段 7 评估 ──→ accuracy
                                                                       │
                                                                       ↓
                                                                 report.{txt,json}
```

| 阶段 | 输出目录 | 格式 | 使用的 EverAlgo 包 | 外部服务 |
|---|---|---|---|---|
| 1 抽取基础 | `stage1_extract_base/` | 每对话 4 个 JSON（memcells / episodes / clusters / stats） | `boundary`, `user-memory` (episode), `clustering` | OpenRouter (`gpt-4.1-mini`), DeepInfra (`Qwen3-Embedding-4B`) |
| 2 反思 | `stage2_reflect/` | 每对话 1 个 JSON（合并后 episodes） | `user-memory` | OpenRouter (`gpt-4.1-mini`) |
| 3 充实 | `stage3_enrich/` | 每对话 1 个 JSON（atomic_facts） | `user-memory` (atomic_fact) | OpenRouter (`gpt-4.1-mini`), DeepInfra (`Qwen3-Embedding-4B`) |
| 4 建索引 | `stage4_index/` | 每对话 3 个 Pickle（bm25 / emb / cluster_index） | `clustering`（仅 `Cluster` 类型） | 无；复用阶段 1/3 的向量 |
| 5 检索 | `stage5_search/` | 单个 JSON | `rank`（融合 + MaxSim）| OpenRouter + DeepInfra (`Qwen3-Reranker-4B`) |
| 6 回答 | `stage6_answer/` | 单个 JSON | — | OpenRouter (`gpt-4.1-mini`) |
| 7 评估 | `stage7_evaluate/` | 单个 JSON | — | OpenRouter (`gpt-4o-mini`) |

---

## 数据预处理（Loader）

`LocomoDataset.load_conversations()` 把 LoCoMo `locomo10.json` 转成统一的 `Conversation { id, speakers, messages }` 值类型，几个关键处理对齐 the upstream reference：

- **message timestamp**：LoCoMo 只有 session 级 `session_<N>_date_time`，没有 per-message timestamp。loader 给同一 session 内每条 message 派 `session_time + i*30s` 递增的毫秒 epoch（mirror main `stage1_memcells_extraction.py:114-123`）。让 BoundaryDetector 看到 monotonically advancing timestamps，避免同 session 内全部 message 同 ts 让 LLM 误判为「并发说话」漏切。
- **sender_id 命名**：`f"{speaker.lower().replace(' ','_')}_{conv_id}"`，其中 `conv_id` 是完整字符串 `locomo_exp_user_<i>`，所以实际形如 `caroline_locomo_exp_user_0`（`loader.py:77`）。同 conv 内 speaker 稳定区分；跨 conv 因 conv_id 不同天然唯一。
- **img_url 拼接**：LoCoMo 5882 条 message 里 910 条（15.5%）含 `img_url` + `blip_caption`。loader 把图片描述拼到 content 前：`"[<speaker> shared an image: <blip_caption>] <text>"`（mirror main `stage1_memcells_extraction.py:134-140`）。EvalQA 里 39.4% 的题 evidence 含图片消息 —— 不拼丢的是真信号。
- **cat 5 过滤**：LoCoMo 的 `category=5` 是 adversarial（设计上无答案的对抗题）。loader 在 `load_qa_pairs` 层直接跳过，下游 search / answer / judge 不浪费算力。

输出的 `Message` 字段：`id` (= raw `dia_id`) / `role` (= `"user"`，LoCoMo 无 system 区分) / `content` (str，含 image caption 前缀) / `timestamp` (ms epoch) / `sender_id` / `sender_name`。

---

## 阶段 1 —— 抽取基础（Extract Base）

**输入**：`LocomoDataset.load_conversations()` 返回的完整对话。LoCoMo 的 `conv_0` 含 419 条消息，跨多个会话日。

**两步流程（每对话一次）**：

```
原始消息
    ↓
[1] BoundaryDetector       ← LLM 判断切片位置
    ↓
list[MemCell]              ← 几十个语义连续的记忆单元
    ↓
[2] EpisodeExtractor       ← 每个 MemCell 调一次 LLM
    ↓
Episode { subject, episode }   ← episode 是叙述正文（字段名为 episode，非 content）
    ↓
[3] Embedding              ← 对 episode text + subject 做 embedding
    ↓
[4] Clustering             ← cluster_by_geometry，基于 episode embedding + 时间窗
```

1. **`BoundaryDetector`**（everalgo-user-memory）：把消息序列切成多个 MemCell，每个是 1~N 条消息的语义连续块。**逐消息增量检测**：调 `adetect_step` 一条条喂（`extract.py:_detect_all_boundaries`），caller 侧只负责 front-2-buffer（前 2 条不触发 LLM）和流末把残余 `tail` flush 成最后一个 cell；smart-mask 阈值门控、masking、force-split（`hard_token_limit=8192` / `hard_message_limit=50`）与 cut-and-bridge 状态转移都封装在 `adetect_step` 内。注意 50 是内部 force-split 的硬上限，**不是「每批 50 条」的批大小**。
2. **`EpisodeExtractor`**：对每个 MemCell 生成叙述 —— `subject`（短标题）+ `episode`（长叙述正文）。
3. **Embedding**：对 episode text 和 subject 分别做 embedding，结果存入 `episodes_conv_<i>.json`。
4. **Clustering**：对每个 Episode 用其 embedding 增量调 `everalgo.clustering.cluster_by_geometry`（cosine + 时间窗，`cluster_similarity_threshold=0.70` / `cluster_max_time_gap_days=7.0`）分簇，落 `clusters_conv_<i>.json`。

注意：**AtomicFactExtractor 在此阶段不再执行**，已移至阶段 3（充实）。

未运行的提取器：`ForesightExtractor` / `ProfileExtractor` 在 stage1 代码里**从未被调用**（也不存在 `enable_foresight_extraction` / `enable_profile_extraction` 这两个开关）。

### 输出：每对话 4 个文件

#### `memcells_conv_<i>.json` —— 纯消息分组

仅包含边界检测产出的 MemCell，不嵌套 episode 或 atomic_facts：

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
    ]
  }
]
```

#### `episodes_conv_<i>.json` —— Episode 实体

每个 Episode 包含叙述正文、主题及对应的 embedding 向量：

```json
[
  {
    "id": "0",
    "owner_id": null,
    "memcell_ids": ["0"],
    "subject": "Caroline reconnects with Melanie",
    "episode": "2023-05-08 03:56 UTC — Caroline and Melanie reconnected after years apart and caught up on each other's lives...",
    "summary": "Caroline and Melanie reconnected after years apart.",
    "timestamp": 1683525360000,
    "embeddings": {
      "episode": [0.01, -0.02, "..."],
      "subject": [0.03, 0.01, "..."]
    }
  }
]
```

字段名是 **`episode`**（算法命名），不是 `content`；`summary` 是必填展示摘要。正文和摘要均来自 LLM，代码不拼接时间文本（`everalgo.user_memory.episode._build_episode`）。`timestamp` 字段（ms epoch）另行保存该切片的关闭时间，但**不进答题上下文**——答题阶段只取 `subject` 与 `episode` 两个字段（见 `common/stages/answer.py`），所以正文中的时间线索必须由模型生成。

#### `clusters_conv_<i>.json` —— 簇信息

```json
{
  "clusters": [
    {
      "id": "cluster_0",
      "centroid": [0.01, -0.02, "..."],
      "count": 3,
      "last_ts": 1683525360000,
      "episode_ids": ["0", "3", "7"],
      "preview": ["Episode body..."]
    }
  ],
  "episode_to_cluster": {
    "0": "cluster_0",
    "3": "cluster_0",
    "7": "cluster_0"
  }
}
```

簇使用 `episode_ids`（不是 `memcell_ids`），`episode_to_cluster` 是反向映射。

会话 ID 由文件名 `*_conv_<i>.json` 隐式承载，无需在记录内重复。419 条消息 → 约 30~50 个 MemCell/Episode。

### 并发

- 跨对话：`asyncio.Semaphore(max_concurrent_convs=10)` 控制并发对话数
- 单对话内：跨 MemCell 用 `asyncio.gather` 并行，全局 `mc_sem = Semaphore(20)`（硬编码，`extract.py`）限制 MemCell LLM 并发上限

---

## 阶段 2 —— 反思（Reflect）

**可选阶段**，由 `enable_reflection=true` 控制（默认关闭）。

**输入**：阶段 1 产出的 `episodes_conv_<i>.json` + `clusters_conv_<i>.json`。

**流程**：对含 2 个及以上成员的簇，调 LLM 将簇内多个 episode 合并成一个更精炼的叙述。单成员簇保持不变。

**输出**：`stage2_reflect/episodes_conv_<i>.json` —— 与阶段 1 格式相同，但部分 episode 已被合并版替换。下游阶段（3~7）优先读取 stage2 的输出，若 stage2 未运行则 fallback 到 stage1 产物。

---

## 阶段 3 —— 充实（Enrich）

**输入**：最终 episodes（stage2 有则用 stage2，否则用 stage1）。

**流程（每对话一次）**：

```
Episode { subject, episode }
    ↓
[1] AtomicFactExtractor    ← 每个 Episode 调一次 LLM
    ↓
list[AtomicFact { content, episode_id }]
    ↓
[2] Embedding              ← 对每个 atomic fact 做 embedding
```

1. **`AtomicFactExtractor`**：从 Episode 正文（`episode` 字段）抽取离散事实列表。每个事实关联其来源 `episode_id`。
2. **Embedding**：对每个 atomic fact 做 embedding，结果存入输出文件。

**如果 atomic_facts 为空则 raise error** —— 不存在 fallback 到 episode text 的逻辑。

### 输出：`atomic_facts_conv_<i>.json`

```json
[
  {
    "id": "af_0_0",
    "content": "Caroline went to LGBTQ support group",
    "episode_id": "0",
    "timestamp": 1683525360000,
    "embeddings": [0.03, -0.01, "..."]
  },
  {
    "id": "af_0_1",
    "content": "Caroline and Melanie reconnected on May 8 2023",
    "episode_id": "0",
    "timestamp": 1683525360000,
    "embeddings": [0.02, 0.04, "..."]
  }
]
```

每个 AtomicFact 的字段名为 **`content`**（事实内容），并携带 `episode_id` 指向来源 Episode。

---

## 阶段 4 —— 建索引（Index）

对每个对话的实体集合建立 BM25、Embedding 和 Cluster 三套本地索引。该阶段只读取已计算的向量，不调用 embedding 服务。

### BM25（关键词检索）

**fact 级索引**：每个 atomic fact 的 `content`、对应 Episode 的 `subject`、Episode 正文前 200 个字符各自 tokenize 成一条独立 BM25 document（不拼接、不加权重复），`fact_to_doc_idx` 把每行映射回父 Episode。检索时取一个 doc 所有行的最高分（MaxSim 聚合）。

```python
# extract_searchable_units：每个 unit 一行 doc
units = list(atomic_fact_contents)   # 每个 fact 一行
if subject: units.append(subject)
if episode: units.append(episode[:200])
# 不使用 summary
# _tokenize：lower -> word_tokenize -> 保留 alpha 且 len>=2 且非停词 -> PorterStemmer
tokens = ["carolin", "reconnect", "melani", "lgbtq", "support"]
bm25 = BM25Okapi(fact_corpus)    # 所有 unit 跨所有 Episode 摊平成一个语料
```

落盘 `bm25_conv_<i>.pkl`，包含 `bm25`、`docs`、`fact_to_doc_idx`、`index_type` 四个键，约 200 KB。纯本地计算。

### Embedding（语义检索）

Stage 4 不执行 embedding。它直接读取 Stage 1 的 Episode 正文/subject 向量和 Stage 3 的 AtomicFact 向量，并转换为 NumPy 数组写入索引。

索引字段如下；这些向量均由上游阶段预先计算：

```python
# 从 atomic_facts_conv_<i>.json 加载 fact embeddings（stage 3 已预计算）
# 从 episodes_conv_<i>.json 加载 subject embedding（stage 1 已预计算）
# 不存在 summary 或 content fallback
```

每个 Episode 存成 `{"doc_id": episode_id, "embeddings": {...}}`，`embeddings` 可含 `atomic_facts: [ndarray, ...]`、`subject: ndarray` 和 `episode: ndarray`。Stage 5 当前只用 AtomicFact 与 subject 做 MaxSim；Episode 正文向量虽被保存，但不参与当前 dense 打分。

落盘 `emb_conv_<i>.pkl`，约 7~10 MB（每个对话）。**这是磁盘占用的大头。**

### 为什么 dim=1024 不是 2560

Qwen3-Embedding-4B 是 Matryoshka 模型。向量在 Stage 1/3 请求时通过 `dimensions=1024` 截断，Stage 4 仅将这些既有向量装入索引。该维度对齐 the upstream reference 的 `HybridVectorizeConfig.dimensions=1024`。

### 为什么 atomic_facts 用一组向量、而非合并成一个

Stage 5 用 MaxSim 策略：对一个 query embedding，跟 Episode 的所有 atomic_facts 向量逐个算 cosine 相似度，取**最大值**作为这个 Episode 的整体得分。

直觉：「这个 Episode 里只要有任何一个事实和查询语义相关，整个 Episode 就值得检索出来」，比合并平均更精准。

---

## 阶段 5 —— 检索（Search，最复杂）

Agentic 多轮检索，每题独立流程：

整个 stage5 委托 `everalgo.rank.aagentic_retrieve`，benchmark 只注入 BM25 / Embedding 检索闭包和 reranker；充分性检查、多查询改写的 prompt 全来自 `everalgo.rank.prompts`。

```
question
  ↓  (everalgo.rank.aagentic_retrieve)
[Step 1] 双路召回（fact 级，exhaustive）
  ├─ BM25 exhaustive recall（全量 atomic_facts + subject）
  └─ Embedding MaxSim recall（全量 atomic_facts + subject）
  ↓
[Step 2] MaxSim 聚合（fact → episode），每路取 top 50 episodes
  ↓
[Step 3] RRF 融合 (k=40)，不截断 —— 保留全部去重 episodes
  ↓
[Step 4] Cluster 选择：first-hit 扫描完整 RRF 结果，选 top cluster_top_k=10 个簇，
         展开全部簇内成员 episodes
  ↓
[Step 5] Round 1 rerank（cross-encoder 对 episode 全文重排），取 top round1_rerank_top_n=10
  ↓
[Step 6] 充分性检查（LLM）
  ↓
  ├─ ✓ 充分    → 返回 top_n = response_top_k = 10
  │
  └─ ✗ 不充分  → 进入 Round 2
       ↓
       [Step 7] 多查询改写 → hybrid_full 全语料检索（逃逸簇范围），cap round2_cap=40
       ↓
       [Step 8] 合并 + 最终 rerank → top 10 episode_ids
```

### Cluster 路径详解

- **Step 1-2**：在 atomic fact 级别做双路（BM25 + Embedding）exhaustive 召回，然后 MaxSim 聚合到 episode 级别，每路取 top 50 episodes。
- **Step 3**：RRF 融合（k=40），不做截断，保留所有去重后的 episodes。
- **Step 4**：first-hit 扫描完整 RRF 结果列表，按 episode 首次命中顺序选择 top 10 个 cluster，展开这些 cluster 的全部成员 episode。
- **Step 5**：cross-encoder reranker 对展开后的 episode 全文重排，取 top 10。
- **Step 6-8**：LLM 充分性检查 → 若不充分则 Round 2 多查询改写，**hybrid_full 在全语料上检索（逃逸 cluster 范围）**，最终 rerank 取 top 10。

### 充分性检查（Sufficiency Check）

LLM 接收 query + Top 10 docs，输出结构化 JSON：

- `is_sufficient: bool`
- `reasoning: str`（解释）
- `missing_info: list[str]`（如「需要知道事件发生的具体日期」）
- `key_information_found: list[str]`（如「已知 Caroline 是 counselor」）

充分 → 跳过第 2 轮省 LLM 调用；不充分 → 用 `missing_info` 指导第 2 轮 query 改写。

### 多查询改写（Multi-Query）

LLM 输入：原 query + missing_info + key_information_found，输出 3 个 refined queries，每个聚焦不同的缺失维度。例如：

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

字段名是 **`members`**（不是 `episode_ids`），值为会话内本地序号（如 `"0"`、`"1"` ...），不是全局唯一 id。`retrieval_metadata` 的 `is_multi_round` / `reasoning` / `missing_info` / `query_strategy` 等直接透传自 `everalgo.rank` 的 `AgenticDecision`。

### 关键参数（aligned with the upstream reference）

| 参数 | 值 | 含义 |
|---|---|---|
| `hybrid_emb_candidates` | 50 | 每路 emb 检索召回数 |
| `hybrid_bm25_candidates` | 50 | 每路 BM25 召回数 |
| `hybrid_rrf_k` | 40 | RRF 融合常数（Level-1 hybrid 与 Round-2 multi-query 共用；库 `RankConfig.rrf_k` 默认 60 在此被 override） |
| `cluster_top_k` | 10 | 簇选择数量 |
| `round1_rerank_top_n` | 10 | R1 rerank 后进充分性检查的窗口 |
| `response_top_k` | 10 | 最终返回 Top（拼 context 用） |
| `multi_query_num` | 3 | 第 2 轮改写数 |
| `max_concurrent_qa` | 20 | QA 并发数（stage 5-7；来自 `benchmarks/config.toml`）|

---

## 阶段 6 —— 回答（Answer）

**输入**：

- `stage5_search/search_results.json`（每题的 `members`，即检索出的 episode 本地 id 列表）
- `stage3_enrich/episodes_conv_*.json`（透传最终 Episode，按 member id 反查）

### 流程（每题一次）

```python
# 1. 从 members 反查完整 episode，取前 response_top_k=10
ep_ids = item["members"]
top_episodes = [episodes_map[conv_id][ep_id] for ep_id in ep_ids[:10] if ep_id in episodes_map[conv_id]]

# 2. 拼 context（每条 doc 间用 "\n---\n\n" 明确分块）
context = f"""Episodes memories for conversation between Caroline and Melanie:

Caroline reconnects with Melanie: 2023-05-08 03:56 UTC — Caroline and Melanie reconnected after years apart...
---

Discussion about LGBTQ support: Caroline shared that...
---

..."""

# 3. 喂给 LLM（answer_model @ temperature=0.0 覆盖 config 0.3；max_tokens=32768 覆盖 config 16384）
prompt = ANSWER_PROMPT.format(context=context, question=question)
# ANSWER_PROMPT 是从 the upstream reference 移植的 CoT 模板，要求 LLM 走 7 步推理

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

Stage 5 的 `aagentic_retrieve` 以 `top_n=response_top_k=10` 返回，`members` 通常已是 10 条；Stage 6 再按 `response_top_k=10` 截取拼 context。Context 中**有意不渲染原始毫秒 timestamp**；时间线索只来自 LLM 生成的 `episode` 正文，代码不会给正文自动添加 UTC 时间戳。

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

## 阶段 7 —— 评估（Evaluate）

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

默认值在 `benchmarks/config.toml`（单一事实源），`BenchmarkConfig` 每次运行从该文件读取。`--config <name>` 可加载自定义 TOML 覆盖默认值。参数对齐 the upstream evaluation reference：

| 类别 | 字段 | 默认值 |
|---|---|---|
| Retrieval | `retrieval_mode` | `"agentic"` |
| Top-N | `hybrid_emb_candidates` | 50 |
| Top-N | `hybrid_bm25_candidates` | 50 |
| Top-N | `hybrid_rrf_k` | 40 |
| Top-N | `multi_query_num` | 3 |
| Top-N | `response_top_k` | 10 |
| Top-N | `round1_rerank_top_n` | 10 |
| LLM | `extract_model` | `openai/gpt-4.1-mini` |
| LLM | `answer_model` | `openai/gpt-4.1-mini` |
| LLM | `llm_temperature` | 0.3（Stage 6 答题时显式覆盖为 0.0）|
| LLM | `llm_max_tokens` | 16384（config 默认；stage 6 答题时显式覆盖为 32768）|
| Reflection | `enable_reflection` | `False` |
| Clustering | `cluster_similarity_threshold` | 0.70 |
| Clustering | `cluster_max_time_gap_days` | 7.0 |
| Cluster retrieval | `cluster_top_k` | 10 |
| Judge | `judge_model` | `openai/gpt-4o-mini` |
| Judge | `judge_temperature` | 0.0 |
| Judge | `judge_runs` | 3 |
| Models | `embedding_model` | `Qwen/Qwen3-Embedding-4B` |
| Models | `embedding_dimensions` | 1024（Matryoshka 截断；aligned with the upstream reference）|
| Models | `reranker_model` | `Qwen/Qwen3-Reranker-4B` |
| Concurrency | `max_concurrent_convs` | 10（stage 1-4 跨对话并发上限）|
| Concurrency | `max_concurrent_qa` | 20（stage 5-7 QA 并发）|
| Session filter | `session_filter` | `None`（运行全部 session；可按 `{conv_idx: [session_ids]}` 过滤） |

### TOML 配置加载

通过 `--config <name>` 从 `benchmarks/<name>.toml` 加载配置，未设置的字段 fallback 到默认值：

```toml
# benchmarks/fast.toml
extract_model = "openai/gpt-4.1-mini"
answer_model = "openai/gpt-4.1-nano"
enable_reflection = false
hybrid_emb_candidates = 30
hybrid_bm25_candidates = 30

[session_filter]
5 = [0, 1, 2]
```

---

## Smoke 模式

指定 `--smoke` 时各阶段的行为：

| 阶段 | 行为 |
|---|---|
| 1 抽取基础 | 截取 `conversations[:1]` —— 只对前 1 个对话做抽取（`smoke_conv_limit=1`） |
| 2 反思 | 不主动截取，glob 自动只看到 Stage 1 写出的 1 个对话 |
| 3 充实 | 不主动截取，glob 自动只看到 1 个对话 |
| 4 建索引 | 不主动截取，glob 自动只看到 1 个对话的 pkl |
| 5 检索 | 每个对话取 `qa_pairs[:10]` —— 共 10 题 |
| 6 回答 | 读 `search_results.json`，已只剩 10 条 |
| 7 评估 | 读 `answers.json`，过滤 cat 5 后再做 judge |

总规模 = 1 x 10 = **10 题**。耗时约 3~5 分钟，成本 < $0.50。

Smoke 的目的是验证 pipeline 不崩，**不是验证分数**（N=10 置信区间极宽）。

---

## 不在 baseline 路径上的 EverAlgo 包

LoCoMo 会触发：`everalgo-boundary` / `everalgo-user-memory`（episode + atomic_fact）/ `everalgo-clustering`（stage1 `cluster_by_geometry` 分簇 + stage5 cluster-scoped 检索）/ `everalgo-rank`（agentic + hybrid + maxsim + cluster facade）。

不会触发：

- `everalgo-agent-memory` —— agent 轨迹场景，与对话场景正交
- `everalgo-parser` —— 多模态文件和 URL 输入；仅 video 路径尚未实现
- `everalgo-knowledge` —— knowledge memory 抽取

如果未来扩展数据集（如 LongMemEval、PersonaMem），在 `benchmarks/datasets/<name>/` 下放新的 `Dataset` 实现即可，不需要改 `common/`。
