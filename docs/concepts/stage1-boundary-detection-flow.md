# Boundary Detection Flow

Benchmark（stage 1）的边界检测流程。将一段对话的消息列表切分为多个 MemCell（记忆单元），每个 MemCell 代表一个连贯的对话片段。

Scope：per-conversation。stage 1 逐 conversation 处理。

---

## Step 1：消息转换

**操作对象**：LoCoMo 数据集的原始对话消息列表

**过程**（`extract.py`）：

1. 遍历对话的所有消息
2. 过滤：只保留 `role` 为 `user` 或 `assistant` 的消息（`extract.py:80`）。LoCoMo 数据集所有消息的 role 均为 `"user"`（`loader.py:72`），无消息被过滤
3. 每条转为 `ChatMessage`：`id`、`role`、`content`、`timestamp`（ms epoch）、`sender_id`、`sender_name`

**输出**：`list[ChatMessage]`，LoCoMo 场景下等于全部消息

---

## Step 2：逐条边界检测（Incremental Loop）

**操作对象**：Step 1 的 `list[ChatMessage]`

**过程**（`extract.py:225-286`，`user_memory/boundary.py:66-143`）：

对每条消息执行以下循环：

**2a. Front-2 Buffer**（`extract.py:273-276`）：
- 前 2 条消息直接加入 `history`，不调 LLM
- 从第 3 条消息开始，每条触发一次 `adetect_step`

**2b. Force-Split 短路**（`boundary.py:113-121`）：
- 如果 `count_tokens(history + new) >= 8192` 或 `len(history) + 1 >= 50`，且 `len(history) >= 2`
- 直接将 `history` 关闭为一个 MemCell，不调 LLM
- 新 history：smart-mask 开时为 `[history[-1], new]`（bridge），否则为 `[new]`

**2c. Smart-Mask**（`boundary.py:111, 125`）：
- 条件：`smart_mask=True`（默认）且 `len(history) > 5`（`smart_mask_threshold=5`）
- 生效时：LLM 只看 `history[:-1]`（遮掉最后一条之前的所有历史），降低 token 消耗
- 未达阈值时：LLM 看完整 `history`

**2d. 时间间隔计算**（`boundary.py:126, 174-204`）：
- 计算 `history` 最后一条和 `new` 之间的时间差
- 分桶：<60s「immediate response」、<1h「N minutes」、<1day「N hours」、≥1day「N days」
- 作为 `time_gap_info` 注入 prompt

**2e. LLM 判断**（`boundary.py:127-133`）：
- 调用 `adetect_boundary_step(masked_history, [new], llm=..., prompt=..., time_gap_info=...)`
- LLM 返回 JSON：`{should_end: bool, reasoning: str, confidence: float, topic_summary: str}`
- `gpt-4.1-mini`，temperature 由 `adetect_boundary_step` 内部决定

**2f. 状态转移**（`boundary.py:135-143`）：
- `should_end=True`：`history` 关闭为 MemCell，新 history 为 `[history[-1], new]`（smart-mask bridge）或 `[new]`（clean cut）
- `should_end=False`：`history = [*history, new]`，继续累积

**2g. 尾部 Flush**（`extract.py:282-284`）：
- 所有消息处理完后，`history` 中剩余消息强制关闭为最后一个 MemCell

**输出**：`list[MemCell]`，每个 MemCell 包含一段连贯对话的 `items: list[ChatMessage]` 和 `timestamp`（最后一条消息的时间戳）

---

## Step 3：Per-MemCell 抽取（Episode + AtomicFact + Embedding）

**操作对象**：Step 2 产出的每个 MemCell

**过程**（`extract.py:137-222`，`extract.py:349-356` 并行调度）：

所有 MemCell 通过 `asyncio.gather` 并行处理，受 semaphore 控制并发（`extract.py:349-356`）。每个 MemCell 执行：

**3a. Episode 抽取**（`extract.py:159-166`）：
- `EpisodeExtractor(llm=llm).aextract(mc)` — 将 MemCell 的原始消息压缩为叙事性摘要
- 产出 `episode.subject`（标题）和 `episode.episode`（正文 body）
- body 为空 raise `ValueError`（fail-loud）
- JSON 解析失败最多重试 `max_attempts=5` 次

**3b. 并行：AtomicFact 抽取 + Episode Body Embedding**（`extract.py:168-183`）：
- `AtomicFactExtractor(llm=llm).aextract_from_text(episode_body)` — 从 episode body 中提取可验证的原子事实
- 使用 `EVENT_LOG_PROMPT`（`extract.py:174`）
- 同时 `embedding_client.embed([episode_body])` — 对 episode body 做 embedding
- 两者通过 `asyncio.gather` 并行执行
- 无 atomic fact 返回 raise `ValueError`

**3c. Atomic Fact Embedding**（`extract.py:189-190`）：
- `embedding_client.embed(fact_strings)` — 对所有 atomic fact 字符串做 embedding
- 必须等 3b 的 fact 抽取完成后才能执行（顺序依赖）

**3d. 组装序列化**（`extract.py:192-210`）：
- `episode_summary = episode_body[:200] + "..."`（截断，无 LLM 摘要，`extract.py:195`）
- 组装为 `EpisodeMemoryRecord`（subject, summary, content, embedding）+ `AtomicFactRecord`（time, atomic_fact, fact_embeddings, timestamp）
- 序列化为 dict

**输出**：每个 MemCell 变为一个 dict，包含：
- `items`：原始 ChatMessage 列表
- `episode`：`{subject, summary, content, embedding}`
- `atomic_facts`：`{time, atomic_fact: [str, ...], fact_embeddings: [[float, ...], ...], timestamp}`

---

## Step 4：聚类（Cluster by Geometry）

**操作对象**：Step 3 产出的全部 MemCell（已有 episode body embedding）

**过程**（`extract.py:435-467` → `algorithm.py:33-63`）：

按 MemCell 顺序**逐个、顺序处理**（不能并行，每步依赖前一步的 cluster 状态，`extract.py:455-462`）。对每个 MemCell：

**4a. 构造新 Cluster**（`extract.py:388-393`）：
- 用 MemCell 的 `episode.content_embeddings` 作为 centroid
- `last_ts` = MemCell 的 timestamp
- `members` = `[mc_id]`
- `preview` = `[episode_body]`

**4b. 匹配已有 Cluster**（`_find_best_within_window`，`algorithm.py:168-192`）：
- 遍历所有已有 cluster，依次检查：
  1. **时间窗口**：`abs(new_ts - cluster.last_ts) > 7天`（`cluster_max_time_gap_days=7.0`，`config.py`）→ 跳过
  2. **Cosine 相似度**：新 MemCell embedding 和 cluster centroid 的 cosine similarity
- 在时间窗口内的所有 cluster 中找 cosine 最高的（top-1）
- 最高分 ≥ `threshold=0.70`（`cluster_similarity_threshold`，`config.py`）→ 匹配成功
- 否则 → 无匹配

**4c. 合并或新建**（`extract.py:400-405`，`algorithm.py:148-158`）：
- **匹配成功**：加权平均 centroid（`(existing.centroid * existing.count + new.centroid * 1) / total`），合并 members 列表，更新 `last_ts` 为两者较大值
- **无匹配**：新建 cluster，分配 `id = "cluster_{len(existing)}"`

**输出**：`clusters_conv_<i>.json`，包含：
- `clusters`：每个 cluster 的 `{id, centroid, count, last_ts, members, preview}`
- `memcell_to_cluster`：`{memcell_id → cluster_id}` 反向映射

> **关键行为**：时间窗口检查在 cosine 之前——超出 7 天的 cluster 即使语义完全匹配也不会合并。这是之前实验中 adoption 主题被拆成 5 个 cluster 的原因。
