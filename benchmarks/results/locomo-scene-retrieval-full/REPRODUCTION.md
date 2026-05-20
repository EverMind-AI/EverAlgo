# LoCoMo Benchmark 复现最终报告

**日期**：2026-05-20
**分支**：`feat/benchmarks` HEAD `1978a4d`
**对照基线**：EverCore `locomo-benchmark` 分支（92.84% mean-of-runs）
**最终结果**：**EverAlgo 91.58%（mean-of-runs，std 0.04pp）**

---

## 一、三档数字对比

**评分方法（mean-of-runs）**：Stage 5 在所有 QA 答案生成完之后，让 judge LLM（`gpt-4o-mini` @ `temperature=0`）独立判 3 次，对**每次 judge run** 单独计算「全部 1540 题的准确率」，最后报告**三次 run-level accuracy 的平均值**（即 `mean-of-runs`）。`std` 是三次 run-level accuracy 的样本标准差，反映 judge LLM 自身的方差，不反映 retrieval/answer 质量。算法 mirror 93 `stage5_eval.py:245-267`。每个备注里的 `3 runs [a, b, c]` 就是这三个 run-level accuracy 原始值。

| 系统 | Overall | C1 single-hop | C2 temporal | C3 open-domain | C4 multi-hop | 备注 |
|---|---|---|---|---|---|---|
| **A. EverCore 论文卡片**（声称）| **93.05%** | — | — | — | — | cherry-pick 3 runs 最高 |
| **B. EverCore `locomo-benchmark` 复现**（我此前用 93 worktree 跑 3 runs mean-of-runs）| **92.84%** | 89.48% | 90.55% | 77.08% | 96.63% | std 0.15pp，3 runs `[92.73, 93.05, 92.73]` |
| **C. EverAlgo `scene-retrieval-full`**（本次最终）| **91.58%** | **92.79%** | 87.85% | 76.39% | 94.33% | std 0.04pp，3 runs `[91.56, 91.56, 91.62]` |

---

## 二、Per-Category 缺口分析（EverAlgo C vs 93 复现 B）

| Category | EverAlgo (C) | 93 复现 (B) | Δ | 评语 |
|---|---|---|---|---|
| C1 single-hop | **92.79%** | 89.48% | **+3.31pp** | ✨ **我们反超** |
| C2 temporal | 87.85% | 90.55% | **-2.70pp** | 缺口最大来源之一 |
| C3 open-domain | 76.39% | 77.08% | -0.69pp | 基本持平 |
| C4 multi-hop | 94.33% | 96.63% | **-2.30pp** | 缺口次大来源 |
| **Overall** | **91.58%** | **92.84%** | **-1.26pp** | — |

→ 缺口集中在 C2 + C4（共贡献约 -5pp 的 cat-level Δ，加权后得整体 -1.26pp）。

---

## 三、剩余 1.26pp 缺口的可能来源（按严苛专家排序）

### 一档可能（高概率贡献，已识别可控）

#### 1. Episode 抽取的 wrapper format 差异

**93** 喂给 Episode LLM 的 conversation 是 pseudo-JSON object per msg（`get_conversation_json_text`，`episode_memory_extractor.py:130-163`）：

```
                {
                    "timestamp": <ts>,
                    "speaker": <name>,
                    "content": <text>
                }
```

**EverAlgo** 喂的是 `[<ts>] <speaker>: <content>` 单行（`_render_conversation`，`packages/everalgo-user-memory/src/everalgo/user_memory/episode.py:130-141`）：

```
[2023-06-09T20:00:00Z] Caroline: [Caroline shared an image: ...] Thanks Mel! ... last week!
```

两边送进 LLM 的 **content 字段字符串 100% 一致**（image caption prepend 算法逐行 identical）；唯一差异是 wrapper format。

**实证**：smoke run QA1 错题（"Caroline 跟 friends/family/mentors 何时见面"）—— 93 Episode LLM 在 mc 5（24 items）保留了「a school event held last week (May 31 - June 6, 2023)」时间锚，我们 Episode LLM 在 mc 2（23 items，几乎同等 boundary 粒度）把 D3:11 「last week meetup」跟 D3:13 「4 years since moved from home country」**合并压缩**，仅保留长期关系叙事，丢失 specific 时间锚。

**合理推断**：pseudo-JSON 让 LLM 把每条 msg 当结构化数据逐字段提取（时间锚不易丢）；单行 chat-log 让 LLM 把相邻 msg 当连续叙事整段压缩（短语易丢）。**这一项最可能贡献 C2 temporal 缺口的大部分。**

#### 2. `response_format={"type": "json_object"}` 在 Stage 1 + Stage 3 强制 JSON mode

| 调用 | 93 | EverAlgo |
|---|---|---|
| Stage 1 BoundaryDetector / Episode / AtomicFact | 未传（text + manual JSON parse） | **强制 `response_format={"type":"json_object"}`** |
| Stage 3 check_sufficiency / generate_multi_queries | 未传 | **强制 `response_format={"type":"json_object"}`** |

OpenAI JSON mode 限制 LLM sampling 空间到 valid JSON 路径，可能让输出更确定（解释我们 std 0.04pp 极低的部分原因），但也可能 hurt 某些抽取/推理的语义自由度。BOSS 早期讨论 stage 1 时拍板「不改」，本次保持不动。

### 二档可能（中等概率，结构差异）

#### 3. Boundary 切分策略差异

- **93** `ConvMemCellExtractor` 逐条 msg 滚动决策 + `smart_mask`（`stage1_memcells_extraction.py:221-360`）
- **EverAlgo** `BoundaryDetector` batch 20 + tail-carry 流式切片

实证：93 conv 0 切 51 个 memcell，我们 19 个（每 session 1 个）。但对 session_3 这种连续单话题，93 也大段不切（24 items 一个 mc），跟我们 23 items 几乎同等粒度。**全局粒度差异存在，但单个关键 mc 的粒度对齐**——所以这一项可能贡献 conv 内多 sub-topic 长 mc 上的 episode 抽取压力（间接影响 C2 / C4）。BOSS 早期决定保留我们的 batch 策略。

#### 4. Episode + AtomicFact 抽取时机

- **93**：boundary 切到 → 立刻串行抽 Episode → 所有 memcell 收齐后并发抽 EventLog
- **EverAlgo**：所有 boundary 出齐后，并发抽 (Episode → AtomicFact-from-text) per memcell

逻辑等价；BOSS 早期拍板保留并发。

### 三档可能（低概率，环境差异）

#### 5. Provider 路由差异

- **93** judge LLM 用 OpenAI 直连（`gpt-4o-mini`）
- **EverAlgo** 全链路走 OpenRouter（`openai/gpt-4o-mini` / `openai/gpt-4.1-mini`）

OpenRouter 在 `temperature=0` 下输出更稳定（可能内部 caching），这解释了我们 judge std 0.04pp vs 93 std 0.15pp 的差异。但 std 低**不影响 mean accuracy**，所以这一项不解释 1.26pp 缺口。

#### 6. LLM nondeterminism

`gpt-4.1-mini @ temperature=0.3` 在长 conv 上 Episode 抽取输出有方差。同 input、同 prompt、同 model，不同 run 输出不同。93 baseline 自己 3 runs 也跑出 [92.73%, 93.05%, 92.73%]（std 0.15pp）——LoCoMo 复现的 noise floor 本身约 ±0.15pp。

EverAlgo 单 run 91.58% 落在「93 mean 92.84% 减去 ~1.26pp」位置，**1.26pp ≈ 8x noise floor**，不可能全由方差解释，但贡献部分可能 ~0.3-0.5pp。

---

## 四、报告位置 + 资产清单

本报告及其引用的所有 in-repo 资产都在 `benchmarks/results/locomo-scene-retrieval-full/` 目录下，由 `benchmarks/.gitignore` 显式 whitelist 入仓供其他人复核：

| 资产 | 仓内路径（相对仓根）| 用途 |
|---|---|---|
| 本报告 | `benchmarks/results/locomo-scene-retrieval-full/REPRODUCTION.md` | 复现说明 + 数字对比 |
| 顶层人类可读总结 | `benchmarks/results/locomo-scene-retrieval-full/report.txt` | Stage banner + per-cat 表格 |
| 顶层程序可读总结 | `benchmarks/results/locomo-scene-retrieval-full/report.json` | Stage timings + token usage + per-cat |
| Run 配置 + 包版本 | `benchmarks/results/locomo-scene-retrieval-full/profile.json` | 复现关键：固定 config + 依赖版本 |
| Git commit 链 | `git log --oneline feat/benchmarks` | commit 历史本身即可追溯每步对齐改动 |

**未入仓**（gitignored，本地重跑可重建）：stage 1 memcells/clusters JSON、stage 2 BM25/emb/scene pkl、stage 3 search_results.json、stage 4 answers.json、stage 5 eval_results.json（28 MB，单文件超出 pre-commit `check-added-large-files` 阈值）。想 audit 错题需要重跑 stage 4-5：`uv run python -m benchmarks --dataset locomo --run-name <name> --stages 4 5`（约 $4.49 + 18 min，前提是 stage 1-3 已重建）。
