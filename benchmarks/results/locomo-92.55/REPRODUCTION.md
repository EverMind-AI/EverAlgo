# LoCoMo Benchmark 复现报告 — 92.55%（agentic retrieval）

**日期**：2026-05-27
**Tag**：`v6-2-92.55`，commit `914ddec`（`🎨 refactor(benchmarks): pass encoding_format="float" to /embeddings explicitly`）
**分支**：`refactor/extract-algo-from-benchmarks`（run 时 HEAD）
**对照基线**：EverCore `locomo-benchmark` 分支（92.84% mean-of-runs）
**最终结果**：**EverAlgo 92.55%（1425/1540）**

> 注：Stage 3 alignment commits（§3.A + §3.B）是在这次 run **之后**才加的；要精确复现这一档分数请 `git checkout v6-2-92.55`。

---

## 一、三档数字对比

**评分方法**：Stage 5 在全部 QA 答案生成完后，让 judge LLM（`gpt-4o-mini` @ `temperature=0`）独立判 `judge_runs=3` 次。下表 Overall 为最终汇报准确率（全 1540 题）。

| 系统 | Overall | C1 single-hop | C2 temporal | C3 open-domain | C4 multi-hop | 备注 |
|---|---|---|---|---|---|---|
| **A. EverCore 论文卡片**（声称）| **93.05%** | — | — | — | — | cherry-pick 最高 run |
| **B. EverCore `locomo-benchmark` 复现**（93 worktree, mean-of-runs）| **92.84%** | 89.48% | 90.55% | 77.08% | 96.63% | std 0.15pp |
| **C. EverAlgo `v6-2-92.55`**（本次，agentic retrieval）| **92.55%** | **92.20%** | 88.68% | **79.17%** | 95.68% | 1425/1540 |

---

## 二、Per-Category 缺口分析（EverAlgo C vs 93 复现 B）

| Category | EverAlgo (C) | 93 复现 (B) | Δ | 评语 |
|---|---|---|---|---|
| C1 single-hop | **92.20%**（260/282）| 89.48% | **+2.72pp** | ✨ 反超 |
| C2 temporal | 88.68%（285/321）| 90.55% | **-1.87pp** | 缺口主来源 |
| C3 open-domain | **79.17%**（76/96）| 77.08% | **+2.09pp** | ✨ 反超 |
| C4 multi-hop | 95.68%（805/841）| 96.63% | -0.95pp | 小幅落后 |
| **Overall** | **92.55%** | **92.84%** | **-0.29pp** | — |

→ 整体缺口收敛到 **0.29pp**，剩余落后集中在 C2 temporal（-1.87pp）；C1 / C3 已反超基线。

---

## 三、关键 Run 配置

| 维度 | 取值 |
|---|---|
| retrieval_mode | `agentic`（hybrid + reranker + multi-query 全开）|
| emb / reranker top-n | 40 / 20 |
| hybrid candidates（emb / bm25）| 50 / 50，RRF k=40 |
| multi_query_num | 3 |
| response_top_k | 10 |
| clustering | 开（threshold 0.7，max-gap 7d，cluster_top_k=10）|
| llm_model | `openai/gpt-4.1-mini` @ temperature 0.3（OpenRouter）|
| judge_model | `openai/gpt-4o-mini` @ temperature 0.0，3 runs |
| embedding_model | `Qwen/Qwen3-Embedding-4B`（DeepInfra，1024 dim）|
| reranker_model | `Qwen/Qwen3-Reranker-4B`（DeepInfra）|

完整 config + 包版本见 `report.txt` 末尾「Config」「Package Versions」段，以及 `report.json`。

## 四、Stage 开销

| Stage | duration | prompt tokens | completion tokens |
|---|---|---|---|
| search | 885.6s | 4,083,323 | 327,004 |
| answer | 484.2s | 4,587,338 | 1,153,323 |
| evaluate | 65.7s | 2,322,720 | 27,753 |

---

## 五、报告位置 + 资产清单

本目录由 `benchmarks/.gitignore` 显式 whitelist 入仓供复核：

| 资产 | 仓内路径（相对仓根）| 用途 |
|---|---|---|
| 本报告 | `benchmarks/results/locomo-92.55/REPRODUCTION.md` | 复现说明 + 数字对比 |
| 顶层人类可读总结 | `benchmarks/results/locomo-92.55/report.txt` | Stage banner + per-cat 表格 + 完整 config |
| 顶层程序可读总结 | `benchmarks/results/locomo-92.55/report.json` | Stage timings + token usage + per-cat |
| Run 配置 + 包版本 | `benchmarks/results/locomo-92.55/profile.json` | 复现关键：固定 config + 依赖版本 |

**未入仓**（gitignored，本地重跑可重建）：`stage1_extract`（memcells / clusters / stats JSON，约 398MB）、`stage2_index`（bm25 / emb / cluster pkl，约 278MB）、`stage3_search/search_results.json`、`stage4_answer/answers.json`、`stage5_evaluate/eval_results.json`、`run.log`。想 audit 错题需重跑 stage 4-5：`uv run python -m benchmarks --dataset locomo --run-name <name> --stages 4 5`（前提 stage 1-3 已重建）。
