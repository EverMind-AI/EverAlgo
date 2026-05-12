# ADR 009: 命名规范 — dash distribution + 共享 dot namespace + lowercase 模块（llama-index 风）

## 状态

✅ **Accepted** — 2026-04-23（v0.6 制定 + v0.7 PEP 8 矫正）

## 背景

EverAlgo 拆为多 distribution（[ADR 002](002-multi-distribution-vs-single.md)）+ PEP 420 共享 namespace（[ADR 003](003-namespace-package-pep420.md)）。**命名规范**决定 distribution 名、import 路径、物理目录名之间的映射规则。

相关硬约束：
- **H1** evermem 文档契约 `everalgo.user_memory.ChatMemCellExtractor`
- **H7** 可识别系列前缀（PyPI 上 grep 出 EverAlgo 套件）
- PEP 8 命名规范（lowercase 模块 + PascalCase 类）

## 候选方案

| 方案 | 描述 | 代表 |
|------|------|------|
| **A. llama-index 风** | distribution dash（`everalgo-user-memory`）+ import dot 共享 namespace（`everalgo.user_memory`）+ 物理 underscore（`everalgo/user_memory/`）+ class PascalCase | LlamaIndex |
| B. langchain 风（A 风格独立顶层）| distribution dash（`langchain-core`）+ import 独立顶层 underscore（`import langchain_core`）+ class PascalCase | LangChain |
| C. HuggingFace 风（独立词无前缀）| distribution dash 或 underscore，无系列前缀（`transformers` / `huggingface-hub`）+ import 独立顶层 | HuggingFace 全家桶 |
| D. PascalCase namespace（Java/.NET 风）| `everalgo.UserMemory.ChatMemCellExtractor` —— evermem 文档原文写法（PEP 8 违例）| 无主流 Python 实证 |

## 客观优劣分析

### A. llama-index 风 优势

| 维度 | 说明 |
|------|------|
| 系列归属感强 | 所有包都在 `everalgo.*` 顶层，import 表整洁 |
| dist 名 ↔ import 路径映射规则一致 | dash → dot 简单转换 |
| 兼容现有 PEP 420 namespace（[ADR 003](003-namespace-package-pep420.md)）| 共享顶层 |
| **与 evermem 文档契约 `everalgo.user_memory.*` 完全兼容** | 1:1 映射 |
| 系列前缀 `everalgo-` 在 PyPI 可识别 | grep `everalgo-` 列出全部套件 |
| PEP 8 合规 | 模块 lowercase + 类 PascalCase |

### A. llama-index 风 劣势

| 维度 | 说明 |
|------|------|
| 多词模块名稍长 | `everalgo.user_memory` vs `everalgo.um`（缩写）|
| 用户需理解 dash → dot 映射 | 一次性学习成本 |

### B. langchain 风 优势

| 维度 | 说明 |
|------|------|
| import 路径短 | `import langchain_core` 一段路径 |
| 物理隔离强 | 各包顶层独立 |
| 兼容老式 setuptools | 不依赖 PEP 420 |
| dist 名 ↔ import 名贴近 | dash → underscore |
| PEP 8 合规 | lowercase + PascalCase |

### B. langchain 风 劣势

| 维度 | 说明 |
|------|------|
| **与 evermem 文档契约不兼容** | 强制 `everalgo_user_memory.X` 平面命名 |
| 无 `everalgo.*` 系列归属感 | import 表中 `everalgo_user_memory` / `everalgo_boundary` 散落 |
| 多包共存 import 噪音 | 每个 dist 一个独立顶层名 |

### C. HuggingFace 风 优势/劣势

新优势：每个包名最短最直接（`transformers`）

新劣势：
- 无系列前缀 → PyPI 上 `core` / `parser` / `boundary` / `rank` 等通用词冲突已被占用
- 无系列归属（用户散落 PyPI 后无法识别 EverAlgo 套件）
- 与 evermem 契约不兼容

### D. PascalCase namespace 优势/劣势

无优势——不符合 Python 任何主流实证

劣势：
- **PEP 8 违例**——模块名应 lowercase，PascalCase 是类名规范
- 无主流 Python 实证（Java/.NET 风格）
- 命名歧义：`everalgo.UserMemory` 看起来像类不像模块

## 对 EverAlgo 适配度评估

### A 优势对 EverAlgo 的适配度

| 优势 | 适配度 |
|------|--------|
| 系列归属感强 | ✅ **强需要**（H7） |
| dash → dot 映射规则一致 | ✅ 受益 |
| 兼容 PEP 420 namespace | ✅ **强需要**（[ADR 003](003-namespace-package-pep420.md) 决议） |
| 与 evermem 文档契约兼容 | ✅ **强需要**（H1）|
| PyPI 上 grep `everalgo-` 可识别 | ✅ 强需要（H7）|
| PEP 8 合规 | ✅ 强需要（社区惯例）|

### A 劣势对 EverAlgo 的适配度

| 劣势 | 适配度 |
|------|--------|
| 多词模块名稍长 | ⚠️ 不在意（`user_memory` 比 `um` 可读性好太多）|
| dash → dot 映射学习成本 | ⚠️ 不在意（一次性 + 与 llama-index 等共享同规则）|

### B / C / D 评估

B：与 H1 evermem 契约**不兼容**——直接排除

C：违反 H7 可识别系列前缀 + 与 H1 不兼容——直接排除

D：PEP 8 违例 + 无 Python 主流实证——**evermem 文档原文这样写但属违例，本文档矫正**

## 决策

**选 A：llama-index 风**。

### 命名映射规则

| 维度 | 规范 | 例 |
|------|------|------|
| Distribution（PyPI）| 全 dash | `everalgo-user-memory` |
| import 路径（dot 分隔）| 共享 `everalgo.*` namespace | `everalgo.user_memory` |
| 物理目录 | underscore | `everalgo/user_memory/` |
| 顶层 `everalgo/` | **PEP 420 隐式命名空间**，无 `__init__.py` | — |
| 子包 `everalgo/<name>/` | regular package，需 `__init__.py`（承载 re-export） | — |
| 多词模块名 | underscore 分词（PEP 8 推荐） | `user_memory` / `agent_memory` / 不是 `usermemory` |
| 类名 | CapWords / PascalCase | `ChatMemCellExtractor` |

### evermem 文档矫正

evermem 设计文档原文使用的 PascalCase namespace 是 PEP 8 违例（Java/.NET 风格），本文档统一矫正：

| 原文（evermem 文档）| 矫正后 |
|------|--------|
| `everalgo.UserMemory.ConvMemCellExtractor` | `everalgo.user_memory.ChatMemCellExtractor` |
| `everalgo.AgentMemory.AgentCaseExtractor` | `everalgo.agent_memory.AgentCaseExtractor` |
| `everalgo.Parser` | `everalgo.parser` |
| `everalgo.Knowledge.KnowledgeExtractor` | `everalgo.knowledge.KnowledgeExtractor` |
| `everalgo.Rank.EpisodicRanker` | `everalgo.rank.EpisodicRanker` |

矫正涵盖两层：① PEP 8 lowercase namespace（`UserMemory` → `user_memory` 等）；② `Conv` 缩写歧义消除（`Conv` → `Chat`，避免与 PyTorch `nn.Conv2d` 等 Convolution 占用冲突；与 OpenAI / Anthropic / LlamaIndex / HuggingFace / DSPy 业界 5/6 主流命名对齐）。

evermem 文档作者侧（zhanglibin）需同步修订；memsys_opensource 现状代码（`ConvMemCellExtractor` / `RawDataType.CONVERSATION` / `CONV_BATCH_BOUNDARY_DETECTION_PROMPT` / `conv_memcell_extractor.py`）落地 EverAlgo 时一并重命名。

## 行业实证印证

主流 Python 项目多词模块命名 100% 用下划线分词：

| 库 | 多词模块例 |
|----|-----------|
| **scikit-learn** | `sklearn.linear_model` / `feature_selection` / `model_selection` / `naive_bayes` / `neural_network` |
| **llama-index** | `llama_index.core.vector_stores` / `chat_engine` / `query_engine` / `node_parser` / `text_splitter` |
| **transformers** | `feature_extraction_utils` / `modeling_outputs` / `tokenization_utils` |
| **langchain** | `langchain.chat_models` / `output_parsers` / `text_splitters` |
| **huggingface_hub** | distribution `huggingface-hub` → import `huggingface_hub` |

PEP 8 原文："Modules should have short, all-lowercase names. **Underscores can be used in the module name if it improves readability**."

主流 AI 库多词模块命名 100% 一致：lowercase + underscore。`usermemory` / `agentmemory` 单词连写**在主流 Python AI 库无任何依据**。

llama-index 风（A）的精确实证：
- distribution `llama-index-core` / `llama-index-llms-openai`（全 dash）
- import path `llama_index.core` / `llama_index.llms.openai`（PEP 420 共享 namespace）
- 顶层 `llama_index/` 无 `__init__.py`（implicit namespace package）
- 各子包 `llama_index/core/` 含 `__init__.py`（regular package，承载 re-export）

EverAlgo 完全套用此模式。

## 后续演化触发条件

1. **evermem 文档修订**：zhanglibin 把 evermem 文档矫正为 PEP 8 lowercase 后，本 ADR 矫正表删除（仅保留命名规范本身）
2. **PEP 420 命名空间出现重大不兼容问题**：转 langchain 风 A 风格独立顶层（同时改 evermem 契约）—— 触发 [ADR 003](003-namespace-package-pep420.md) 重新评估
3. **某子包名与第三方包冲突**：调整子包命名（不影响 ADR 整体规则）

## 相关 ADR

- [ADR 003 PEP 420 namespace](003-namespace-package-pep420.md) — namespace 模式选择是命名规范的前置
- [ADR 002 多 distribution](002-multi-distribution-vs-single.md) — distribution 命名规则
