# ADR 003: namespace 模式 — PEP 420 共享 namespace（B 风格）

## 状态

✅ **Accepted** — 2026-04-23（v0.6 提议；v0.7 PEP 8 矫正后）

## 背景

[ADR 002](002-multi-distribution-vs-single.md) 决定多 distribution 后，每个 PyPI 包对应一个 import 路径。**namespace 模式**决定多 dist 之间的 import 路径如何组织。

相关硬约束：
- **H1** EverOS 文档契约 `evercore.user_memory.ChatMemCellExtractor`（要求共享 `evercore.*` 顶层）
- **H7** 可识别系列前缀

## 候选方案

| 方案 | 描述 | 代表项目 |
|------|------|---------|
| **A. 独立顶层 namespace（A 风格）** | 每个 dist 一个独立顶层 import 路径，dist 名 dash → import 名 underscore，**不共享顶层**。例：`langchain-core` → `import langchain_core`；`huggingface-hub` → `import huggingface_hub` | LangChain / HuggingFace |
| **B. PEP 420 共享 namespace** | 多 dist 共享一个顶层 namespace package（无 `__init__.py`），各 dist 提供顶层下不同子包。例：`llama-index-core` → `from llama_index.core import ...`；`google-cloud-storage` → `from google.cloud import storage` | LlamaIndex / google-cloud-* / Azure SDK |

## 客观优劣分析

### A. 独立顶层 namespace 优势

| 维度 | 说明 |
|------|------|
| import 路径短 | `import langchain_core` 一段路径 |
| 物理隔离强 | 各包顶层完全独立，不相互影响 |
| dist 名与 import 名贴近 | 用户从 PyPI dash 名一眼推断 import underscore 名 |
| 兼容老式 setuptools | 不依赖 PEP 420 implicit namespace package |
| 命名冲突风险低 | 顶层名独占，不依赖 namespace 协调 |

### A. 独立顶层 namespace 劣势

| 维度 | 说明 |
|------|------|
| 无系列归属感 | `import langchain_core` / `import langchain_openai` 散落在 import 表，用户难以一眼看出"这是 LangChain 套件" |
| 多包共存时 import 噪音大 | 用户写 `from langchain_core import X; from langchain_openai import Y` 多前缀 |
| 不能与 EverOS 契约 `evercore.user_memory.X` 兼容 | 强制 `evercore_user_memory.X` 平面命名 |

### B. PEP 420 共享 namespace 优势

| 维度 | 说明 |
|------|------|
| 系列归属感强 | `from evercore.user_memory import X` / `from evercore.boundary import Y` 都在 `evercore.*` 下，用户一眼看到套件 |
| import 表整洁 | 多个子包共享顶层，import 体验接近单 dist |
| **与契约 `evercore.user_memory.*` 完全兼容** | EverOS 文档契约直接 1:1 实现 |
| 用户视角无割裂 | 不论装 1 个还是 8 个 dist，import 路径都在 `evercore.*` 下 |

### B. PEP 420 共享 namespace 劣势

| 维度 | 说明 |
|------|------|
| 必须删 `evercore/__init__.py` | 不能在顶层 namespace 放任何代码（顶层是 implicit namespace package）|
| 命名冲突风险 | 多个 dist 都用 `evercore` 顶层时，子包名不能撞车（设计阶段需协调）|
| 老 IDE / 工具兼容性 | 早期 mypy / setuptools 对 PEP 420 支持弱（**2026 年已不是问题**，现代工具链全面支持）|
| 学习曲线 | 用户需理解"PyPI dash → import dot 共享" 的映射规则 |

## 对 EverCore 适配度评估

### A 风格优势对 EverCore 的适配度

| 优势 | 适配度 |
|------|--------|
| import 路径短 | ⚠️ 不在意（`evercore.user_memory` 也是两段，可接受）|
| 物理隔离强 | ⚠️ 用不上（同算法团队维护，无相互影响顾虑）|
| dist 名与 import 名贴近 | ⚠️ 不在意（`evercore-user-memory` → `evercore.user_memory` 映射规则一次学习长期受益）|
| 兼容老式 setuptools | ⚠️ 用不上（EverCore 全程现代工具链 uv / hatchling）|
| 命名冲突风险低 | ⚠️ 不在意（8 包同团队设计，子包名设计阶段已协调）|

### A 风格劣势对 EverCore 的适配度

| 劣势 | 适配度 |
|------|--------|
| 无系列归属感 | ❌ **强烈介意**（H7 可识别系列前缀；散落的 `evercore_user_memory` / `evercore_boundary` 失去套件感）|
| 多包共存 import 噪音 | ❌ 介意（算法同学日常多个 evercore-* 包共存）|
| **与契约 `evercore.user_memory.*` 不兼容** | ❌ **致命**——直接违反 H1 EverOS 文档契约 |

### B 风格优势对 EverCore 的适配度

| 优势 | 适配度 |
|------|--------|
| 系列归属感强 | ✅ **强需要**（H7） |
| import 表整洁 | ✅ 强需要（算法同学日常 multi-package import）|
| **与契约 `evercore.user_memory.*` 兼容** | ✅ **强需要**（H1 直接命中） |
| 用户视角无割裂 | ✅ 强需要（装多包视觉上等同单包）|

### B 风格劣势对 EverCore 的适配度

| 劣势 | 适配度 |
|------|--------|
| 删顶层 `__init__.py` | ⚠️ 不在意（顶层本就不放代码，所有功能在子包内）|
| 命名冲突风险 | ⚠️ 可 mitigate（设计阶段已确定 8 个子包名，无冲突）|
| 老 IDE / 工具兼容性 | ⚠️ 用不上（2026 年现代工具链完全支持 PEP 420）|
| 学习曲线 | ⚠️ 可 mitigate（一次性映射规则学习，与 llama-index 等共享同样规则）|

## 决策

**选 PEP 420 共享 namespace（B 风格）**。

逐条统计：
- A 致命劣势 1 条（违反 H1 EverOS 契约），强烈介意劣势 1 条（违反 H7）
- A 优势 EverCore **全部用不上 / 不在意**
- B 强需要优势 4 条
- B 劣势全部不在意 / 可 mitigate

## 实施细节

```
evercore/                  # PEP 420 namespace package（无 __init__.py）
├── core/                  # evercore-core 提供
│   ├── __init__.py        # regular package
│   └── ...
├── user_memory/           # evercore-user-memory 提供
│   ├── __init__.py        # regular package（承载 re-export）
│   └── ...
└── ...
```

**关键约定**：
- 顶层 `evercore/` 没有 `__init__.py`（PEP 420 隐式命名空间）
- 各子包 `evercore/<name>/` 是 regular package（必须有 `__init__.py`，承载 re-export，见 [ADR 008](008-re-export-vs-client-facade.md)）

命名映射规则：
- distribution（PyPI）`evercore-user-memory`（dash）
- import 路径 `evercore.user_memory`（dot 共享 namespace）
- 物理目录 `evercore/user_memory/`（underscore）

详见 [ADR 009 命名规范](009-naming-convention-llama-index-style.md)。

## 行业实证印证

PEP 420 共享 namespace 在主流多 dist 项目里的实证：

| 项目 | dist 命名 | import 共享 namespace | 业务结构同构度 |
|------|----------|---------------------|---------------|
| **LlamaIndex** | `llama-index-core` / `llama-index-llms-openai` | `llama_index.core` / `llama_index.llms.openai` | ✅ 完全同构（core + 紧密联动 integrations）|
| **google-cloud-python** | `google-cloud-storage` / `google-cloud-bigquery` | `google.cloud.storage` / `google.cloud.bigquery` | ⚠️ 业务独立场景（与 EverCore 略不同）|
| **Azure SDK for Python** | `azure-storage-blob` / `azure-identity` | `azure.storage.blob` / `azure.identity` | ⚠️ 业务独立 |

**反例（A 风格）**：
- LangChain：`langchain-core` → `import langchain_core`（独立顶层）
- HuggingFace：`huggingface_hub` / `transformers` 各自顶层

LangChain 的 A 风格选择不适合 EverCore，**核心原因是 EverOS 文档契约 `evercore.user_memory.*` 已确定**——LangChain 没有这种外部契约约束。

## 后续演化触发条件

1. **PEP 420 在生态中遇到不可调和问题**（如关键工具链放弃支持）→ 切回 A 风格 + 修订 EverOS 文档契约
2. **EverOS 文档契约改写**：如 zhanglibin 决定把 `evercore.user_memory.X` 全改为 `evercore_user_memory.X` → 切回 A 风格
3. **第三方贡献者大量加入**：A 风格独立 namespace 对外贡献门槛更低（fork 单仓单顶层）；当前内部项目此场景不存在

## 相关 ADR

- [ADR 002 多 distribution](002-multi-distribution-vs-single.md) — namespace 模式作用于多 dist 场景
- [ADR 008 re-export 式 facade](008-re-export-vs-client-facade.md) — 各子包 `__init__.py` 的 re-export 实现
- [ADR 009 命名规范](009-naming-convention-llama-index-style.md) — 系列前缀 + dash/dot 命名映射
