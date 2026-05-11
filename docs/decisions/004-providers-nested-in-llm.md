# ADR 004: LLM provider 物理位置 — 内嵌于 `llm/providers/` 子包

## 状态

✅ **Accepted** — 2026-04-23（v0.8 矫正 v0.5 错位决策）

## 背景

EverAlgo 的 LLM facade 提供 `everalgo.llm.complete(...)` / `stream_json` / scene 路由等抽象。**provider 实现位置**——OpenAI / Anthropic / vLLM / Bedrock 等具体后端的代码——放在哪里？

相关硬约束：
- **H3** 算法同学迭代速度（增删 provider 不应高负担）
- 子包组织一致性（与其他子包结构对称）

## 候选方案

| 方案 | 物理布局 | 用户调用 |
|------|---------|---------|
| **A. providers 顶层平级** | `everalgo/llm/` + `everalgo/providers/llm/<provider>/` 两个顶层子包 | `everalgo.llm.complete(...)` 内部委托到 `everalgo.providers.llm.<provider>.Client` |
| **B. providers 内嵌 llm/** | `everalgo/llm/{__init__, routing, client, providers/<provider>/}/` 单一子包 | `everalgo.llm.complete(...)` 内部委托到 `everalgo.llm.providers.<provider>.Client` |
| C. 顶层函数 + 单一 providers 子包 | `everalgo/main.py` + `everalgo/llms/<provider>/` | 顶层 `everalgo.complete(...)` |

## 客观优劣分析

### A. providers 顶层平级 优势

| 维度 | 说明 |
|------|------|
| facade 与实现物理分离 | `llm/` 只装抽象、`providers/llm/` 只装实现，"门面" vs "插件"边界清晰 |
| `llm/` 子包代码量小 | facade 不被 N 个 provider 子目录污染 |
| 多类型 provider 统一管理 | 未来若加 `providers/embed/` / `providers/storage/` 与 `providers/llm/` 平级，类型对齐 |

### A. providers 顶层平级 劣势

| 维度 | 说明 |
|------|------|
| **跨子包调用 = 暴露内部** | facade 子包要 `import everalgo.providers.llm.openai`，把 provider 内部模块路径暴露在 facade 代码里，破坏封装 |
| **子包导航割裂** | 算法同学改 LLM 调用要跨 `llm/` 和 `providers/llm/` 两处，一个领域两个目录 |
| **顶层 `providers/` 容器无 LLM 上下文** | 顶层 `providers/` 是无语义容器（仿 `common/` 反模式），看名字不知装什么 |
| 与主流 AI 库无任何一家对位 | 见下文行业实证 |

### B. providers 内嵌 llm/ 优势

| 维度 | 说明 |
|------|------|
| **LLM 域内聚** | facade + routing + provider 实现都在 `llm/` 一个子包，单领域单目录 |
| 算法同学增删 provider 路径短 | 改 `everalgo/llm/providers/new_provider/` 不需要跨子包改 |
| **顶层目录列表干净** | 顶层只有产品性 / 工具性子包，无无语义容器 |
| 与主流 AI 库 100% 对位 | 见下文行业实证 |

### B. providers 内嵌 llm/ 劣势

| 维度 | 说明 |
|------|------|
| `llm/` 子包代码量大 | 含 N 个 provider 子目录，但**这是 LLM 领域的真实复杂度**，不是组织问题 |
| 未来加 embed/storage providers 时需另选位置 | 但 EverAlgo 已决"不负责 embed"（[ADR 005 / design.md §1.2]），无此场景 |

### C. 顶层函数 + 单一 providers 子包 优势/劣势

新优势：调用最短（`everalgo.complete(...)`）

新劣势：
- 与 `everalgo.llm.complete(...)` 既定 API 不兼容（§2.5 scene 路由用了 `everalgo.llm` namespace）
- `everalgo.complete(...)` 无 LLM 语义指向（用户不知道这个 complete 是 LLM 调用还是别的）

## 对 EverAlgo 适配度评估

### A 优势对 EverAlgo 的适配度

| 优势 | 适配度 |
|------|--------|
| facade 与实现物理分离 | ⚠️ 不在意（algorithm library 不需要这种"插件式" 严格分层）|
| `llm/` 子包代码量小 | ⚠️ 用不上（LLM 域复杂度本就大，强行拆出去不解决问题，只挪位置）|
| 多类型 provider 统一管理 | ⚠️ 用不上（EverAlgo 只有 LLM provider，不需要 embed/storage 等） |

### A 劣势对 EverAlgo 的适配度

| 劣势 | 适配度 |
|------|--------|
| 跨子包调用暴露内部 | ❌ **强烈介意**（facade 内部不应直接 import 别的子包路径，破坏 H3 子包封装）|
| 子包导航割裂 | ❌ 强烈介意（H3 算法同学迭代速度）|
| 顶层 `providers/` 无语义 | ❌ 介意（违反 §1.2 命名原则"无 common/utils 容器"）|
| 与主流 AI 库无对位 | ❌ 介意（cargo cult 风险） |

### B 优势对 EverAlgo 的适配度

| 优势 | 适配度 |
|------|--------|
| LLM 域内聚 | ✅ **强需要**（H3 单领域单目录）|
| 增删 provider 路径短 | ✅ 强需要（H3）|
| 顶层目录干净 | ✅ 强需要（与 §1.2 命名原则一致）|
| 与主流 AI 库 100% 对位 | ✅ 受益 |

### B 劣势对 EverAlgo 的适配度

| 劣势 | 适配度 |
|------|--------|
| `llm/` 子包代码量大 | ⚠️ 不在意（真实复杂度，组织上无解，强行拆出去也是同样代码量）|
| 未来加多类型 providers 时需另选位置 | ⚠️ 用不上（EverAlgo 不负责 embed） |

### C 评估

直接排除——与 `everalgo.llm.complete(...)` 既定 API 命名不兼容，且 `everalgo.complete(...)` 无语义指向。

## 决策

**选 B：providers 内嵌于 `llm/` 子包内（`everalgo/llm/providers/<provider>/`）**。

逐条统计：
- A 强烈介意劣势 2 条 + 介意 2 条 + 优势 EverAlgo 全部不在意 / 用不上
- B 强需要优势 3 条 + 受益 1 条 + 劣势全部不在意 / 用不上

## 实施细节

```
everalgo/llm/
├── __init__.py            # facade（complete / stream_json / use / scene 路由）
├── routing.py             # scene → provider 路由查表
├── client.py              # 抽象基类
└── providers/
    ├── __init__.py
    ├── openai_compat/
    ├── anthropic/
    ├── vllm/
    └── bedrock/
```

`llm/__init__.py` 内部 `from everalgo.llm.providers.<provider> import Client` 在 routing 时调用——**同子包内 import，不跨子包**。

## 行业实证印证

明星 LLM 库 100% 都是"LLM 抽象 + provider 实现收在一个根子包内"模式（WebFetch 2026-04-23 核验 4 仓库）：

| 库 | 物理布局 |
|----|---------|
| **litellm** | 顶层 `__init__.py/main.py` + 单一 `litellm/llms/<provider>/` 子目录 |
| **instructor** | 顶层 `client.py/patch.py` + 单一 `instructor/providers/` 子包 |
| **dspy** | 单一 `dspy/clients/` 子包（抽象 + providers 都在里面）|
| **llama-index** | 单一 `llama_index/core/llms/<provider>/` 子包 |

**没有一家明星 LLM 库**采用 A 风格（providers 顶层平级 facade 子包）。EverAlgo v0.5 错位选了 A，v0.8 矫正回 B 与生态一致。

`providers` 命名贴近 instructor 模式；不用 litellm 的 `llms/`（顶层已是 `llm/`，子目录再叫 `llms` 套娃尴尬）；不用 dspy 的 `clients/`（与 `client.py` 文件名冲突）。

## 后续演化触发条件

1. **EverAlgo 增加非 LLM 类 provider**（如 storage provider / embed provider）→ 重新评估是否拆出顶层 `providers/<kind>/<provider>/`
2. **provider 数量极速增长（100+）**：`llm/providers/` 子目录过多时考虑分类（如 `llm/providers/cloud/<provider>` / `llm/providers/local/<provider>`）

## 相关 ADR

- [ADR 003 PEP 420 namespace](003-namespace-package-pep420.md) — `everalgo/llm/` 是 regular package，内嵌 providers 不影响 namespace 共享
- [ADR 008 re-export 式 facade](008-re-export-vs-client-facade.md) — `everalgo.llm.complete` 是顶层 re-export
