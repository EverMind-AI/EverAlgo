# ADR 008: facade 实现模式 — re-export 式 facade（非 Client 类）

## 状态

✅ **Accepted** — 2026-04-23（v0.5 物理按算法族 + re-export；v0.6 加算法库 vs SDK 二分论据）

## 背景

EverCore 子包按算法职责物理组织（`boundary/` / `clustering/` 等），EverOS 文档契约要求 `evercore.user_memory.ChatMemCellExtractor` 这种调用路径。物理位置和契约路径**解耦**意味着需要一个 facade 机制把契约路径映射到实现路径。

Python 提供两种 facade 实现路径：
- **re-export 式**：`__init__.py` 里 `from <impl> import <Class>`，把实现对象绑定到外部契约 namespace
- **Client 类式**：定义一个 Client 类（持有连接 / 状态 / 资源），方法委托到内部实现

**facade 实现模式**决定 EverCore 选哪种。

相关硬约束：
- **H1** EverOS 文档契约 `evercore.user_memory.*`
- **H4** 无状态接口（library 不持业务状态）

## 候选方案

| 方案 | 描述 | 代表 |
|------|------|------|
| **A. re-export 式 facade** | `__init__.py` 里 `from .impl_module import Class`，用户 `from evercore.user_memory import ChatMemCellExtractor` 走 re-export 拿到实现路径下的同一类对象 | numpy / pandas / pytorch / sklearn / dspy / llama_index.core |
| B. Client 类 facade | 定义 `EverCoreClient` 等类，用户 `client = EverCoreClient(); client.user_memory.extract_episode(...)` | OpenAI SDK / Anthropic SDK / google-genai SDK |

## 客观优劣分析

### A. re-export 式 facade 优势

| 维度 | 说明 |
|------|------|
| 算子可独立组合 | 用户按任务挑算子，import 即用 |
| 无跨调用状态承载需求 | 算子调用之间彼此独立 |
| **零运行时开销** | re-export 是 Python 模块字典的 attribute 绑定，无 wrapper class |
| 短路径访问 | `from evercore.user_memory import EpisodeExtractor` 直达 |
| IDE 跳转无影响 | 编辑器跳转仍指向真实定义位置（实现模块） |

### A. re-export 式 facade 劣势

| 维度 | 说明 |
|------|------|
| 需要写 `__all__` | 否则 mypy strict 模式报"implicit re-export"（一次性写好长期受益） |
| 命名冲突需协调 | 多个子包 re-export 同名 class 会互相覆盖（设计阶段可避免） |
| 不能持有跨调用状态 | 用户调用 `EpisodeExtractor.extract(...)` 之间无状态续存 |

### B. Client 类 facade 优势

| 维度 | 说明 |
|------|------|
| 跨调用状态自然承载 | client 持有 connection pool / auth token / rate limiter / retry policy 等 |
| 资源生命周期清晰 | `__init__` / `__enter__` / `__exit__` / `__del__` 管理资源 |
| 调用上下文清晰 | `client.chat.completions.create(...)` 资源树语义直观 |
| 配置注入到 client | 用户在构造 client 时注入 config，避免全局状态 |

### B. Client 类 facade 劣势

| 维度 | 说明 |
|------|------|
| **需要 client instance 才能调用** | 算子使用门槛高一层 |
| 算子组合不灵活 | 用户难以"挑选 Episode + Profile 两个算子"独立调用，必须经 client |
| 资源开销 | client 实例化 / 持有有 cost（虽然小） |
| Python 算法库无此实证 | 见下文行业实证 |

## 对 EverCore 适配度评估

### A 优势对 EverCore 的适配度

| 优势 | 适配度 |
|------|--------|
| 算子可独立组合 | ✅ **强需要**（EverCore 多个 Extractor / Ranker 用户按任务挑）|
| 无跨调用状态承载需求 | ✅ 强需要（H4 无状态接口） |
| 零运行时开销 | ✅ 受益 |
| 短路径访问 | ✅ 强需要（H3 算法同学迭代速度） |
| IDE 跳转无影响 | ✅ 受益 |

### A 劣势对 EverCore 的适配度

| 劣势 | 适配度 |
|------|--------|
| 需要写 `__all__` | ⚠️ 不在意（一次性写好）|
| 命名冲突需协调 | ⚠️ 可 mitigate（设计阶段已规划）|
| 不能持有跨调用状态 | ✅ **正合 EverCore 意图**（H4 不持状态）|

### B 优势对 EverCore 的适配度

| 优势 | 适配度 |
|------|--------|
| 跨调用状态自然承载 | ⚠️ **用不上**（H4 EverCore 不持业务状态）|
| 资源生命周期清晰 | ⚠️ 用不上（EverCore 算子无连接 / token / pool 资源） |
| 调用上下文清晰 | ⚠️ 不在意（re-export 模式 `evercore.user_memory.X` 也清晰）|
| 配置注入到 client | ⚠️ 可 mitigate（`evercore.configure(...)` 全局 setter + contextmanager 实现配置注入，无需 client 持有）|

### B 劣势对 EverCore 的适配度

| 劣势 | 适配度 |
|------|--------|
| 需要 client instance 才能调用 | ❌ **强烈介意**（H3 算法同学迭代速度受损）|
| 算子组合不灵活 | ❌ 强烈介意（H3 算子按任务挑选）|
| 资源开销 | ⚠️ 不在意 |
| Python 算法库无此实证 | ❌ 介意（cargo cult 风险反向）|

## 决策

**选 A：re-export 式 facade**。

逐条统计：
- A 强需要优势 3 条 + 受益 2 条；劣势全部不在意 / 可 mitigate / 反而契合
- B 优势 EverCore 全部用不上 / 不在意；劣势强烈介意 2 条

## 实施细节

```python
# evercore/user_memory/__init__.py
from evercore.boundary.chat      import ChatMemCellExtractor      # re-export
from evercore.boundary.workspace import WorkspaceMemCellExtractor
from evercore.user_memory.episode    import EpisodeExtractor
from evercore.user_memory.foresight  import ForesightExtractor
from evercore.user_memory.atomic_fact import AtomicFactExtractor
from evercore.user_memory.profile    import ProfileExtractor

__all__ = [  # 必须显式声明，避免 mypy implicit re-export 警告
    "ChatMemCellExtractor",
    "WorkspaceMemCellExtractor",
    "EpisodeExtractor",
    "ForesightExtractor",
    "AtomicFactExtractor",
    "ProfileExtractor",
]
```

两条访问路径同时有效：
- **EverOS 及外部用户路径**：`from evercore.user_memory import ChatMemCellExtractor`
- **算法同学物理路径**：`from evercore.boundary.chat import ChatMemCellExtractor`

两条路径指向**同一个类对象**（`A is B` 为 True），re-export 只是 namespace 绑定，无运行时开销。

## 行业实证印证

Python 生态 facade 模式有清晰的"算法库 vs 网络 SDK"二分（WebFetch 2026-04-23 核验 7 仓库）：

| 类型 | 模式 | 代表 |
|------|------|------|
| **算法库**（多算子组合 / 无跨调用状态）| **re-export 式**（顶层 `__init__.py` 把子包对象抽上来）| **transformers / pytorch / scikit-learn / dspy / numpy / pandas** |
| **网络 API SDK**（单一服务 / auth/session/pool 跨资源 / 持 client）| **Client 类式**（`_client.py` + `resources/`）| **openai-python / anthropic-sdk / google-genai** |

- 大公司原厂 SDK **100% 用 Client 类**（OpenAI / Anthropic / Google 三家全部 `_client.py` + `resources/` 结构）
- 大公司算法库 **100% 用 re-export**（Meta PyTorch、HuggingFace transformers、scikit-learn 全部 `__init__.py` 顶层 re-export）
- **两阵营泾渭分明，无中间地带**

EverCore 按 §1.1 定位是算法库（无状态接口 + 多个独立可组合算子 + 算法同学按任务挑），明确属算法库阵营，与该阵营所有大公司库一致选 re-export。

## 后续演化触发条件

1. **EverCore 改为持有跨调用状态**（如 long-lived session / connection pool）→ 重新评估 Client 类模式（但与 H4 冲突，应慎重）
2. **算子需要 batch / 异步流水线编排**：可能需要某种 client 持有 pipeline 状态（届时考虑加 `evercore.Pipeline` 类，与 re-export 共存）
3. **mypy / IDE 工具链对 implicit re-export 处理重大变化**：仍仅影响 `__all__` 声明负担，方案不需变

## 相关 ADR

- [ADR 003 PEP 420 namespace](003-namespace-package-pep420.md) — re-export 在 namespace package 模式下的实现机制
- [ADR 004 providers 内嵌](004-providers-nested-in-llm.md) — `evercore.llm.complete` 是顶层 re-export
