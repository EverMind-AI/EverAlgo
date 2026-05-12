# ADR 011: 接口约定 — Protocol（structural subtyping）over ABC

## 状态

✅ **Accepted** — 2026-04-28

## 背景

EverAlgo 需要为以下接口定义类型契约：
- `LLMClient`（调用 LLM provider）
- `Embedder`（向量嵌入）
- `Reranker`（重排序）
- 各 `Extractor` / `Ranker` 算子族签名

Python 提供两种主流接口约定方式：
- **ABC + `@abstractmethod`**（nominal subtyping）：实现类显式继承基类，子类必须实现 abstract method 才可实例化
- **Protocol**（structural subtyping，[PEP 544](https://peps.python.org/pep-0544/)）：实现类形状匹配即视为兼容，不需显式继承

相关硬约束：
- **H3** 算法同学迭代速度
- **H4** 无状态接口（library 不持业务状态）

## 候选方案

| 方案 | 描述 | 代表 |
|------|------|------|
| A. ABC + `@abstractmethod` | 所有接口 `abc.ABC` 基类，强制显式继承 | LangChain `BaseLLM` / litellm `CustomLLM` |
| **B. Protocol structural** | 接口 `typing.Protocol`，结构兼容即可 | DSPy `CodeInterpreter` / LlamaIndex `VectorStore` / instructor `*Handler` |
| C. 双轨混用 | 持状态/lifecycle 类用 ABC，无状态算子用 Protocol | LlamaIndex（核心 ABC + 插件接口 Protocol）|

## 客观优劣分析

### A. ABC 优势

| 维度 | 说明 |
|------|------|
| 子类必须实现完整 → "我该实现什么" 心智模型清晰 | 实例化时缺 abstract method 会运行时报错 |
| `isinstance` 检查工业级合规 | runtime 检查无 caveat |
| 提供共用方法（mixin） | 基类可放公共逻辑给子类 |
| IDE 重构友好 | 接口变更时显式继承类全部被连带提示 |
| Python 主流（早于 typing.Protocol）| `abc.ABC` 自 Python 2.6（[PEP 3119](https://peps.python.org/pep-3119/)）已有 |

### A. ABC 劣势

| 维度 | 说明 |
|------|------|
| 实现类必须显式继承 | 强耦合：用户 `import` ABC 才能写实现 |
| 不适用于第三方已有类 | 无法 retroactively 把已有类标记为兼容 |
| MRO 复杂 | 实现类继承多个不相关 ABC 时 mixin 冲突 |

### B. Protocol 优势

| 维度 | 说明 |
|------|------|
| 实现类不需显式继承 | [PEP 544](https://peps.python.org/pep-0544/) 原文 "_not necessary_ to subclass explicitly" |
| 第三方已有类自动兼容 | duck typing 的类型化版本 |
| IDE / mypy 静态检查 | [Python typing 官方文档](https://docs.python.org/3/library/typing.html#typing.Protocol) "structural subtyping (static duck-typing)"  |
| 实现类零 `import` 依赖 | 实现方写实现时不必 `import` EverAlgo 接口模块 |
| 符合 PEP 544 现代趋势 | Python 3.8+ 标准 |

### B. Protocol 劣势

| 维度 | 说明 |
|------|------|
| 不能持有共用方法（默认）| Protocol 方法默认无实现（虽支持 default 但不是主用法）|
| `isinstance` 需 `@runtime_checkable` | 仅查属性存在性，不查签名 |
| "我该实现什么" 心智模型弱 | 实现类缺方法不报错，仅 IDE / mypy 静态警告 |

### C. 双轨混用 优势/劣势

新优势：精准匹配语义（持状态 ABC + 无状态 Protocol）

新劣势：维护两套规范 + team 学习成本 + 边界判断分歧（"持多少状态算持状态"）

## 对 EverAlgo 适配度评估

### B（Protocol）优势对 EverAlgo 适配度

| 优势 | 适配度 |
|------|--------|
| 实现类不需显式继承 | ✅ **强需要**（H3 算法同学迭代速度，零 `import` 依赖） |
| 第三方已有类自动兼容 | ✅ 受益（用户已有 LLM client wrapper 不必改） |
| IDE / mypy 静态检查 | ✅ 强需要 |
| 实现类零 `import` 依赖 | ✅ **强需要**（H3） |
| 符合 PEP 544 现代趋势 | ✅ 受益 |

### B 劣势对 EverAlgo 适配度

| 劣势 | 适配度 |
|------|--------|
| 不能持有共用方法 | ⚠️ **不在意**（H4 算子无状态，无共用 lifecycle 方法）|
| `isinstance` 需 `@runtime_checkable` | ⚠️ 不在意（EverAlgo 不依赖 `isinstance` 运行时检查）|
| "我该实现什么" 心智模型弱 | ⚠️ 可 mitigate（docstring + 类型注解 + mypy strict 配置）|

### A（ABC）优势对 EverAlgo 适配度

| 优势 | 适配度 |
|------|--------|
| 心智模型清晰 | ⚠️ 可 mitigate |
| `isinstance` 合规 | ⚠️ **用不上** |
| 共用方法（mixin）| ⚠️ **用不上**（H4 无状态）|
| IDE 重构友好 | ⚠️ 受益（但 Protocol IDE 也支持）|

### A 劣势对 EverAlgo 适配度

| 劣势 | 适配度 |
|------|--------|
| 实现类必须显式继承 | ❌ **强烈介意**（H3 算法同学写 LLMClient 实现得 `import` EverAlgo，迭代心智成本高）|
| 第三方已有类不能 retroactively 兼容 | ❌ 介意（用户已有 OpenAI wrapper 必须改继承）|

### C（混用）评估

EverAlgo 算子全部归 H4 无状态阵营，**无任何持状态 lifecycle 接口需要 ABC**。混用引入双轨复杂度但 EverAlgo 无 ABC 适用场景 → 排除。

## 决策

**选 B：Protocol structural subtyping**。

逐条统计：
- B 强需要优势 3 条 + 受益 2 条；劣势全部不在意 / 可 mitigate
- A 优势 EverAlgo 用不上 / 可 mitigate；劣势强烈介意 1 条 + 介意 1 条

## 实施细节

```python
# everalgo/llm/_types.py
from typing import Protocol, runtime_checkable
from everalgo.llm.types import CompletionRequest, CompletionResponse

@runtime_checkable  # 仅供 EverAlgo 内部 sanity check（如配置层验证）；用户不依赖
class LLMClient(Protocol):
    """LLM 调用接口契约。实现类不需显式继承，结构匹配即兼容。"""

    async def acomplete(self, req: CompletionRequest) -> CompletionResponse: ...
    def complete(self, req: CompletionRequest) -> CompletionResponse: ...


# 用户实现（任意类，不需 import LLMClient）
class MyOpenAIWrapper:
    async def acomplete(self, req): ...
    def complete(self, req): ...


# 类型注解处使用
def configure(client: LLMClient) -> None: ...
configure(MyOpenAIWrapper())  # ✅ structural match，无需继承
```

要点：
- 接口模块下划线前缀 `_types.py`（公开类型集中入口由 `__init__.py` re-export）
- `@runtime_checkable` 仅供库内部 sanity check，用户不依赖运行时 `isinstance`
- async + sync 双方法签名同时声明（[ADR 010](010-sync-async-dual-interface.md)）

## 行业实证印证

### EverAlgo 同定位（"有外部调用能力的算法库"）实证

| 项目 | 接口 | 用法 |
|------|------|------|
| **DSPy** | `CodeInterpreter` | `@runtime_checkable Protocol`，用户自定义代码执行后端 |
| **DSPy** | `GEPAFeedbackMetric` / `PredictorFeedbackFn` | callable Protocol，metric 函数签名 |
| **LlamaIndex** | `VectorStore` / `GraphStore` | `@runtime_checkable Protocol`，docstring 自称 "Abstract vector store protocol" |
| **instructor** | `InstructorChatCompletionCreate` + 4 个 `*Handler` | callable Protocol，hooks 接口 |

共性：算法库面向"有自己实现的第三方"提供接口契约 → Protocol 让实现方零 `import` 依赖，duck typing 类型化即可。

证据文件路径：
- `dspy/primitives/code_interpreter.py:59` — `@runtime_checkable / class CodeInterpreter(Protocol)`
- `llama_index/core/vector_stores/types.py:269` — `@runtime_checkable / class VectorStore(Protocol)`
- `llama_index/core/graph_stores/types.py:216` — `@runtime_checkable / class GraphStore(Protocol)`
- `instructor/core/patch.py:37` — `class InstructorChatCompletionCreate(Protocol)`
- `instructor/core/hooks.py:22-40` — 4 个 handler Protocol

### 反例分析：LangChain

LangChain 同定位（算法+外部调用），但选 ABC：`BaseLLM` / `BaseChatModel` / `BaseRetriever` 全 `ABC + @abstractmethod`。

证据：`langchain/libs/core/langchain_core/language_models/llms.py:293` — `class BaseLLM(BaseLanguageModel[str], ABC)`。计数：8 Protocol（全内部辅助类型 `_RunnableCallable*` / `Stringifiable` / `SupportsAdd`）/ 34 ABC / 64 `@abstractmethod`。

LangChain 选 ABC 的合理性：
- chain 编排器**持组合状态**（callbacks / metadata / memory）—— 持状态
- `BaseLLM` 提供 lifecycle hooks（`on_llm_start` / `on_llm_end`）+ mixin 共用方法
- chain 编排需要 `isinstance` 区分 LLM / ChatModel / Retriever 类型

EverAlgo 与 LangChain 的关键不同：
- 算子**无状态**（H4）→ 无 lifecycle / mixin 需求
- 算子被 evermem 单调用，**不组装 chain** → 不需类型分发
- 算子通过 module 级 facade 直接 `import`（[ADR 008](008-re-export-vs-client-facade.md)）→ 不需"我必须继承谁"约束

→ LangChain 选 ABC 与 chain 框架定位匹配；EverAlgo 非 chain 框架场景 → Protocol 与 DSPy / LlamaIndex 插件接口同推理路径。

### 不引用 SDK 阵营

OpenAI Python SDK / Anthropic SDK / Google genai 虽然大量用 Protocol（`SSEBytesDecoder` / `CursorPageItem` / `HeadersLikeProtocol` 等），但其定位是**纯网络代理**（无算法成分），Protocol 用法源于 stainless 自动生成器对 typing 便利的偏好，与 EverAlgo 算法库接口约定动机不同。

**SDK 阵营不作 EverAlgo 同行先例引用** —— 避免"OpenAI SDK 这么做所以我们也这么做"的 cargo cult 推理。

## 后续演化触发条件

1. **某接口需要 lifecycle hooks / 共用 mixin 方法**（如 LLM client 需要 `on_request` / `on_response` 回调）→ 该接口转 ABC，与 Protocol 共存（C 双轨方案）
2. **PEP 544 重大变更**（如 Protocol 默认 `@runtime_checkable`）→ 重新评估 Protocol 与 `isinstance` 配合方式
3. **算法同学普遍反馈"不知道该实现什么"**（structural typing 心智模型不够清晰）→ 加 docstring 模板 + mypy strict 模式 + 实现指南文档 mitigate；mitigate 失败再考虑转 ABC

## 相关 ADR

- [ADR 008 re-export facade](008-re-export-vs-client-facade.md) — re-export 模式不需要 Client 类持状态，与 Protocol 无状态接口同推理路径
- [ADR 010 sync/async 双接口](010-sync-async-dual-interface.md) — Protocol 同时声明 `acomplete` + `complete` 两方法签名符合双接口规范
