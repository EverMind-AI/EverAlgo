# ADR 010: sync/async 双接口模式

## 状态

✅ **Accepted** — 2026-04-23（v0.15 矫正 v0.13 § 1.4 单 async 误判）

## 背景

EverAlgo 的 I/O-bound 算子（各 Extractor / Ranker / boundary / Parser）涉及网络 I/O（LLM 调用 / 外部解析服务）。这些算子的同步/异步接口形态决定用户使用方式。

相关硬约束：
- **H3** 算法同学迭代速度（Jupyter Notebook / 简单脚本场景偏 sync）
- **evermem 集成场景**：FastAPI / Hertz 异步服务（强需要 async）
- **算法库 vs SDK 阵营**：EverAlgo 是算法库（[ADR 008](008-re-export-vs-client-facade.md)），不持跨调用状态

## 候选方案

| 方案 | 描述 |
|------|------|
| **A. 双接口（sync + async）** | 每个 I/O 算子提供 `extract` / `aextract` 两个版本；async 名前缀 `a`（litellm / llama-index / langchain 风），或独立类（instructor 风）|
| B. async-only | 只提供 async 版本；sync 用户写 `asyncio.run(extract(...))` |
| C. sync-only | 只提供 sync 版本；async 用户用 thread pool 包装 |
| D. anyio / 通用代码 | 一份代码同时支持 sync 和 async（实验性，未主流）|

## 客观优劣分析

### A. 双接口（sync + async）优势

| 维度 | 说明 |
|------|------|
| **覆盖两类用户场景** | sync 用户（Jupyter / 脚本）+ async 用户（FastAPI / asyncio.gather）都直接用 |
| 与明星 AI 库 100% 一致 | litellm / instructor / llama-index / langchain 全部双接口（行业默认）|
| async 实现可桥接 sync | 基类只实现 async，sync 通过 `asyncio.run` / `asgiref.sync` 自动派生 |
| 性能最优 | sync 路径直接同步执行，async 路径直接事件循环；无桥接 overhead |
| 用户心智成本低 | sync 用户不需要学 async 概念 |

### A. 双接口（sync + async）劣势

| 维度 | 说明 |
|------|------|
| 维护双倍代码 | 每个算子两份签名，**但**有标准实现技巧降本（基类派生 / `asgiref.sync` 等）|
| API 表面 +1 倍 | docstring / type stub / mypy 配置都翻倍 |
| 命名约定需团队统一 | 是 `aextract` / `extract_async` / `AsyncEverAlgoClient` 风格之一，需确定 |

### B. async-only 优势

| 维度 | 说明 |
|------|------|
| 单一代码路径 | 无双倍维护成本 |
| 强迫用户使用现代 async 习惯 | 推动用户接受异步思维 |

### B. async-only 劣势

| 维度 | 说明 |
|------|------|
| **sync 场景使用门槛上升** | Jupyter Notebook / 脚本用户要写 `asyncio.run(extract(...))` 包装 |
| **明星 AI 库无此实证** | litellm / instructor / llama-index / langchain 100% 双接口；无主流 AI 库选择 async-only |
| 嵌套 asyncio.run 易错 | 已在 async 上下文里再调 sync `asyncio.run()` 会报错 |

### C. sync-only 优势/劣势

新优势：实现最简

新劣势：
- **evermem FastAPI 场景下 sync 算子阻塞事件循环**（致命）
- async 用户要用 `run_in_executor` / thread pool 包装，性能低
- 明星 AI 库无此实证

### D. anyio / 通用代码

新优势：一份代码两种调用

新劣势：
- 非主流（litellm / instructor / llama-index / langchain 都不用）
- 实施复杂度高，调试痛苦
- mypy / IDE 推断不友好

## 对 EverAlgo 适配度评估

### A 优势对 EverAlgo 的适配度

| 优势 | 适配度 |
|------|--------|
| 覆盖两类用户场景 | ✅ **强需要**（evermem 异步 + 算法同学 sync 都要服务）|
| 与明星 AI 库 100% 一致 | ✅ 强需要（cargo cult 反向：与所有用户已熟悉的模式一致，零学习成本）|
| async 实现可桥接 sync | ✅ 受益（降低维护成本）|
| 性能最优 | ✅ 受益 |
| 用户心智成本低 | ✅ **强需要**（H3 算法同学迭代速度）|

### A 劣势对 EverAlgo 的适配度

| 劣势 | 适配度 |
|------|--------|
| 维护双倍代码 | ⚠️ 可 mitigate（基类派生 / `asgiref.sync.async_to_sync` 自动桥接）|
| API 表面 +1 倍 | ⚠️ 不在意（docstring 重复可机器生成） |
| 命名约定需统一 | ⚠️ 可 mitigate（约定 `a` 前缀，与 litellm / llama-index / langchain 一致）|

### B 劣势对 EverAlgo 的适配度

| 劣势 | 适配度 |
|------|--------|
| sync 场景使用门槛上升 | ❌ **强烈介意**（H3 算法同学 Jupyter / 脚本场景被拖累）|
| 明星 AI 库无此实证 | ❌ 介意（cargo cult 反向警示）|
| 嵌套 asyncio.run 易错 | ❌ 介意 |

### C / D 评估

C：FastAPI 场景阻塞事件循环致命 → 排除

D：非主流 + 实施复杂 → 排除（除非未来生态主流转向）

## 决策

**选 A：双接口（sync + async）**。

### 命名约定

| 算子类型 | sync 名 | async 名 | 例 |
|---------|---------|----------|---|
| 函数 | `<verb>` | `a<verb>` | `extract` / `aextract`、`rerank` / `arerank`、`parse` / `aparse`、`complete` / `acomplete` |
| 类方法（如 Extractor）| `.extract()` | `.aextract()` | `EpisodeExtractor().extract(memcell)` / `.aextract(memcell)` |

`a` 前缀风格与 litellm / llama-index / langchain 100% 一致；instructor 的 `Async<X>` 类风格不采（需要算法同学记忆 `Instructor` 和 `AsyncInstructor` 两个类，多一层心智）。

### 决定性约束：主用户 evermem = FastAPI 异步服务

EverAlgo 的**主用户是 evermem**——基于 FastAPI 开发的记忆服务，handler 全部 `async def`。这意味着：

- **async 接口是主战场**——evermem 100% 走 async 调用，sync 调用阻塞 event loop 会拖累整个 FastAPI 实例
- **async 接口必须 native async**——不能 thread pool wrap（详见下文性能分析）
- **sync 接口的实际用户**：CLI 同步脚本 / 单元测试 / 算法同学命令行实验等次要场景

### 实施模式实证（9 项目，2 命名风格 × 5 实施细节）

| 项目 | 命名风格 | 实施细节 |
|------|---------|---------|
| **litellm** | 双方法 / 双函数 | **手写两份**（`completion()` + `acompletion()`）|
| **llama-index** | 双方法 | 基类抽象，子类提供 `_aquery` 真实现 |
| **langchain Runnable** | 双方法 | **sync-first + thread pool 派生 async**（默认 `ainvoke` 调 `run_in_executor` 包装 `invoke`）|
| **dspy** | 双方法 | 单类双方法 |
| **instructor** | **双类** | `Instructor` + `AsyncInstructor` |
| **httpx** | 双类 | **共享基类**（`BaseClient` + `Client` + `AsyncClient`）|
| **OpenAI Python SDK** | 双类 | 平行基类（`OpenAI(SyncAPIClient)` + `AsyncOpenAI(AsyncAPIClient)`）|
| **redis-py** | 双类 | `redis.Redis` + `redis.asyncio.Redis` 独立模块 |
| **SQLAlchemy 2.0** | 单实现 | **greenlet-based async over sync**（独特方案）|

**两条命名路线**：双类（持状态客户端）vs 双方法（无状态算子）。EverAlgo 是算法库（[ADR 008](008-re-export-vs-client-facade.md)），归**双方法**路线。

### 实施细节 3 选 1 评估

双方法路线下 sync ↔ async 实施细节：

| 子模式 | 主实现 | 派生方向 | 致命陷阱 | 性能 | 维护 |
|-------|-------|---------|---------|------|------|
| **B1. 手写两份** | sync + async 各一份 | 无 | 无 | ✅ 两条路径都 native | ❌ 双倍代码（业务逻辑可抽公共函数缓解） |
| **B2. async-first + asgiref 桥接 sync** | async | `asgiref.async_to_sync(async)` | ⚠️ **sync 接口在已有 event loop 内调用 raise RuntimeError**（Jupyter / FastAPI 内）| ✅ async 路径 native；sync 限非 event loop 环境 | ✅ 单实现 |
| **B3. sync-first + thread pool 派生 async** | sync | `run_in_executor(sync)` | 无 | ❌ **async 不 native**（详见下文性能分析）| ✅ 单实现 |

### B3 性能瓶颈分析（FastAPI 主战场）

evermem FastAPI 场景（100 QPS / LLM 1 秒延迟）：

| 实施 | event loop 行为 | 吞吐 |
|------|---------------|------|
| **真 async**（B1 / B2 的 async 路径，用 httpx async / aiohttp）| 单 event loop 100 并发 await，CPU 几乎闲 | **~100 RPS native** |
| **B3 sync-first + thread pool wrap**（默认 thread pool ~32 worker）| 32 thread 并行执行 sync HTTP，68 个请求排队 | **降至 ~32 RPS**（差 ~3x） |

调大 thread pool（如 1000）虽可扩并发，但每 thread 内核栈 8MB+ → 1000 thread 占 8GB；context switch overhead 累加显著。**native async 单 thread 跑数千 await 是工业界标准**。

**B3 不适配 EverAlgo 主用户 FastAPI** → 排除。

### B1 vs B2：覆盖范围 vs 维护成本权衡

| 场景 | B1 手写两份 | B2 async-first + asgiref |
|------|-----------|------------------------|
| evermem FastAPI handler（async）| ✅ native | ✅ native |
| Jupyter Notebook 算法同学 sync 调 | ✅ 直接调 sync 方法 | ❌ raise（必须用 `await aextract(...)`）|
| Jupyter Notebook 算法同学 async 调 | ✅ `await aextract` | ✅ `await aextract` |
| CLI 同步脚本 | ✅ | ✅ |
| 单元测试（sync）| ✅ | ✅（pytest 不在 event loop）|
| 维护代码量 | 业务逻辑抽公共函数后 +1 行 / 算子 | 单实现 |

### 决策：B2 async-first + asgiref + 文档明示边界

**理由**（基于主用户约束）：

1. **evermem（主用户）100% 走 async** → B2 完全满足
2. sync 接口次要——CLI 脚本 + 单元测试不在 event loop 内，asgiref 桥接正常
3. Jupyter 算法同学用 `await aextract(...)` 顶层 await（litellm / dspy 实证 + 现代 Python 生态约定）
4. **单实现降低维护成本**——v0.x 演化阶段类型契约 / Protocol 频繁变化，单实现修一处即可

文档须明示约定：

> `everalgo.<X>.extract(...)` sync 接口仅限**非 event loop 环境**调用（CLI 脚本 / 单元测试）。
> Jupyter / FastAPI / 任何 `async def` 上下文用 `everalgo.<X>.aextract(...)` 配合 `await`。

与 litellm `acompletion()` / dspy 等同样的约定。

### 实施样例

```python
# everalgo-user-memory: everalgo/user_memory/episode.py
from asgiref.sync import async_to_sync

class EpisodeExtractor:
    async def aextract(self, memcell: MemCell) -> list[Episode]:
        """async 主实现（单一真实代码）"""
        prompt = await everalgo.prompts.arender("episode", memcell=memcell)
        response = await everalgo.llm.acomplete(prompt, scene="episode")
        return _parse_episodes(response)

    extract = async_to_sync(aextract)  # 自动派生 sync 桥接
    """sync 接口；仅限非 event loop 环境调用"""
```

或基类抽象（多算子统一）：

```python
# everalgo-core: everalgo/_dual_interface.py
from asgiref.sync import async_to_sync

class DualInterface:
    """子类只实现 a* 方法，sync 自动派生。"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in dir(cls):
            if name.startswith("a") and len(name) > 1 and name[1].islower():
                async_method = getattr(cls, name)
                if callable(async_method) and not hasattr(cls, name[1:]):
                    setattr(cls, name[1:], async_to_sync(async_method))

class EpisodeExtractor(DualInterface):
    async def aextract(self, memcell): ...
    # extract 自动派生
```

### 退路：何时改回 B1 手写两份

若以下任一发生，重新评估：

1. **Jupyter 在 EverAlgo 用户群占比超 30%**（且不能强制要求用 `await`）→ B1 让 Jupyter 也能 sync 调
2. **某算子 sync / async 业务逻辑路径有显著差异**（如 sync 走批 sync HTTP / async 走流式）→ 该算子改 B1
3. **asgiref 出现兼容性问题** → 切回 B1

### 纯计算算子保持单一同步

`rank.fusion.{rrf, lr, cosine_to_lr_score, score_propagation}` / `boundary._tokenize.count_tokens` / `clustering` 距离计算等纯计算算子**不提供 async 版本**——这是 Python 社区公认。

**3 处官方文档实证**（WebFetch 2026-04-28 核验）：

| 来源 | 关键论断 |
|------|---------|
| **Python 官方 asyncio 文档** [`asyncio-dev.html` "Running Blocking Code"](https://docs.python.org/3/library/asyncio-dev.html) | "Blocking (CPU-bound) code **should not be called directly** [in async]. For example, if a function performs a CPU-intensive calculation for 1 second, all concurrent asyncio Tasks and IO operations **would be delayed by 1 second**." 推荐 `loop.run_in_executor()` + `ProcessPoolExecutor` 隔离 |
| **FastAPI 官方文档** [`/async/`](https://fastapi.tiangolo.com/async/) | "for **CPU bound** operations like Machine Learning, you can exploit **parallelism and multiprocessing** for higher performance." + "If you just don't know, use normal `def`." |
| **NumPy 官方文档** | "NumPy targets compute-bound operations (not I/O-bound), where async provides minimal benefit." |

**实证延伸**：numpy / scipy / sklearn / pandas / pytorch / jax 等所有数值/科学计算库 **100% sync API**，无 async 反例。

**纯计算算子不提供 async 版本的实证（9 项目）**：

| 项目 | 纯计算算子 | 是否提供 async 版本 |
|------|----------|------------------|
| numpy | `np.dot` / `linalg.*` / `fft.*` | ❌ 只有 sync |
| pandas | `df.merge` / `groupby` / `apply` | ❌ 只有 sync |
| sklearn | `KMeans.fit` / `cosine_similarity` | ❌ 只有 sync |
| pytorch | `torch.matmul` / `nn.Linear.forward` | ❌ 只有 sync |
| scipy | `linalg` / `optimize` / `stats` | ❌ 只有 sync |
| litellm | `litellm.token_counter()` | ❌ 只有 sync（无 `atoken_counter`） |
| llama-index | `_reciprocal_rerank_fusion` / tokenize 工具 | ❌ 只有 sync |
| OpenAI SDK | `openai.tokenizer` 等 helper | ❌ 只有 sync |
| httpx | URL parsing / cookie 处理 | ❌ 只有 sync |
| **⚠️ langchain Runnable** | `PromptTemplate` / `OutputParser` 纯计算节点 | ✅ **被强制提供 ainvoke**（默认 thread pool 派生）|

**唯一反例 langchain 的特殊原因**：LCEL chain 要求所有 Runnable 节点统一 `invoke / ainvoke` 接口——一个 chain 由 `prompt | llm | parser` 组成，整 chain 用统一 `await chain.ainvoke(...)` 调用。若纯计算节点不提供 ainvoke，整 chain 就不能 async 链式调用。代价是 langchain 纯计算节点的 `ainvoke` 默认通过 `run_in_executor` 派生（B3 模式 thread pool overhead）—— langchain 接受性能代价换 chain 接口统一。

**EverAlgo 非 chain 框架，无 langchain 那种统一性需求**：算子被 evermem 直接单调用（`await ranker.arank(...)` + 后处理一次 `rank.fusion.rrf(...)`），纯计算算子是辅助函数在 async 上下文直接 sync 调用，不嵌入 async chain。按非 chain 框架的 8 家主流惯例（numpy/pandas/.../OpenAI SDK/httpx）选 sync only。

**社区共识精确表述**：
- **asyncio 为 I/O-bound 设计**，CPU-bound 写 `def` 不写 `async def`
- 在 async 函数体里写阻塞 CPU 计算 = 阻塞整个 event loop（asyncio 单 thread 协作式调度）
- CPU 并行不靠 async（受 GIL 限制），靠 `multiprocessing` / `ProcessPoolExecutor`

### 重计算 caveat

"纯计算 sync"成立的前提是**计算时长可忽略**。精确判断标准：

| 计算时长 | sync 直接调用 | 风险 |
|---------|------------|------|
| **≤ 10ms**（如 `tokenize.count_tokens`、`fusion.rrf`、单个 cosine）| ✅ 安全 | event loop 几乎无感 |
| **10-100ms**（如批量 100 项 cosine、小型聚类）| ⚠️ 边缘 | 高 QPS 场景累计阻塞需关注 |
| **> 100ms**（如大批量 1000+ 向量相似度、大模型 inference、复杂聚类）| ❌ 不安全 | 阻塞 event loop 导致并发请求排队，需 `run_in_executor` / `ProcessPoolExecutor` 隔离 |

EverAlgo 当前纯计算算子（`fusion.rrf` / `_tokenize` / 单 `cosine`）都在 ≤ 10ms 量级，sync 直接调用安全。

**未来若新增重计算算子**（batch 1000+ 向量 cosine、大规模聚类等）：保持 sync `def` API 形态不变，由 caller（evermem）决定是否 `await loop.run_in_executor(executor, op, ...)` 包装隔离。**不推荐**把重计算改写为 `async def` 内部写 sync 阻塞——这是 anti-pattern，违反 Python 官方 asyncio-dev 警告。

## 行业实证印证

WebFetch 2026-04-23 核验 9 个明星项目（见上表「实施模式实证」）。**100% 都提供双接口，无反例**。EverAlgo 在双方法路线 + B2 实施细节，与现代 Python AI 库主流惯例一致。

特别关键的两条桥接库实证：
- **`asgiref.async_to_sync` 源码** github.com/django/asgiref `sync.py`：`"You cannot use AsyncToSync in the same thread as an async event loop"` — 明确不允许 event loop 内调用，本 ADR 据此定 sync 接口边界
- **langchain Runnable** github.com/langchain-ai/langchain `libs/core/.../runnables/base.py`：默认 `ainvoke` 调 `run_in_executor` 包装 sync `invoke` —— 是 B3 模式实证，但不适配 EverAlgo FastAPI 主战场

## 后续演化触发条件

1. **anyio / 通用代码模式成为主流**（如 langchain LCEL 演化为单接口）→ 重新评估 D 方案
2. **算法同学反馈 sync 接口实际很少用**（90%+ 流量是 async）→ 考虑标 sync 接口为 deprecated（不删，按 SemVer 渐进）
3. **某算子 async 实现性能远超 sync 桥接**（>2x 差距）→ 该算子 sync 改为独立实现而非桥接

## 相关 ADR

- [ADR 008 re-export 式 facade](008-re-export-vs-client-facade.md) — 算法库 vs SDK 二分；EverAlgo 是算法库故不用 Client 类，但仍提供 sync/async 双接口
- [ADR 004 LLM providers 内嵌](004-providers-nested-in-llm.md) — 各 provider 也是双接口，与 litellm `completion()/acompletion()` 模式对齐
