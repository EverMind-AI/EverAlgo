# 🧠 EverAlgo 架构总览

> 给算法同学和新同学的 onboarding 文档。读完应该能够回答：EverAlgo 是什么、负责什么、不负责什么、怎么调用、命名/调用契约是什么。
>
> - 想知道某个决策**为什么这样定** → 跳转 [`decisions/`](../decisions/) 里对应的 ADR
> - 想跑 hello world → 见 `getting-started/`（待补）

---

## 1. 鸟瞰：evermem 依赖 EverAlgo

**evermem 是 AI 记忆管理系统**，对外提供完整产品能力（API、持久化、编排、记忆生命周期管理）。**EverAlgo 是算法库**，被 evermem 依赖来实现 Extract / Rank 算法 IP。两者是"产品 ↔ 依赖库"关系：

```
┌──────────────────────────────────────────────┐
│  evermem（AI 记忆管理系统）                   │
│  API 网关 / 持久化 / 编排 / 锁 / scene 路由   │
└────────────────┬─────────────────────────────┘
                 │ 依赖（in-memory 数据结构调用）
                 ▼
┌──────────────────────────────────────────────┐
│  EverAlgo（算法库）                           │
│  无状态 · Extract / Rank 算法 IP             │
└──────────────────────────────────────────────┘
```

**evermem** —— 记忆管理系统本身。负责 API 网关、DB 持久化、调用编排、并发与锁、scene 路由（哪个算法步骤用哪个模型）、分布式协调、记忆生命周期管理。可以是开源版用户的本地 Python 应用，也可以是商业版云平台的 Go 主体 + Python 算法微服务。

**EverAlgo** —— evermem 依赖的算法库。**无状态、不碰 DB、不读写文件系统、不持有业务编排概念**。输入输出全部是 in-memory 数据结构，类比 evermem 之于 EverAlgo 就像 FastAPI 应用之于 numpy / sklearn——产品负责一切，库只负责算法。

边界用一句话讲清：

- EverAlgo 不知道有 DB，由 caller 把数据备好传入
- EverAlgo 不知道"哪个场景用哪个 LLM"，由 caller 选好 client 传入
- EverAlgo 不知道并发和分布式锁，由 caller 自行串行化 read-modify-write

---

## 2. 算法库定位

EverAlgo 与 sklearn / pytorch / DSPy 同阵营，**不是** LangChain / LlamaIndex 那种端到端框架：

| 维度 | 算法库（EverAlgo 阵营）| 端到端框架（LangChain 阵营）|
|------|---------------------|--------------------------|
| 持有状态？ | 否 | 是（chain / memory / agent）|
| 持有业务编排？ | 否 | 是 |
| 接口形态 | 模块级函数 + 全局 default 配置 | Client 类 / chain 组合 |
| 主要用户 | 算法实现者 + 二次开发者 | 应用开发者 |

EverAlgo **提供**：算法 IP（Extract + Rank 双主轴的算子）、LLM 调用门面、Provider 路由、prompt validator、testing 辅助。

EverAlgo **不提供**：API 网关、持久化、并发编排、retry / fallback、多 key 轮转、多租户、配额、observability。这些归工程载体（evermem）。

> 阵营判断的详细论证见 [ADR 008](../decisions/008-re-export-vs-client-facade.md)。

---

## 3. 双主轴契约：Extract + Rank

EverAlgo 的算法职责分两条主轴，对外契约对称（**无状态、不碰 DB、输入输出都是内存数据结构**）：

| 主轴 | 时机 | 输入 | 输出 |
|------|------|------|------|
| **Extract** | 记忆写入链路 | 对话数组 `list[Message]`（边界检测前）/ `MemCell`（边界检测后，下游 Extractor 入口）| 结构化记忆（Episode / Profile / Case / Skill / KnowledgeMemory ...）|
| **Rank** | 记忆检索链路 | 多路召回候选 + 预取关联 | 排序好的记忆列表 |

Extract 主轴典型链路是两段——上游 `*MemCellExtractor.adetect(messages | agent_trace | jira_ticket | ...)` 把原始输入切成 `list[MemCell]`；下游各 `*Extractor.aextract(memcell)` 在 MemCell 级别做派生记忆抽取。`MemCell` 是上下游的桥接结构，不同业务源（聊天 / Agent 轨迹 / Workspace 数据）走各自的 boundary 算子产出统一形态的 MemCell。

**Rank 不读任何存储**。所有跨记忆关联（如 `Episode → AtomicFact`）由 evermem 在 Recall 阶段一并预取传入；Ranker 在内存里做 hierarchy 展开，无任何 DB 调用。

```python
import asyncio
from everalgo import user_memory, rank

# Extract 主轴：写入链路
memcells = await user_memory.ChatMemCellExtractor().adetect(messages, llm=...)
for memcell in memcells:
    episodes = await user_memory.EpisodeExtractor().aextract(memcell, llm=...)

# Rank 主轴：检索链路（rank_input 由 evermem 备好，含召回候选 + 预取关联）
ranked = await rank.episodic.arank(rank_input)
```

---

## 4. 子包总览

11 个 subpackage（Python import 单位），打成 8 个 PyPI distribution（发布单位）：

```
                       everalgo-core
            （types / llm（含 providers）/ prompts / testing）
                              ▲
        ┌───────────┬─────────┴───────────┬───────────┐
        │           │                     │           │
   everalgo-   everalgo-              everalgo-   everalgo-
   boundary    clustering                rank      parser
        ▲           ▲                                  ▲
        │           │                                  │
   everalgo-   everalgo-                        everalgo-
   user-       agent-                           knowledge
   memory      memory
```

| 类型 | subpackage | 职责 |
|------|-----------|------|
| 产品性 | `user_memory` | `Episode / Foresight / AtomicFact / Profile` 产出 |
| 产品性 | `agent_memory` | `AgentCase / AgentSkill` 产出 |
| 产品性 | `knowledge` | 文件型 `KnowledgeMemory` 产出 |
| 工具性 | `parser` | 多模态原始文件 → `ParsedContent` |
| 工具性 | `boundary` | 3 种 MemCell 切分 + `_tokenize` / `_force_split` 共享 |
| 工具性 | `rank` | 4 业务 facade + `fusion` / `weight` / `rerank` 算法工具 |
| 工具性 | `clustering` | `cluster_by_geometry` + `cluster_by_llm` + `ClusterState` |
| 基础设施 | `llm` / `prompts` / `types` / `testing` | 全部在 `everalgo-core` distribution |

**两条 import 路径同时有效**——按你的角色选：

- **evermem 工程同学**（按业务路径调用，对齐契约）：`from everalgo.user_memory import ChatMemCellExtractor`
- **算法同学**（按算法物理路径迭代 boundary 共享底层）：`from everalgo.boundary.chat import ChatMemCellExtractor`

两条路径指向**同一个类**。物理路径与外部契约通过 `__init__.py` re-export 解耦。

### 4.1 物理目录速览（你打开 IDE 看到的样子）

monorepo + uv workspace 布局，每个 distribution 独立 `pyproject.toml`，源码在 `src/everalgo/<subpackage>/`：

```
everalgo/                                   # 仓库根（uv workspace）
├── pyproject.toml                          # workspace 根
├── uv.lock                                 # 整 workspace 共享 lockfile
├── packages/
│   ├── everalgo-core/                      # 基础设施 dist
│   │   ├── pyproject.toml                  # 独立 SemVer + 独立 PyPI 发布
│   │   └── src/everalgo/
│   │       ├── types/                      # 公共数据契约
│   │       ├── llm/                        # LLM 调用门面 + Provider 路由
│   │       │   └── providers/              # openai_compat / anthropic / bedrock
│   │       ├── prompts/                    # validator + 多语言子模块约定
│   │       └── testing/                    # assertions + fake_llm
│   ├── everalgo-boundary/  src/everalgo/boundary/      # MemCell 切分（chat / workspace / agent）
│   ├── everalgo-clustering/src/everalgo/clustering/    # cluster_by_geometry / cluster_by_llm
│   ├── everalgo-rank/      src/everalgo/rank/          # 4 业务 facade + fusion / weight / rerank
│   ├── everalgo-parser/    src/everalgo/parser/        # image / audio / document / video / url
│   ├── everalgo-user-memory/   src/everalgo/user_memory/   # episode / foresight / atomic_fact / profile
│   ├── everalgo-agent-memory/  src/everalgo/agent_memory/  # case / skill
│   └── everalgo-knowledge/     src/everalgo/knowledge/     # extractor
├── docs/                                   # 你正在读的目录
├── tests/                                  # 跨包集成测试
└── local/                                  # 本地产物（git 忽略）
```

要点：

- 顶层 `everalgo/` 是 [PEP 420](https://peps.python.org/pep-0420/) **隐式命名空间包**——**无 `__init__.py`**，多个 distribution 共享 `everalgo.*` import 路径
- 子包 `everalgo/<name>/` 是 regular package，**有 `__init__.py`** 承载 re-export
- `prompts` 在 `everalgo-core/src/everalgo/prompts/` 下；具体 prompt 字符串就近放各业务子包 `prompts/{en,zh}/<name>.py`（不外置 `.md` / `.yaml`）

> "为什么允许两条路径" → [ADR 008](../decisions/008-re-export-vs-client-facade.md)
> "为什么 clustering 独立子包" → [ADR 006](../decisions/006-clustering-independent-subpackage.md)
> "为什么按 PEP 420 共享 namespace" → [ADR 003](../decisions/003-namespace-package-pep420.md)

---

## 5. 命名强契约

### 5.1 a 前缀 = native async

EverAlgo 命名强契约：

- **`a` 前缀**（`aextract` / `arank` / `adetect` / `aparse`）= **native async**（含 LLM / 外部 I/O），调用必须 `await`
- **无 `a` 前缀**（`extract` / `rank` / `count_tokens` / `rrf`）= **sync**（纯计算 / 无 I/O），调用不要 `await`

看名字即知接口形态，无需查算子表。同 DSPy `acall` / `aforward` / litellm `acompletion` 命名约定。

```python
# a 前缀，必须 await
episodes = await EpisodeExtractor().aextract(memcell, llm=...)

# 无 a，纯计算，不要 await
score = rank.fusion.rrf(vec_hits, kw_hits)
n = boundary._tokenize.count_tokens(text)
```

### 5.2 I/O 算子双接口、纯计算单接口

| 类别 | 接口形态 |
|------|---------|
| I/O 算子（含 LLM / OCR / ASR / 抓取）| **双接口** sync + async |
| 纯计算算子（fusion / tokenize / 距离）| **仅 sync**，不实现 async |

双接口约束：sync 接口**仅限非 event loop 环境**调用（CLI 脚本 / 单元测试）；Jupyter / FastAPI / 任何 `async def` 上下文须用 `await aextract(...)`。

> "为什么纯计算不实现 async" → [ADR 010](../decisions/010-sync-async-dual-interface.md)（含 9 项目实证 + 性能对比）

### 5.3 命名速查

| 维度 | 规范 | 例 |
|------|------|-----|
| Distribution（PyPI）| 全 dash | `everalgo-user-memory` |
| import 路径 | underscore + 共享 `everalgo.*` | `everalgo.user_memory` |
| 物理目录 | underscore | `everalgo/user_memory/` |
| Class | PascalCase | `ChatMemCellExtractor` |
| 函数 / 方法 | snake_case；async 加 `a` 前缀 | `aextract` / `count_tokens` |

> 详见 [ADR 009](../decisions/009-naming-convention-llama-index-style.md)。

---

## 6. 算子调用形态

EverAlgo 算子用 **3 层 LLM 注入**（DSPy 同款），优先级 **per-call > scoped > default**：

| 层 | 用法 | 优先级 | 典型场景 |
|----|------|--------|---------|
| **per-call** | `aextract(..., llm=client)` 直接传参 | 最高 | evermem 单调用按 scene 注入（**生产主路径**）|
| **scoped** | `async with everalgo.llm.use(client):` | 次高 | evermem pipeline 段批量同 client |
| **default** | 启动期 `everalgo.configure(llm=...)` | 兜底 | dev / 测试 / Jupyter |

```python
import everalgo
from everalgo.llm import LLMConfig
from everalgo import user_memory, rank

# (1) 启动期 default：dev / 测试必备；生产 evermem 可以不调
everalgo.configure(llm=LLMConfig(
    provider="openai_compat",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_KEY"],
    model="openai/gpt-4.1-mini",
))

# (2) Per-call：生产主路径，按 scene 注入
episodes = await user_memory.EpisodeExtractor().aextract(
    memcell, llm=scene_router.get("episode"))

# (3) Scoped：多调用同 client 时减少重复传参
async with everalgo.llm.use(scene_router.get("rerank")):
    ranked_eps = await rank.episodic.arank(rank_input)
    ranked_cases = await rank.case.arank(rank_input)
```

**EverAlgo 无 scene 概念**——"哪个算法步骤用哪个模型"是业务决策，由 evermem 持有 `SceneRouter`，3 层注入按场景选用。EverAlgo 算子签名只接 `llm: LLMClient | None = None`，内部用 `everalgo.llm.resolve(llm)` 一行解析三层 fallback。

> "为什么 scene 路由不在 EverAlgo" + "为什么 3 层注入" → [ADR 012](../decisions/012-llm-stack-architecture.md)

---

## 7. 状态管理边界

EverAlgo 是无状态算法库。任何"累积状态"都遵循同一函数式模式：

| 谁 | 职责 |
|---|------|
| **EverAlgo** | 定义"state 是什么"（值对象类型 + 字段 + 序列化）+ "state 怎么演化"（值对象方法返回新实例）|
| **caller**（evermem 编排器）| "state 何时载入 / 存哪里 / 谁加锁"——持久化、并发控制、事务回滚 |

以 `ClusterState` 为例：

```python
from everalgo.clustering import (
    cluster_by_geometry, ClusterState, ClusterConfig,
)

# caller 加载 state（介质自选 MongoDB / Redis / 文件）
raw = await state_store.load(user_id)
state = ClusterState.from_dict(raw) if raw else ClusterState.empty()

# caller 加锁串行化 read-modify-write
async with caller.lock(f"cluster:{user_id}"):
    cluster_id, new_state = await cluster_by_geometry(
        vector, timestamp, state, config=ClusterConfig(),
    )
    await state_store.save(user_id, new_state.to_dict())
```

值对象 in / out（`@dataclass(frozen=True)`），EverAlgo 永不 mutate 入参——算法异常不污染原 state，事务安全。

不同业务的 state 实例物理隔离：user_memory 编排器持一个 `ClusterState` 实例装 episode 簇、agent_memory 编排器持另一个实例装 case 簇。**类型在 EverAlgo 单一定义**，**实例由 caller 各自持有**。

> "为什么 state 在 caller 持久化、core 写演化逻辑" + "为什么实例物理隔离" → [ADR 006](../decisions/006-clustering-independent-subpackage.md)

---

## 8. 何时读哪份文档

新同学按需深入：

| 想做什么 | 读哪 |
|---------|------|
| 装一个 dist 跑 hello world | `getting-started/01-installation.md`（待补） |
| 30 行跑通 ChatMemCellExtractor → Episode | `getting-started/02-first-extraction.md`（待补） |
| 新增一个 Extractor | `how-to/add-extractor.md`（待补） |
| 新增一个 Ranker | `how-to/add-ranker.md`（待补） |
| 自定义某个 prompt | `how-to/customize-prompt.md`（待补） |
| 接入新 LLM provider | `how-to/implement-llm-provider.md`（待补） |
| 写测试（用 fake_llm + assertions）| `how-to/write-tests.md`（待补） |
| 查 API / 字段 / 配置精确签名 | [`reference/`](../reference/)（占位，含写作时机说明；schema 定稿后填充）|
| 理解 Extract / Rank 边界细节 | `concepts/two-axis-extract-rank.md`（待补）|
| 理解为什么 EverAlgo 无状态 | `concepts/stateless-design.md`（待补）|
| 理解 ClusterState 设计 | `concepts/cluster-state.md`（待补）|
| 理解 LLM 注入 3 层细节 | `concepts/llm-injection.md`（待补）|
| 知道某个决策**为什么这样定** | [`decisions/`](../decisions/) 下对应 ADR |

---

## 附录：12 个 ADR 决策快速索引

主文档每个 why 都链接到下面对应 ADR：

| ADR | 主题 |
|-----|------|
| [001](../decisions/001-multi-repo-vs-monorepo.md) | monorepo + uv workspace |
| [002](../decisions/002-multi-distribution-vs-single.md) | 多 distribution + 独立 SemVer |
| [003](../decisions/003-namespace-package-pep420.md) | PEP 420 共享 `everalgo.*` namespace |
| [004](../decisions/004-providers-nested-in-llm.md) | providers 内嵌在 `llm/` 子包内 |
| [005](../decisions/005-testing-as-public-subpackage.md) | testing 并入 core 而非独立 dist |
| [006](../decisions/006-clustering-independent-subpackage.md) | clustering 独立工具子包 + 函数式 state |
| [007](../decisions/007-version-compatibility-strategy.md) | HuggingFace 风版本兼容策略 |
| [008](../decisions/008-re-export-vs-client-facade.md) | re-export facade vs Client 类 |
| [009](../decisions/009-naming-convention-llama-index-style.md) | llama-index 风命名规范 |
| [010](../decisions/010-sync-async-dual-interface.md) | sync/async 双接口规范 |
| [011](../decisions/011-protocol-vs-abc.md) | Protocol vs ABC |
| [012](../decisions/012-llm-stack-architecture.md) | LLM 抽象层架构 + 3 层注入 |
