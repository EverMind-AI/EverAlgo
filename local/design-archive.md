# EverAlgo 架构设计文档（Working Draft）

| 字段 | 值 |
|------|-----|
| 版本 | v0.21 — 2026-04-28 |
| 状态 | Working Draft，随讨论迭代 |
| 设计源头 | [Confluence「记忆提取 EverAlgo」](https://npcwork.atlassian.net/wiki/spaces/AI3/pages/2141028371/EverAlgo)（zhanglibin）<br>[Confluence「检索 EverAlgo」](https://npcwork.atlassian.net/wiki/spaces/AI3/pages/2139194126)（zhanglibin）<br>[Confluence「evermem Markdown First For Agent」](https://npcwork.atlassian.net/wiki/spaces/AI3/pages/2157772920)（zhanglibin） |
| 背景理论 | [LLM Wiki vs RAG 总结](https://npcwork.atlassian.net/wiki/spaces/AI3/pages/2154037557) |

本文档只记录**已对齐**的决策与方向，未拍板的全部放进「待讨论」清单，后续逐条推进。

---

## 1. 已拍板

### 1.1 项目定位

- **EverAlgo = 算法同学的开发大本营**。所有记忆提取 / 融合 / 重排等算法策略落在这里，算法团队在这里灵活迭代。
- **对外以无状态接口供上游消费**。library 自身不持有任何业务状态、不连数据库、不读写文件系统。
- **流程编排全部在 evermem**：什么时候调、调用顺序、并发/互斥、持久化到 markdown 文件系统 —— 全部由 evermem 负责。
- EverAlgo 不关心调用方是开源版还是云商业版，两者共用同一份代码。

#### 双主轴：Extract + Rank

EverAlgo 的算法职责分两条主轴，两轴对上游契约完全对称（**无状态、不碰 DB、输入输出都是内存数据结构**）：

| 主轴 | 时机 | 输入 | 输出 | 对应 evermem |
|------|------|------|------|-------------|
| **Extract** | 记忆写入链路 | `MemCell` 等结构化输入单元 | 结构化记忆（Episode / Profile / Case / Skill / ...） | MarkdownWriter 消费 |
| **Rank** | 记忆检索链路 | 多路召回候选 + 预取关联（与具体存储引擎解耦） | 排序好的记忆列表 | 检索 API 消费 |

**关键契约**：Rank 层**不读任何存储**，所有跨记忆关联关系（如 Episode → AtomicFact）必须由 evermem 在 Recall 阶段一并预取传入。Ranker 在内存里做 hierarchy 展开，无任何 DB 调用。（来源：「检索 EverAlgo」文档明确 Recall/evermem vs Rank/EverAlgo 分工。）

### 1.2 子包划分（物理按算法职责 + 外部契约 re-export）

**核心模式**：物理目录按算法职责组织（算法同学迭代友好），外部契约通过 `__init__.py` re-export 暴露（对齐 evermem 设计文档的命名）。两者独立解耦。

**算法库 vs 网络 SDK 选型**：EverAlgo 是算法库（无状态 + 多算子组合），选 **re-export 式 facade**（非 Client 类），属 transformers / pytorch / scikit-learn 阵营。详细优劣分析（含 7 仓库 WebFetch 实证 + 阵营二分逻辑）见 [ADR 008](decisions/008-re-export-vs-client-facade.md)。

**物理布局选型**：物理按算法族组织 + 通过 re-export 对齐外部契约——这是 Python 算法库主流（transformers / pytorch / scikit-learn / pandas / django 5 家实证）。详细见 ADR 008。

#### 子包列表

**产品性子包（3）** —— 每个领域负责一种结构化记忆的**产出**：

| 子包 | 职责 | 对应 evermem 环节 |
|------|------|------------------|
| `user_memory` | 用户侧记忆产出：`Episode / Foresight / AtomicFact / Profile` | 环节 4 / 7 |
| `agent_memory` | agent 侧记忆产出：`AgentCase / AgentSkill` | 环节 4 |
| `knowledge` | 文件型知识 → `KnowledgeMemory` | 环节 8 |

**工具性子包（4）** —— 被多个产品性子包或 evermem 直接消费的横切算子族（不产出记忆类型，是中间转换工具）：

| 子包 | 职责 | 消费者 |
|------|------|--------|
| `parser` | 多模态原始文件 → `ParsedContent`（OCR / ASR / 版面 / 抓取） | knowledge（文件解析后做知识抽取）/ evermem 环节 1（直接调用获取 ParsedContent） |
| `boundary` | 3 种 MemCell 切分 + 共享 `_tokenize` / `_force_split` / LLM 边界 prompt 模板 | user_memory（通过 re-export）/ agent_memory（通过 re-export） |
| `rank` | 4 Ranker + 共享 `fusion.py`（RRF / LR / cosine_to_lr_score / score_propagation） | 检索链路（独立 API，不 re-export 到别处） |
| `clustering` | 双公开函数 `cluster_by_geometry` / `cluster_by_llm` + 值对象 `ClusterState`（centroids / counts / last_ts 三字段，caller 持久化）；输入预计算好的 embedding 向量；state-in / state-out 函数式接口（详见 §2.4 / [ADR 006](decisions/006-clustering-independent-subpackage.md)）| user_memory.profile（episode 簇）/ agent_memory.skill（case 簇）|

**基础设施（3）**：

| 子包 | 职责 |
|------|------|
| `llm` | LLM 调用门面（`chat` / `stream` / `use()` contextmanager + LLMClient Protocol + LLMError 7 子类 + Provider 路由）；内嵌 `llm/providers/<provider>/` 装具体 provider 实现（`openai_compat / anthropic / bedrock`）；**不持有 scene 业务路由**（归 evermem） |
| `prompts` | Prompt validator 机制（占位符 / 长度校验）+ 多语言子模块组织约定；具体 prompt 字符串就近放各子包 `prompts/{en,zh}/<name>.py` 内作为 module-level 常量；evermem 自定义路径：算子 per-call `prompt=` 参数（细粒度，主路径）/ caller monkey-patch 模块常量（启动期粗粒度全局）|
| `types` | 公共数据契约（`Message / MemCell / Episode / ParsedContent / RankInput / RankOutput / ...`） |

**测试辅助**：`testing/`（assertions + fake_llm 两件套，对标 numpy.testing / torch.testing 模式；详见 [ADR 005](decisions/005-testing-as-public-subpackage.md)）

**元信息**：`__init__.py` / `config.py` / `protocols.py`

#### 外部契约对齐机制（re-export）

evermem 文档契约 `everalgo.user_memory.ChatMemCellExtractor` 等通过各产品性子包的 `__init__.py` **re-export** 实现，与**物理路径解耦**。两条访问路径同时有效：

- **evermem 及外部用户**：`from everalgo.user_memory import ChatMemCellExtractor`（对齐设计文档契约）
- **算法同学迭代边界策略**：`from everalgo.boundary.chat import ChatMemCellExtractor`（按算法族，改 prompt / tokenize 就近）

re-export 的 `__init__.py` 写法、`__all__` 声明、跨 distribution 工作原理见 [ADR 008 §实施细节](decisions/008-re-export-vs-client-facade.md)。

> **注**：evermem 文档原文使用 PascalCase namespace `everalgo.UserMemory.*` 不符合 PEP 8，本文档统一矫正为 lowercase package + PascalCase class（详见 [ADR 009](decisions/009-naming-convention-llama-index-style.md)）；evermem 作者侧待同步修订。

#### 目录速览（标注每个子包归属哪个 PyPI distribution，详见 §1.3）

```
everalgo/                           # PEP 420 namespace package（无 __init__.py）
│
├── ┌─ everalgo-core ──────────────────────────────────────────┐
│   │ config.py protocols.py                                    │
│   │ types/       {common, raw, memcell, memories, agent,      │
│   │               parsed, knowledge, rank}.py                 │
│   │ llm/         {__init__, routing, client}.py               │
│   │   └─ providers/  {openai_compat, anthropic, vllm, bedrock}/
│   │ prompts/     validator.py                                 │
│   │              # 各子包 prompts/{en,zh}/<name>.py 自带 module 常量 │
│   │ testing/     {assertions, fake_llm}.py                    │
│   └───────────────────────────────────────────────────────────┘
│
├── ┌─ everalgo-parser ────────────────────────────────────────┐
│   │ parser/     {image, audio, document, video, url}.py + prompts/
│   └───────────────────────────────────────────────────────────┘
│
├── ┌─ everalgo-boundary ──────────────────────────────────────┐
│   │ boundary/   {chat, workspace, agent}.py                   │
│   │             {_tokenize, _force_split}.py + prompts/       │
│   └───────────────────────────────────────────────────────────┘
│
├── ┌─ everalgo-rank ──────────────────────────────────────────┐
│   │ rank/       {episodic, profile, case, skill}.py        # 4 业务 facade（evermem 调用面）│
│   │             {fusion, weight, rerank}.py                 # 算法工具（算法同学迭代面，4 facade 内部组合调用）│
│   │             prompts/                                     │
│   └───────────────────────────────────────────────────────────┘
│
├── ┌─ everalgo-clustering ────────────────────────────────────┐
│   │ clustering/ _algorithm.py + prompts/                       │
│   │             # 双公开函数 cluster_by_geometry / cluster_by_llm │
│   │             # + 值对象 ClusterState，详见 §2.4             │
│   └───────────────────────────────────────────────────────────┘
│
├── ┌─ everalgo-user-memory（re-export from boundary）─────────┐
│   │ user_memory/  __init__.py (re-export Chat/WorkspaceMemCellExtractor)
│   │               {episode, foresight, atomic_fact, profile}.py + prompts/
│   └───────────────────────────────────────────────────────────┘
│
├── ┌─ everalgo-agent-memory（re-export from boundary）────────┐
│   │ agent_memory/ __init__.py (re-export AgentMemCellExtractor)
│   │               {case, skill}.py + prompts/                 │
│   └───────────────────────────────────────────────────────────┘
│
└── ┌─ everalgo-knowledge ─────────────────────────────────────┐
    │ knowledge/  __init__.py + {extractor}.py + prompts/       │
    └───────────────────────────────────────────────────────────┘
```

每个 distribution 自带 `pyproject.toml`（独立 version + 独立 dependencies）。

命名原则：**平坦胜嵌套（PEP 20）**。所有子包（产品性 / 工具性 / 基础设施）并列顶层，不藏在 `common/` / `utils/` / `infra/` 下。

Python 术语澄清：

| 概念 | 是什么 |
|------|------|
| module | 单个 `.py` 文件 |
| package | 含 `__init__.py` 的目录 |
| distribution | PyPI 的发布单位，`pip install <X>` 装的就是它 |

**11 个 subpackage**（3 产品性 + 4 工具性 + 3 基础设施 + 1 测试辅助）按物理布局组织 Python import 路径；对外发布为 **8 个独立 PyPI distribution**——其中 `everalgo-core` 一个 dist 打包了 `llm` / `prompts` / `types` / `testing` 共 4 个 subpackage，其他 7 个产品性/工具性 subpackage 各对应一个 dist。通过 PEP 420 namespace package 共享 `everalgo.*` import 路径（详见 §1.3）。

> ✅ **设计自检（详细优劣分析与行业实证见 [decisions/](decisions/) ADR）**
>
> 每条决策的"详细优劣分析 + 对 EverAlgo 适配度评估 + 行业实证印证"放到独立 ADR 文档，主文档只列简短结论 + 链接：
>
> - **Why re-export 式 facade（非 Client 类）**：EverAlgo 是算法库（无状态 + 多算子组合），属 transformers / pytorch / scikit-learn 阵营；OpenAI / Anthropic SDK 的 Client 类适配的是网络 API SDK 场景。详细：[ADR 008](decisions/008-re-export-vs-client-facade.md)
> - **Why 物理按算法族 + 外部 re-export**：算法同学跨包迭代速度（H3）+ 共享底层（tokenize / fusion / centroid）需就近迭代；evermem 文档契约路径通过 `__init__.py` re-export 无损对齐。
> - **Why `rank/` 独立且不 re-export**：「检索 EverAlgo」文档自身组织即 4 Ranker；外部契约与物理路径一致；Ranker 输入（多路召回候选）与 Extractor（MemCell / ParsedContent）完全不同，物理独立避免类型/时机混淆。
> - **Why 4 业务 facade + 算法工具双层**：`rank/{episodic, profile, case, skill}` 是业务 facade（evermem 调用面，封装业务参数如 `quality_score` 字段映射 / `case_rerank` LLM scene 等不外泄）；`rank/{fusion, weight, rerank}` 是算法工具（算法同学迭代面，4 facade 内部组合调用 + 算法同学新增 ranker 时直接 import 复用）。Case / Skill 高度同构（都 `fusion → weight → (可选) rerank`），双层抽象消除代码冗余；类比 LangChain `BaseRetriever` 接口 + 实现 / LlamaIndex `BaseNodePostprocessor` 接口 + 实现。EverAlgo 按业务记忆类型分接口是少数派（业界 LangChain/LlamaIndex/DSPy 都按算法策略分），合理性来自封装边界（业务参数藏在 EverAlgo 内部，evermem 调用对齐 memory_type 分支无需懂 rank 细节）。
> - **Why `clustering/` 独立工具性子包**：user_memory.profile 按 episode 簇上下文做增量编辑 + agent_memory.skill 按 case 簇聚合都需聚类，独立子包避免代码重复。详细：[ADR 006](decisions/006-clustering-independent-subpackage.md)
> - **Why providers 内嵌在 `llm/` 子包内**：明星 LLM 库（litellm / instructor / dspy / llama-index）100% 是"LLM 抽象 + providers 收在一个根子包内"模式。详细：[ADR 004](decisions/004-providers-nested-in-llm.md)
> - **Why 删 `embed/` facade 和 `providers/embed/`**：BOSS 确认 EverAlgo 不负责 embed；embedding 由 evermem 外部调用，Rank 层向量由 evermem 预取传入。
> - **Why prompts 不进 providers**：prompts 是算法 IP（与算法绑定 / 调优 prompt = 调算法），不是可替换 provider；具体 prompt 字符串就近放各子包 `prompts/{en,zh}/<name>.py` 作为 Python 模块常量；自定义路径见 §1.4 prompt 实现段

### 1.3 发布策略：多 distribution + 独立版本号（namespace package / PEP 420）

EverAlgo 拆为 **7 个独立 PyPI distribution**，各包独立版本号、按 SemVer 各自演进；通过 **PEP 420 命名空间包**共享 `everalgo.*` 顶层 import path（用户视角无割裂感）。

#### 行业参照矩阵（按维度独立选最优，非任意挑选）

多 distribution 项目的设计涉及 5 个独立维度，没有一个明星库在所有维度都最优。按 EverAlgo 三条硬约束（① evermem 文档契约 `everalgo.user_memory.*` ② BOSS"升级 A 不动 B" ③ 算法库定位）逐维选最适合者：

| 维度 | HuggingFace 全家桶 | llama-index | langchain | EverAlgo 选 | 选择依据（硬约束） |
|------|---------------------|-------------|-----------|-------------|-------------------|
| **命名是否带系列前缀** | ❌ 独立词（`transformers` / `datasets`）| ✅ `llama-index-*` | ✅ `langchain-*` | **llama-index** | PyPI 上 `core` / `parser` / `boundary` 等通用词被占；EverAlgo 系列需可识别 |
| **import namespace** | A 风格独立顶层（`import transformers`）| **B 风格共享 PEP 420**（`llama_index.core`）| A 风格独立顶层（`import langchain_core`）| **llama-index** | evermem 文档契约 `everalgo.user_memory.*` 要求共享 namespace |
| **仓库形态** | multi-repo（4 独立仓，仅适用业务独立场景）| **monorepo**（顶层 `llama-index-*` 多包）| **monorepo**（`libs/{core, partners/*, ...}`）| **langchain / llama-index** | AI 圈 "core + N 紧密联动包" 业务结构主流；EverAlgo 业务结构同构（`everalgo-core` + 7 紧密依赖产品/工具包），与 LangChain `langchain-core` + partners、LlamaIndex `llama-index-core` + integrations 完全对位。HuggingFace multi-repo 仅适用 transformers / datasets / accelerate 这种业务独立场景，不适配 EverAlgo |
| **版本约束策略** | **宽松（`>=X.Y,<2.0` 或无 upper）+ 严守 SemVer** | 宽松（无 upper） | partner 包跨整 minor 段 + 部分同步 release | **HuggingFace** | diamond dependency 自然消解 + 不需同步 release |
| **兄弟包互依赖** | **不在 `install_requires`**（仅 extras）| 有少量 | 有 | **HuggingFace** | EverAlgo 同层兄弟（产品性 3 包 + 工具性 4 包）各自横向独立 |

**总览**：命名规范 + namespace + 仓库形态维度参照 **langchain / llama-index**（AI 圈 monorepo 主流）；版本管理 + 互依赖维度参照 **HuggingFace**。

**不存在"主参照"概念**——各维度独立选最优是 Python 多 distribution 项目的工程常态（langchain 自己也是混合：命名带前缀像 huggingface-hub 模式，仓库管理又用 monorepo 像 sklearn 模式）。每个维度的选择都能溯到 EverAlgo 三条硬约束之一，不是任意挑选有利证据。后续各小节自检不再重复"为什么参照 X"，只列具体规范和实证细节。

#### 拆分清单

| Distribution | 物理目录（namespace package 内） | import 路径 | 依赖 distribution |
|--------------|------------------------------|-------------|-------------------|
| `everalgo-core` | `everalgo/{types, llm（含 llm/providers/）, prompts, config.py, protocols.py, testing}/` | `from everalgo.{types,llm,prompts,config,protocols,testing} import ...` | — |
| `everalgo-boundary` | `everalgo/boundary/` | `from everalgo.boundary import ...` | core |
| `everalgo-rank` | `everalgo/rank/` | `from everalgo.rank import ...` | core |
| `everalgo-clustering` | `everalgo/clustering/` | `from everalgo.clustering import ...` | core |
| `everalgo-parser` | `everalgo/parser/` | `from everalgo.parser import ...` | core |
| `everalgo-user-memory` | `everalgo/user_memory/` | `from everalgo.user_memory import ...` | core, boundary, clustering |
| `everalgo-agent-memory` | `everalgo/agent_memory/` | `from everalgo.agent_memory import ...` | core, boundary, clustering |
| `everalgo-knowledge` | `everalgo/knowledge/` | `from everalgo.knowledge import ...` | core, parser |

#### 依赖关系图

```
                          everalgo-core
                  （types/llm（含 providers）/prompts/config/protocols/testing）
                              ▲
        ┌───────────┬─────────┴───────────┬──────────┐
        │           │                     │          │
   everalgo-   everalgo-             everalgo-   everalgo-
   boundary    clustering              rank       parser
        ▲          ▲                                  ▲
        │          │                                  │
        ├──────────┤                                  │
        │          │                                  │
   everalgo-   everalgo-                       everalgo-
   user-       agent-                          knowledge
   memory      memory
```

**箭头说明**：`▲` 指向被依赖方（即"上方是被依赖包"）。例如 `everalgo-knowledge ▲ everalgo-parser` 表示 knowledge 依赖 parser。

满足 BOSS "升级 A 不动 B" 愿景：
- 升级 `everalgo-user-memory` 的 Episode 算法 → `everalgo-agent-memory` / `everalgo-knowledge` 不动
- `everalgo-boundary` 接口变更（major bump）→ user-memory / agent-memory 在自己 `pyproject.toml` 用宽松 SemVer 约束（`everalgo-boundary>=0.5.0,<2.0.0`，仿 HuggingFace transformers 对 hub）锁定，自主决定何时升

#### 版本兼容策略（参照 HuggingFace 全家桶）

**前置概念：SemVer（Semantic Versioning，语义化版本，[semver.org](https://semver.org)）**——版本号 `MAJOR.MINOR.PATCH`：MAJOR 用于不兼容变更（breaking）、MINOR 用于向后兼容的新功能、PATCH 用于向后兼容的 bug 修复。`0.x.x` 阶段视作不稳定（任何变更可能 breaking），`1.0.0` 才是稳定 API 承诺起点。本策略要求 `everalgo-core` **严守 SemVer**——minor/patch 不 breaking，breaking 集中 major bump，这是下游宽松约束能 work 的根本前提。

HuggingFace 实证（WebFetch 2026-04-23 核验 4 个 setup.py）：

| 包 | 对共同 base 包的约束 | 兄弟包间互依赖 |
|----|---------------------|-----------------|
| **transformers** | `huggingface-hub>=1.5.0,<2.0` | datasets / accelerate **不在 install_requires**（仅 extras） |
| **datasets** | `huggingface-hub>=0.25.0,<2.0` | 不依赖 transformers / accelerate |
| **accelerate** | `huggingface_hub>=0.21.0`（**无 upper**）| 不依赖 transformers / datasets |
| **huggingface_hub** | — | — |

提炼出 4 条策略，EverAlgo 全部采纳：

1. **同层兄弟包之间不互相依赖**——产品性 3 包（user_memory / agent_memory / knowledge）之间 + 工具性 4 包（boundary / clustering / rank / parser）之间各自横向独立，**互相不在 `install_requires`**。跨层依赖（产品包依赖工具包，如 user_memory 依赖 boundary / clustering）不属"兄弟互依赖"，是合理的拓扑下行依赖。EverAlgo 设计已天然符合（user_memory 不依赖 agent_memory；boundary 不依赖 clustering）。
2. **下游对 base 包用宽松约束**：`everalgo-core>=X.Y,<2.0`（仿 transformers 对 hub）或 `>=X.Y`（仿 accelerate 对 hub）。**禁止 `<X.Y+1` 这种紧约束**。
3. **`everalgo-core` 严守 SemVer**：minor + patch 必须向后兼容；breaking 集中 major bump。这是宽松约束能 work 的前提（HuggingFace hub 多年实战兑现这个承诺）。
4. **不强制同步 release**：每个 distribution 独立演进，不需 monorepo 同步 bump。这与 BOSS"升级 A 不动 B"完全契合。

**Diamond dependency 自然消解**：
- 假设 `everalgo-user-memory 0.5` 写 `everalgo-core>=0.1.0,<2.0.0`
- `everalgo-agent-memory 0.6` 写 `everalgo-core>=0.3.0,<2.0.0`
- 同时安装时 pip 解析 → 找到 `everalgo-core 0.3+` 任一兼容版本，**两者并存无冲突**
- HuggingFace 全家桶千万级用户量级实证此模式有效

**用户侧 lockfile 兜底**（推荐做法）：用 `uv lock` / `poetry lock` 锁一组兼容版本到 `uv.lock`；升级时局部 `uv lock --upgrade-package everalgo-user-memory`。

#### 仓库管理：monorepo + uv workspace + PEP 420 namespace（3 层独立举证）

8 个 distribution 收在**单一 GitLab 仓库** `<gitlab>/<group>/everalgo`，使用 `uv workspace` 管理多包开发；8 个 dist 共享 `everalgo.*` namespace 用 **PEP 420 native namespace** 实现。这是 3 个独立维度，参照库各不同：

- **① monorepo 形态**：AI 圈"core + N 紧密联动包"业务结构主流做法——LangChain（`libs/{core, partners/*, ...}`）和 LlamaIndex（顶层 `llama-index-*` 多目录）均采用此布局。**注**：LangChain / LlamaIndex 自身**不**用 uv workspace（LangChain `libs/` 子包独立 venv + 各自 `uv.lock`；LlamaIndex 子包靠 Pants 编排），它们只作 monorepo 形态实证。
- **② uv workspace 工具**：参照 **Apache Airflow**（100+ workspace members + 单 `uv.lock`，CONTRIBUTING 推荐 `uv sync --all-packages`）和 **pydantic-ai**（`pydantic_ai_slim` / `pydantic_evals` / `pydantic_graph` 多 dist + workspace）。**注**：Airflow / pydantic-ai 只作 uv workspace 工具实证——它们的 namespace 实现不是 PEP 420（见 ③），是另一回事。
- **③ PEP 420 native namespace 共享**：每个 `packages/everalgo-*/src/everalgo/` 顶层**省略 `__init__.py`**（仅子包目录有），通过 [PEP 420](https://peps.python.org/pep-0420/) 实现多 dist 共享 `everalgo.*` import path。**PyPA 官方推荐写法**（[packaging.python.org Native namespace packages](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/#native-namespace-packages) 原话："This is recommended if packages in your namespace only ever need to support Python 3 and installation via pip"——EverAlgo 满足两条：`requires-python=">=3.12"` + uv/pip 安装）。**工业实证**（实测各项目 `__init__.py` 状态）：
  - **google-cloud-*** 100+ dist 共享 `google.cloud.*`（PEP 420 native，验证 `google/__init__.py` 与 `google/cloud/__init__.py` 均 404）—— 工业级最大实证
  - **sphinxcontrib-*** 6 个 dist 共享 `sphinxcontrib.*`（PyPA 官方分发的 Sphinx 扩展，PEP 420 native）
  - [PyPA 官方示例 sample-namespace-packages](https://github.com/pypa/sample-namespace-packages) `native/` 子目录
  - **反例（不是 PEP 420）**：Apache Airflow（`airflow/__init__.py` 末行 `__path__ = pkgutil.extend_path(__path__, __name__)`）和 Azure SDK 用的是 **pkgutil-style legacy namespace**（PEP 420 之前的旧式写法，PyPA 已 discouraged 但保留 Py2 兼容）；pydantic-ai 是 3 个**独立** namespace（`pydantic_ai` / `pydantic_evals` / `pydantic_graph` 各自有 `__init__.py`），不是同 namespace 多 dist。这 3 家只作"形态 / 工具"实证，不作 PEP 420 实证。

仓内目录结构：

```
<gitlab>/<group>/everalgo/
├── pyproject.toml              # workspace 根（uv workspace 配置）
├── uv.lock                     # 整 workspace 共享 lockfile
├── .gitlab-ci.yml              # 单仓 CI（path-based trigger 跑变动包）
├── packages/
│   ├── everalgo-core/
│   │   ├── pyproject.toml      # 独立 dist + 独立 SemVer
│   │   └── src/everalgo/{types, llm, prompts, ..., testing}/
│   ├── everalgo-parser/
│   │   ├── pyproject.toml
│   │   └── src/everalgo/parser/
│   ├── everalgo-boundary/
│   ├── everalgo-rank/
│   ├── everalgo-clustering/
│   ├── everalgo-user-memory/
│   ├── everalgo-agent-memory/
│   └── everalgo-knowledge/
└── docs/
```

每个 `packages/everalgo-*/` 子目录独立含 `pyproject.toml` + 独立 SemVer + 独立 PyPI 发布（仿 LangChain `libs/partners/openai/` 模式）。

**关键澄清**：monorepo 是**仓库形态**决定开发流程便利度，与**发布粒度**正交——
- 7 个 distribution 仍各自独立 PyPI dist + 独立 SemVer + 独立版本演进
- "升级 A 不动 B" 完全成立（PyPI 端 dist 独立，与仓内 layout 无关）
- monorepo 只影响开发流程：`uv sync --all-packages` 一键拉齐所有 dev 依赖、跨包改动一次 MR 完成

> ✅ **Why monorepo + uv workspace**
>
> 简短结论：跨包重构原子性（H6 v0.x 演化阶段）+ 算法同学跨包迭代速度（H3）+ 新人上手三个核心需求强烈倾向 monorepo；multi-repo 优势（物理隔离 / 不同 release cadence / 对外贡献门槛低）EverAlgo 全部用不上；CI 隔离用 path-based trigger 等价补齐。AI 圈业务结构同构的项目（LangChain / LlamaIndex / Apache Airflow / Dagster / Prefect）都收敛 monorepo，HuggingFace multi-repo 适配的是业务独立场景与 EverAlgo 不同。
>
> 详细优劣分析与适配度评估见 [ADR 001](decisions/001-multi-repo-vs-monorepo.md)。

#### 命名规范（参照 llama-index）

| 维度 | 规范 | 例 |
|------|------|------|
| Distribution（PyPI）| 全 dash | `everalgo-user-memory` |
| import 路径（dot 分隔）| 共享 `everalgo.*` namespace | `everalgo.user_memory` |
| 物理目录 | underscore | `everalgo/user_memory/` |
| 顶层 `everalgo/` | **PEP 420 隐式命名空间**，无 `__init__.py` | — |
| 子包 `everalgo/<name>/` | regular package，需 `__init__.py`（承载 re-export） | — |

#### Re-export 在多 distribution 下仍然成立

```python
# everalgo-user-memory 包的 everalgo/user_memory/__init__.py
from everalgo.boundary.chat      import ChatMemCellExtractor      # 来自 everalgo-boundary
from everalgo.boundary.workspace import WorkspaceMemCellExtractor
from everalgo.user_memory.episode    import EpisodeExtractor
from everalgo.user_memory.foresight  import ForesightExtractor
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.profile    import ProfileExtractor

__all__ = [
    "ChatMemCellExtractor", "WorkspaceMemCellExtractor",
    "EpisodeExtractor", "ForesightExtractor",
    "AtomicFactExtractor", "ProfileExtractor",
]
```

`everalgo-user-memory` 的 `pyproject.toml` 在 `dependencies` 里宽松约束 `everalgo-boundary>=X.Y,<2.0`，pip 自动拉齐（约束写法依据见后文「版本兼容策略」）。

#### 安装样例

**生产端（用户 pip install 单 dist）**：

```bash
# 完整算法链路（对话场景：边界 + Episode + Foresight + AtomicFact + Profile）
pip install everalgo-user-memory          # 自动拉 everalgo-core + everalgo-boundary + everalgo-clustering

# Agent 场景
pip install everalgo-agent-memory         # 自动拉 everalgo-core + everalgo-boundary + everalgo-clustering

# 检索 Rank
pip install everalgo-rank                 # 自动拉 everalgo-core

# 多模态知识录入
pip install everalgo-knowledge            # 自动拉 everalgo-core + everalgo-parser
```

**开发端（算法同学 clone monorepo + uv workspace）**：

```bash
# 一次 clone 拿到所有 7 包源码
git clone <gitlab>/<group>/everalgo
cd everalgo

# 一键 editable install 整 workspace（所有 7 包到共享 venv）
uv sync --all-packages                    # workspace 模式（仿 Apache Airflow / pydantic-ai；
                                           # LangChain / LlamaIndex 是 monorepo 形态参照，不用 workspace）
                                           # 改 everalgo-boundary 任何代码 everalgo-user-memory 即时生效

# 仅 sync 单个包的依赖（聚焦开发某一个包时）
uv sync --package everalgo-user-memory
```

不提供 `everalgo` meta-package（与"独立升级"意图相符；用户按需装）。

#### 单包 pyproject 草案骨架（以 everalgo-user-memory 为例）

```toml
[project]
name = "everalgo-user-memory"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
  "everalgo-core>=0.1.0,<2.0.0",        # 跨整个 major 段（仿 HuggingFace transformers 对 huggingface-hub 写法）
  "everalgo-boundary>=0.1.0,<2.0.0",
  "everalgo-clustering>=0.1.0,<2.0.0",  # episode 簇用（profile 增量编辑上下文）
]

[project.optional-dependencies]
dev = ["pytest", "pytest-asyncio", "respx", "mypy", "ruff"]
```

每个 distribution 都有自己 `pyproject.toml`，独立 version、独立依赖约束。

> ✅ **设计自检（详细优劣分析与行业实证见 [decisions/](decisions/) ADR）**
>
> - **Why 多 distribution + 独立 SemVer**：用户独立升级 A 不动 B（H2）+ 子模块独立演化节奏（H6）+ 按需安装重依赖（parser）。详细：[ADR 002](decisions/002-multi-distribution-vs-single.md)
> - **Why namespace package（PEP 420 共享 `everalgo.*`）**：与 evermem 文档契约 `everalgo.user_memory.*` 兼容（H1）+ 系列归属感（H7）+ 多包共存 import 整洁。详细：[ADR 003](decisions/003-namespace-package-pep420.md)
> - **Why monorepo + uv workspace + PEP 420 namespace**：3 维度独立举证（详 §1.3 仓库管理段）——monorepo 形态参 LangChain / LlamaIndex / Airflow；uv workspace 工具参 Apache Airflow（100+ members）/ pydantic-ai；PEP 420 native namespace 参 google-cloud-*（100+ dist）/ sphinxcontrib-* / PyPA 官方示例（PyPA 官方原话推荐 Py3-only + pip-only 项目用 PEP 420）。**反例区分**：Airflow 用 pkgutil-style legacy namespace 不是 PEP 420，仅作 uv workspace 工具实证，不作 namespace 实证。详细：[ADR 001](decisions/001-multi-repo-vs-monorepo.md)
> - **Why 命名 dash-dist + dot-import + lowercase 模块**：与 llama-index 模式对齐 + 兼容 evermem 文档契约 + 系列前缀 PyPI 可识别 + PEP 8 合规。详细：[ADR 009](decisions/009-naming-convention-llama-index-style.md)
> - **Why HuggingFace 风版本兼容（宽松约束 + 严守 SemVer + 不强制同步 release）**：diamond dependency 自然消解 + 各下游独立演化（H2）。详细：[ADR 007](decisions/007-version-compatibility-strategy.md)
> - **Why testing 并入 core 而非独立 distribution**：与 numpy / pandas / pytorch 模式一致；testing 与 core 紧密绑定，独立 dist 反而带来 diamond 风险。详细：[ADR 005](decisions/005-testing-as-public-subpackage.md)
> - **Why 不加 meta package `everalgo`**：① 主用户 evermem 按场景调 user_memory / agent_memory / knowledge / rank 单装即可，"全装"是低频场景（仅算法同学或 demo / 集成测试用）；② 全装手列 4 个顶层 dist 一行命令搞定（`pip install everalgo-user-memory everalgo-agent-memory everalgo-knowledge everalgo-rank` 自动拉齐 8 个 dist）；③ meta 包要持续维护对各 dist 的兼容性约束（即便用宽松约束，新增 dist / breaking 升级时仍要发新 meta），边际收益低于维护成本；④ HuggingFace transformers/datasets/accelerate 同模式不提供 meta。**注**：早期 "meta 必 pin 整套版本" 表述不准——LlamaIndex `llama-index` / LangChain `langchain` meta 都用宽松约束（`>=X.Y,<next_major`），不 pin 死版本，"升级 A 不动 B"（H2）在 meta + 宽松约束模式下仍兑现。这条理由本身不构成反对 meta 的依据。
>
> **行业参照矩阵**见本节顶部，5 维度 × 3 家明星库对比与 EverAlgo 选择溯源。

### 1.4 代码风格：Pythonic 直写

> **EverAlgo 命名强契约**：方法名带 `a` 前缀（`aextract` / `arank` / `adetect` / `aparse`）= **native async**（含 LLM / 外部 I/O），调用必须 `await`；不带 `a` 前缀（`rank` / `extract` / `count_tokens` / `rrf`）= **sync**（纯计算 / 无 I/O），调用不要 `await`。**看名字即知接口形态，无需查算子表**。同 DSPy `acall`/`aforward` / litellm `acompletion` / instructor `AsyncInstructor` 命名约定。
>
> 用户区分 await/no-await 由命名契约消除：场景 E 中 `await rank.episodic.arank(...)`（带 a → async）与 `rank.profile.rank(...)`（无 a → sync）的混用，看名字即知规则，零认知负担。

- **I/O 算子 async-first + sync 桥接**（主用户 evermem = FastAPI 异步服务，async 是主战场）
  - 主路径：`await everalgo.user_memory.EpisodeExtractor().aextract(memcell)`
  - sync 桥接：`everalgo.user_memory.EpisodeExtractor().extract(memcell)` —— **仅限非 event loop 环境**（CLI 脚本 / 单元测试）；Jupyter / FastAPI / 任何 `async def` 上下文须用 `await aextract(...)`
- **纯计算算子只提供 sync，不实现 async 版本**（`fusion.rrf` / `_tokenize.count_tokens` / clustering 距离等）
  - **Python 官方 / FastAPI / NumPy 文档共识**：asyncio 为 I/O-bound 设计，CPU-bound 写 `def` 不写 `async def`
  - **9 项目实证一致**：numpy / pandas / sklearn / pytorch / scipy / litellm / llama-index / OpenAI SDK / httpx 100% 不为纯计算提供 async（唯一反例 **langchain LCEL** 为 chain 接口统一 `invoke / ainvoke` 接受 thread pool 性能代价；EverAlgo 非 chain 框架场景不同）
  - 轻量纯计算（毫秒级）在 async 上下文直接 sync 调用安全；**若未来某算子计算时长超 ~100ms**（如大批量向量相似度），caller 用 `run_in_executor` / `ProcessPoolExecutor` 包装隔离（保持算子 API sync `def` 不变）。详见 §2.3 + [ADR 010](decisions/010-sync-async-dual-interface.md)
- **Prompt 是 Python 字符串模块**（如 `prompts/en/cluster_decision.py` 内 `CLUSTER_DECISION_PROMPT = "..."`），**不外置 `.md` / `.yaml` / `.toml`**；与算法库阵营 DSPy / LlamaIndex / instructor / mem0 / evermem 现状 5/5 一致；多语言通过子模块组织（`prompts/en/` + `prompts/zh/`）；改 prompt = 改 `.py` 字符串。**端到端框架阵营**（LangChain / CrewAI / Semantic Kernel）外置 YAML / Jinja2 不适配 EverAlgo 算法库定位
- **evermem 自定义 prompt** 两条路径（KISS，无额外 framework）：
  - **算子 per-call `prompt=` 参数**（细粒度主路径）：`cluster_by_llm(..., prompt=my_prompt)` 单次注入；适合少量定制 / A/B 测试。已在 §2.4 算子签名预留
  - **caller monkey-patch 模块常量**（粗粒度全局）：`from everalgo.clustering.prompts.en import cluster_decision; cluster_decision.CLUSTER_DECISION_PROMPT = "..."` 启动期一次性覆盖；适合项目级全局替换。LlamaIndex `update_prompts` / HuggingFace `tokenizer.chat_template = "..."` 同款
  - **不引入** `prompt_dir` 参数 / `everalgo.configure(prompts={...})` 全局 default 层 / scoped contextmanager / setter API —— 过度设计，per-call + monkey-patch 已覆盖 100% 场景
- **Protocol 作为类型注解**（[PEP 544](https://peps.python.org/pep-0544/) structural subtyping）—— EverAlgo 算子无状态（H4），实现类**不需显式继承**（DSPy `CodeInterpreter` / LlamaIndex `VectorStore` / instructor `*Handler` 同模式）。详见 [ADR 011](decisions/011-protocol-vs-abc.md)
- **测试通过 `monkeypatch` / `unittest.mock.patch` / contextmanager** 覆盖全局（DSPy `with dspy.context(lm=...)` 同款 scoped 替换）

> ✅ **设计自检**
> - **Why async-first + asgiref 桥接 sync**：主用户 evermem = FastAPI 异步服务，async 接口必须 native（不能 thread pool wrap——sync-first + thread pool 32 worker 模式在 100 QPS / 1s LLM 场景吞吐降到 ~32 RPS，比 native async ~100 RPS 差 ~3x）；sync 接口由 `asgiref.async_to_sync` 单实现派生，仅限非 event loop 环境调用。详细 9 项目实证 + B1/B2/B3 优劣对比见 [ADR 010](decisions/010-sync-async-dual-interface.md)
> - **Why 纯计算算子同步**：无外部 I/O，async 是 overhead；与 numpy / scipy 数值计算同步惯例一致。
> - **Why 模块级函数 + 全局配置（不用 Client 类）**：算法库阵营选择。详细：[ADR 008](decisions/008-re-export-vs-client-facade.md)
> - **Why Protocol（不用 ABC）**：H4 无状态算子 → DSPy / LlamaIndex 插件接口同推理路径；LangChain 选 ABC 是 chain 持状态 + lifecycle hooks 场景，与 EverAlgo 不同。详细：[ADR 011](decisions/011-protocol-vs-abc.md)

---

## 2. 已对齐方向（细节待推进）

以下方向已在讨论中形成共识或默认采纳，具体签名 / 类型定义 / 目录细节待后续对齐。

### 2.1 上游 evermem 的典型调用形态

EverAlgo 算子提供 **3 层注入路径**（DSPy `dspy.settings.configure` + `dspy.context` + `predictor(..., lm=...)` 同款，优先级 **per-call > scoped > default**）：

启动期一次（**可选**——dev / 测试 / Jupyter 必备；生产 evermem 可不调用，强制全部显式注入）：

```python
import everalgo
from everalgo.llm import LLMConfig

everalgo.configure(
    llm=LLMConfig(                                       # 全局 default（兜底，未 wrap 调用走这）
        provider="openai_compat",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_KEY"],
        model="openai/gpt-4.1-mini",
    ),
)
```

EverAlgo **不持有 scene 路由**——业务编排（哪个算子用哪个模型）归 evermem。evermem 端 `SceneRouter` 持有 scene → client 映射，调用算子时通过 per-call 参数 / scoped contextmanager 注入。完整 3 层注入机制见 §2.5。

evermem 端先持有 SceneRouter（业务编排层）：

```python
scene_router = SceneRouter({
    "boundary":    OpenAICompatClient(model="qwen/qwen3-235b"),
    "episode":     OpenAICompatClient(model="anthropic/claude-sonnet-4-6"),
    "foresight":   OpenAICompatClient(model="openai/gpt-4.1-mini"),
    "atomic_fact": OpenAICompatClient(model="openai/gpt-4.1-mini"),
    "profile":     OpenAICompatClient(model="openai/gpt-4.1-mini"),
    "agent_case":  OpenAICompatClient(model="anthropic/claude-sonnet-4-6"),
    "agent_skill": OpenAICompatClient(model="openai/gpt-4.1-mini"),
    "knowledge":   OpenAICompatClient(model="openai/gpt-4.1-mini"),
    "rerank":      OpenAICompatClient(model="openai/gpt-4o-mini"),
})
```

运行期五个典型场景（示范两条访问路径，per-call `llm=` 主路径 + scoped wrap 批量切换）：

```python
from everalgo import parser, user_memory, agent_memory, knowledge, rank, boundary

# === 外部用户路径（evermem 契约，对齐设计文档） ===

# 场景 A: 对话 → User Memory（per-call llm 注入主路径）
memcells = await user_memory.ChatMemCellExtractor().adetect(
    messages, llm=scene_router.get("boundary"))
for memcell in memcells:
    episodes   = await user_memory.EpisodeExtractor().aextract(
        memcell, llm=scene_router.get("episode"))
    foresights = await user_memory.ForesightExtractor().aextract(
        memcell, llm=scene_router.get("foresight"))
    facts      = await user_memory.AtomicFactExtractor().aextract(
        memcell, llm=scene_router.get("atomic_fact"))

# 场景 B: Workspace 数据（Jira/Email/...）
memcells = await user_memory.WorkspaceMemCellExtractor().adetect(
    jira_ticket, llm=scene_router.get("boundary"))
for memcell in memcells:
    episodes = await user_memory.EpisodeExtractor().aextract(
        memcell, llm=scene_router.get("episode"))

# 场景 C: Agent 执行轨迹（asyncio.gather 并行，每个传 llm）
memcells = await agent_memory.AgentMemCellExtractor().adetect(
    agent_trace, llm=scene_router.get("boundary"))
for memcell in memcells:
    cases, skill_deltas = await asyncio.gather(
        agent_memory.AgentCaseExtractor().aextract(
            memcell, llm=scene_router.get("agent_case")),
        agent_memory.AgentSkillExtractor().aextract(
            memcell, llm=scene_router.get("agent_skill"), existing=known_skills),
    )

# 场景 D: 多模态文件 → Knowledge
parsed   = await parser.aparse(RawFile(uri=..., mime="application/pdf"))
memories = await knowledge.KnowledgeExtractor().aextract(
    parsed, llm=scene_router.get("knowledge"))

# 场景 E: 检索 Rank — 同 client 批量调用用 scoped wrap 减少重复传参
async with everalgo.llm.use(scene_router.get("rerank")):                    # scoped 批量切换
    ranked_eps    = await rank.episodic.arank(rank_input)                   # fusion → MaxHeap → ep→fact 展开 → rerank
    ranked_cases  = await rank.case.arank(rank_input)                       # fusion → quality_score 加权 → rerank
    ranked_skills = await rank.skill.arank(rank_input)                      # fusion → maturity + confidence 加权
ranked_profs  = rank.profile.rank(rank_input)                               # 同步纯计算（cosine → threshold → 去重，无 LLM）
merged        = rank.fusion.rrf(vec_hits, kw_hits)                          # 同步纯计算

# === 算法同学路径（物理路径，改 boundary 策略时用） ===
# 3 种 MemCellExtractor 共享的 tokenize / prompt / force_split 都在 boundary/ 下
from everalgo.boundary import chat, workspace, agent as agent_boundary
from everalgo.boundary._tokenize import count_tokens, force_split
# 改边界 prompt / 调 tokenize 窗口 / 改切分策略，都改 boundary/ 一处
```

### 2.2 I/O 形态的约定

- **所有提取/切分类算子统一返回 `list[T]`**（`gather` 友好，代码直白）
  - 边界检测算子（`*MemCellExtractor.adetect`）返回 `list[MemCell]` —— 与 LlamaIndex `NodeParser.get_nodes_from_documents() -> List[BaseNode]` / LangChain `TextSplitter.split_documents() -> List[Document]` 同款，**切分类算子主流不流式**
  - 未来若需要 LLM streaming 边切边消费，可加 `astream_detect()` 互补接口（litellm `acompletion(stream=True)` 模式），**不替换** `adetect()`
- Fusion / Tokenizer 等纯计算算子是**同步函数**
- **错误处理：依赖底层 SDK 默认重试，高可用编排不属于开源版需求**（与 LangChain Core 哲学一致）
  - EverAlgo 库自身**不加额外重试层**；OpenAI / Anthropic SDK 内置 `DEFAULT_MAX_RETRIES = 2`（覆盖 connection error / 408 / 409 / 429 / 5xx + `x-should-retry` header）—— 这 2 次重试本来就在，不显式关闭即生效
  - 跨 Provider fallback / multi-key 轮转 / 长时间退避 / 配额降级 / 租户级路由策略 **开源版不做**——这些是部署侧 reliability 关注点，不属于"复现 SOTA 算法"开源版需求；`LLMClient` Protocol（[ADR 011](decisions/011-protocol-vs-abc.md)）天然支持装饰器扩展，部署方有需要时可自行实现
  - 算法层重试为何不加：业界算法层（DSPy 3 / LlamaIndex 3-10 / instructor 1）在 SDK 之上叠加 retry 是早期 SDK 不可靠的历史包袱，现代 SDK 内置重试已成熟，再叠会双层放大延迟；EverAlgo 取 LangChain Core "核心不重试 + 依赖 provider SDK" 路线

### 2.3 算子分层（草拟，未定稿）

按主轴 × 接口形态分（I/O 算子双接口 sync/async 共存，详见 [ADR 010](decisions/010-sync-async-dual-interface.md)）：

| 主轴 | 算子（物理路径） | 接口形态 | I/O 特性 |
|------|----------------|---------|----------|
| Extract | `parser.{image, audio, document, video, url}.{parse, aparse}` | **双接口** | I/O-bound（外部 OCR / ASR / 抓取） |
| Extract | `boundary.{chat, workspace, agent}.{detect, adetect}` | **双接口** | I/O-bound（调 LLM 检边界） |
| Extract | `user_memory.{episode, foresight, atomic_fact}.{extract, aextract}` | **双接口** | I/O-bound（独立 LLM 提取，与 cluster 无关）|
| Extract | `user_memory.profile.{extract, aextract}` | **双接口** | I/O-bound；接 `memcells: Sequence[MemCell]`（chronological，last = most recent；caller 按 cluster_id 反查后拼接传入）|
| Extract | `agent_memory.case.{extract, aextract}` | **双接口** | I/O-bound（独立 LLM 提取）|
| Extract | `agent_memory.skill.{extract, aextract}` | **双接口** | I/O-bound；**直接接 `cluster_id: str`**（同 cluster 下多个 case 聚合为 skill）|
| Extract | `knowledge.extractor.{extract, aextract}` | **双接口** | I/O-bound（独立 LLM 提取）|
| Rank facade | `rank.{episodic, case, skill}.{rank, arank}` | **双接口** | I/O-bound（含 LLM rerank） |
| Rank facade | `rank.profile.rank` | 同步 | 纯计算（cosine + threshold + 去重，无 fusion 无 rerank） |
| 算法工具 | `rank.fusion.*`（RRF / LR / cosine_to_lr_score / score_propagation） | 同步 | 纯计算（业务 facade 内部调用 + 算法同学独立可用）|
| 算法工具 | `rank.weight.*`（weighted_score / multi_field_weighting：业务字段加权抽象，case=`quality_score` / skill=`maturity + confidence`）| 同步 | 纯计算 |
| 算法工具 | `rank.rerank.{rerank, arerank}`（LLM rerank，prompt 由 caller 指定如 `episodic_rerank` / `case_rerank`）| **双接口** | I/O-bound（调 LLM）|
| 共享底层 | `boundary._tokenize.*`（count_tokens / force_split） | 同步 | 纯计算 |
| 共享底层 | `clustering.cluster_by_geometry`（cosine + 时间窗 + 阈值；user_memory 调）/ `clustering.cluster_by_llm`（embedding 召回 → fast path → LLM 决策；agent_memory 调）| **均 async** | 几何路径纯计算包 async（vector 由 caller 算后传入）/ LLM 路径 I/O-bound |

**双接口实现规范**：主用户 evermem 是 FastAPI 异步服务，**async 接口必须 native async**（不能 thread pool wrap）。每个 I/O 算子**手写 async 主实现**，sync 通过 `asgiref.sync.async_to_sync` 自动派生为桥接接口。**sync 接口仅限非 event loop 环境调用**（CLI 脚本 / 单元测试）；Jupyter / FastAPI / 任何 `async def` 上下文须用 `await aextract(...)`。详见 [ADR 010](decisions/010-sync-async-dual-interface.md)。

错误处理与 §2.2 同款：EverAlgo 自身不加重试层，依赖底层 SDK 默认 2 次（OpenAI / Anthropic SDK `DEFAULT_MAX_RETRIES = 2`）；跨 Provider fallback / multi-key 轮转 / 长时间退避 / 配额降级 **开源版不做**（详见 §2.5）。

### 2.4 聚类算子的处置

聚类作为独立工具性子包 `everalgo-clustering`，对外暴露 **双公开函数 + 值对象 `ClusterState`**——按 user_memory / agent_memory 包分层各调一个，物理隔离两类簇。

#### 公开 API

```python
# everalgo.clustering

# === 类型 ===
@dataclass(frozen=True)
class ClusterState:
    """累积的算法状态——online incremental K-means 必需的 3 项信息。"""
    centroids: dict[ClusterId, np.ndarray]   # 每簇中心向量；cosine 决策的对照向量
    counts: dict[ClusterId, int]              # 每簇事件数；centroid 增量公式 (C*n+v)/(n+1) 的 n
    last_ts: dict[ClusterId, float]           # 每簇最后更新时间（unix epoch seconds）；时间窗约束用

    @classmethod
    def empty(cls) -> "ClusterState": ...
    @classmethod
    def from_dict(cls, d: dict) -> "ClusterState": ...
    def to_dict(self) -> dict: ...
    def assign(
        self,
        cluster_id: ClusterId | None,         # None → 内部分配新 ID（max(existing_idx)+1 派生）；非 None → 归入该簇
        vector: np.ndarray,                   # 新事件向量（caller 提前算好）；用于 centroid 增量更新或新簇初始化
        timestamp: float,                     # 新事件时间（unix epoch）；更新 last_ts = max(prev, ts)
    ) -> tuple[ClusterId, "ClusterState"]:    # 返新值对象（frozen，不 mutate 原 state；事务安全）
        """新建 / 增量更新 centroid (C*n+v)/(n+1) / counts +=1 / last_ts = max。"""

@dataclass(frozen=True)
class ClusterConfig:
    """聚类阈值族（caller 启动期一次配好）。"""
    threshold: float = 0.65                   # 几何决策阈值；cosine ≥ 此值则归入候选 top-1（cluster_by_geometry 用 / cluster_by_llm 失败降级用）。算法可解释默认；生产值 caller 自调（业务不同 embedder + 数据分布最优阈值差异大）
    time_window_days: float = 7.0             # 时间窗（仅 cluster_by_geometry 用）；超过此 gap 的旧簇不参与召回。语义："久未活动的簇不应再吸收新事件"
    k_candidates: int = 30                    # 仅 cluster_by_llm 用；Top-K embedding 召回数，给 LLM 决策的候选范围
    llm_skip_threshold: float = 0.85          # 仅 cluster_by_llm fast path 用；top-1 cosine ≥ 此值则跳过 LLM 直接归入（省一次 LLM 调用）

@dataclass(frozen=True)
class Candidate:
    """候选簇（_find_candidates 输出 / _decide_by_* 输入）。"""
    cluster_id: ClusterId                     # 候选簇 ID
    similarity: float                         # 与新事件 vector 的 cosine 相似度

# === 公开算子（双函数）===

async def cluster_by_geometry(
    vector: np.ndarray,                       # caller 算好的 embedding（EverAlgo 不做 embed）
    timestamp: float,                         # 新事件时间（unix epoch seconds，caller 已 parse）
    state: ClusterState,                      # 当前累积状态（caller 持久化 + load）
    *,
    config: ClusterConfig,                    # 阈值族（threshold + time_window_days）
) -> tuple[ClusterId, ClusterState]:          # (分配的 cluster_id, 更新后的新 state 值对象)
    """纯几何聚类（cosine + 时间窗 + 阈值决策）。user_memory 编排器调，无 LLM。"""

async def cluster_by_llm(
    vector: np.ndarray,                       # caller 算好的 embedding（同上）
    timestamp: float,                         # 新事件时间（同上）
    query_text: str,                          # 新事件文本（拼 LLM prompt 用；典型为 agent_case.task_intent）
    state: ClusterState,                      # 当前累积状态（caller 持久化 + load）
    *,
    config: ClusterConfig,                    # 阈值族（k_candidates + llm_skip_threshold + threshold 兜底）
    llm: LLMClient,                           # LLM 客户端（caller 按 LLMScene 路由后注入；ADR 012）
    cluster_previews: dict[ClusterId, list[str]],  # caller 提前批量查的所有簇最近文本
                                                   #   key:   state.centroids 中所有 cluster_id
                                                   #   value: 该簇内最近 N 条 event 文本（N 由 caller 决定）
                                                   #   算子 Top-K 召回后从此 dict 取候选对应项
) -> tuple[ClusterId, ClusterState]:          # (分配的 cluster_id, 更新后的新 state 值对象)
    """LLM 精排聚类（embedding 召回 → fast path → LLM 决策，含失败降级回退几何）。agent_memory 编排器调。"""
```

`_find_candidates` / `_decide_by_threshold` / `_decide_by_llm` 是模块内部私有原子,不暴露——避免过度暴露 API 表面（未来定制需要再升公开）。

#### 双函数对应包分层

| 业务路径 | 编排器 | 调用 | 文本来源（上游产出）|
|---|---|---|---|
| user_memory（episode 簇）| `everalgo.user_memory` 编排 | `cluster_by_geometry` | `episode.text`（EpisodeExtractor 产物）|
| agent_memory（case 簇）| `everalgo.agent_memory` 编排 | `cluster_by_llm` | `agent_case.task_intent`（CaseExtractor 产物，缺位 fallback `episode.text`）|

每个产品包**持自己的 ClusterState**（互相物理隔离）；不再有"单一 state 同时装两类簇 + `case_cluster_ids` 标记"的现状模式。

#### 责任边界

| 谁 | 职责 |
|---|---|
| **caller**（user_memory / agent_memory 编排器）| ① 算 vector（EverAlgo 不做 embed）/ ② 持久化 `ClusterState`（介质自选 MongoDB / Redis / 文件）/ ③ 批量预取 `cluster_previews: dict[ClusterId, list[str]]`（caller 遍历 `state.centroids.keys()` 自家事件存储反查最近文本，作为 `cluster_by_llm` 入参传入；可缓存）/ ④ 注入 `LLMClient`（按 `LLMScene` 路由，[ADR 012](decisions/012-llm-stack-architecture.md)）/ ⑤ 重试和降级 / ⑥ 加分布式锁串行化 read-modify-write |
| **everalgo-clustering** | 算法本质：候选检索（cosine + 时间窗）+ 几何决策（threshold）+ LLM 精排（含 prompt 内置 + 解析 + 失败降级回退几何）+ centroid 增量更新公式 + state 演化 |

#### 算子内部流程

`cluster_by_geometry`（user_memory）3 步：

```
candidates = _find_candidates(state, vector, k=1, time_window=cfg.time_window_days*86400)
cid_or_none = _decide_by_threshold(candidates, cfg.threshold)
return state.assign(cid_or_none, vector, timestamp)
```

`cluster_by_llm`（agent_memory）5 步：

```
candidates = _find_candidates(state, vector, k=cfg.k_candidates, time_window=None)
# fast path
if not candidates:
    return state.assign(None, vector, ts)
if candidates[0].similarity >= cfg.llm_skip_threshold:
    return state.assign(candidates[0].cluster_id, vector, ts)
# LLM 决策（不含向量入参——LLM 是文本模型）
# 从 caller 传入的全量 cluster_previews 中取候选对应项（无 callback 反向调用）
previews = {c.cluster_id: cluster_previews.get(c.cluster_id, []) for c in candidates}
cid_or_none = await _decide_by_llm(candidates, query_text, previews,
                                    llm=llm, threshold=cfg.threshold)
return state.assign(cid_or_none, vector, ts)
```

`_decide_by_llm` 给 LLM 看的是 **text + 标量**（query_text + cluster_id + similarity 标量 + recent_texts），不传向量——LLM 是文本模型，raw float 数组对它无意义。LLM 失败 → 降级到 `_decide_by_threshold(candidates, threshold)`（这是算法 IP，不是基础设施 retry）。

#### `cluster_id` 的下游消费

聚类不"合并"任何业务对象——只产出**分组依据 `cluster_id`**。下游消费分两类：

| 算子 | 消费方式 | 算子签名是否含 `cluster_id` |
|------|---------|---------------------------|
| `AgentSkillExtractor`（agent_memory）| 同 cluster 下的多个 `AgentCase` 聚合为一个 `AgentSkill` | ✅ 直接接 `cluster_id: str` |
| `ProfileExtractor`（user_memory）| 把同 cluster 的历史 episode 列表作为上下文 | ❌ 不接 `cluster_id`，接 `cluster_episodes: list[MemCell]`（caller 按 `cluster_id` 反查 + fetch 后传入）|

不依赖 `cluster_id` 的算子（独立路径，单条 MemCell 提取）：

- `AtomicFactExtractor` — 单 MemCell → list[AtomicFact]
- `EpisodeExtractor` — 单 MemCell → list[Episode]
- `ForesightExtractor` — 单 MemCell → list[Foresight]

#### 编排顺序（caller 负责）

按 user_memory / agent_memory 包分层各自编排，分两支并行（两支用各自独立的 ClusterState，物理隔离；每支各自调对应的双函数之一）：

**Phase 0：上游 Extractor 产出派生记忆**

```
episode    = await EpisodeExtractor.aextract(memcell)    # 永远跑（任何 MemCell 类型）
agent_case = await CaseExtractor.aextract(memcell)       # 仅 agent_conversation 类型
```

**Phase 1：cluster（user_memory 支）—— 调 `cluster_by_geometry`**

```python
text   = episode.text
vector = await caller.embedder.aembed(text)
async with caller.lock(f"trigger_clustering:{user_id}"):
    state = ClusterState.from_dict(state_store.load(user_id)) or ClusterState.empty()
    cluster_id, new_state = await cluster_by_geometry(
        vector, timestamp, state,
        config=cfg,
    )
    await state_store.save(user_id, new_state.to_dict())
```

**Phase 1':cluster（agent_memory 支）—— 调 `cluster_by_llm`**

```python
text   = agent_case.task_intent or episode.text
vector = await caller.embedder.aembed(text)
async with caller.lock(f"trigger_clustering:{agent_id}"):
    state = ClusterState.from_dict(state_store.load(agent_id)) or ClusterState.empty()
    # 批量预取所有簇最近文本（agent_memory 自家 case_repo 反查，可缓存）
    cluster_previews = await case_repo.fetch_recent_intents_by_clusters(
        list(state.centroids.keys()), max_per_cluster=5,
    )
    cluster_id, new_state = await cluster_by_llm(
        vector, timestamp, query_text=text, state,
        config=cfg, llm=llm,
        cluster_previews=cluster_previews,
    )
    await state_store.save(agent_id, new_state.to_dict())
```

**Phase 2：派生记忆生成**

```
独立路径并发（不依赖 cluster_id）：
    - AtomicFactExtractor.aextract(memcell)
    - ForesightExtractor.aextract(memcell)

依赖 cluster_id：
    - user_memory:  ProfileExtractor.aextract([*cluster_memcells, memcell], sender_id=..., ...)
                    （caller 按 cluster_id 反查 cluster 内 MemCells 后拼入，chronological，last = current）
    - agent_memory: AgentSkillExtractor.aextract(case, cluster_id=...)
                    （直接接 cluster_id，独立锁 trigger_agent_skill:{agent_id}:{cluster_id}）
```

**关于 v0.34 `IncrementalClusterer` 单类双入口**：v0.35 已拆为 `cluster_by_geometry` / `cluster_by_llm` 双公开函数（详见上方"公开 API"段 + 自检 "Why 拆双函数"）；编排顺序里不再出现 `IncrementalClusterer` 类名。

> ✅ **设计自检**
> - **Why 拆双函数 `cluster_by_geometry` / `cluster_by_llm`** 而非单函数内部 if/else 分发：包分层后 user_memory 永远走几何、agent_memory 永远走 LLM——拆双函数对齐包分层 + 类型签名直接表达必填参数（不再 7 个 Optional 中 50% 仅 LLM 路径用）+ IDE/mypy 友好；scikit-learn `cluster.{KMeans, DBSCAN, OPTICS}` 同模式（按算法变体平级独立 API，不是单类内部 flag 分发）
> - **Why state 在 caller 持久化、core 写演化逻辑**：函数式状态管理标准模式（Redux / Elm / Python `frozenset.union` 同理）。EverAlgo 定义"state 是什么"（类型 + 字段 + 序列化）+ "state 怎么演化"（`assign` 方法），evermem 决定"state 何时载入 / 存哪里 / 谁加锁"。值对象 in/out 比 sklearn `kmeans.fit(X)` 后 mutate 更事务安全（算法异常不污染原 state）
> - **Why `ClusterState` 仅 3 字段**（centroids/counts/last_ts）：online incremental K-means 算法本质 3 项累积信息——几何对照 + 增量公式参数 + 时间窗约束。现状 `MemSceneState` 10 字段中 7 个是历史包袱（`event_ids` / `timestamps` / `vectors` / `cluster_ids` / `eventid_to_cluster` / `case_cluster_ids` / `next_cluster_idx` 全删）
> - **Why EverAlgo 不算 embed**：与 LLM 不同——LLM 含 prompt 是算法 IP 载体（决策手段）；embed 无 IP 载体（model 选择属业务决策，调用是 SDK 适配）。对齐 sklearn / FAISS / DSPy 模式（caller 算 vector 后传入）；与 LangChain / LlamaIndex 内置 embed 不对齐（他们是端到端框架定位，与 EverAlgo 算法库定位不同）。详见 §1.2
> - **Why `cluster_previews` 直接传 dict 而非 callback**：v0.42 修正——早期 callback `fetch_previews` 论据基于「caller 不能提前预取所有簇文本（O(N) 浪费）」假设；但 EverAlgo 实际场景单 owner_id 簇数 N ≤ 50（按时间窗 + 业务隔离），`k_candidates = 30`，IO 比仅 ~1.6 倍，不构成「严重浪费」。直接传 `dict[ClusterId, list[str]]` 优势：① 纯数据传入，无反向依赖（callback 让算法库调 caller 代码是 code smell）；② 函数签名干净（dict 比嵌套 Callable 直观）；③ caller 端可缓存同一 owner 短时间内的 previews。对齐 sklearn `KMeans.fit(X)` / FAISS `index.add(vectors)` / BIRCH 等算法库阵营——caller 提供完整数据，算法库做几何，不通过 callback 增量获取
> - **Why LLM prompt 内置 everalgo-clustering**：prompt 是算法 IP（评估/调优 prompt = 调算法），属于算法不可分割部分（与 LLMClient SDK 解耦层不同）；caller 可选 override 但默认值由算法提供
> - **Why LLM 失败降级保留在算子内**：降级策略（"LLM 失败 → 几何 top-1 + threshold"）是 cluster 算法对 LLM 失败的处置，属算法 IP；与"基础设施 retry"不同（后者由 caller 处理，对齐 ADR 012 算法层不加 retry）
> - **Why `cluster_id` 不进 `ProfileExtractor` 算子签名**：Profile 算子需要的是"一批历史 episode 上下文"，"按 `cluster_id` 取" 是编排层的实现选择（按时间窗口取 / 按 user_id 取 / 按 cluster 取都可），不是算子的需求；算子签名干净接 `memcells: Sequence[MemCell]` 即可（chronological，last = most recent）
> - **Why 聚类独立子包**：`AgentSkill` 直接消费 + `Profile` 间接消费 + 算法策略多次迭代（centroid 公式 / fast path 阈值 / prompt）独立 SemVer，独立子包合理（[ADR 006](decisions/006-clustering-independent-subpackage.md)）

#### 与现状代码的关键差异

| 现状（`evermem/src/memory_layer/cluster_manager/manager.py`）| 新设计 | 删除原因 |
|---|---|---|
| 单入口 `cluster_memcell(memcell, state, has_case)`（manager.py:263-289）| 双公开函数 | `has_case` 由包分层表达 |
| 算子内 `_get_embedding`（manager.py:307）调 module-global vectorize_service | caller 算 vector 传入 | EverAlgo 不算 embed |
| `MemSceneState.case_cluster_ids: set` | 删字段 | 包分层后单 state 单语义 |
| `MemSceneState.eventid_to_cluster: dict` | 删字段 | caller 自家存储已有 |
| `MemSceneState.next_cluster_idx: int` | 删字段 | 用 `max(existing_idx)+1` 派生 |
| `MemSceneState.event_ids` / `timestamps` / `vectors` / `cluster_ids: List` | 删字段 | 算法路径不读，现状死字段 |
| `_call_llm_for_clustering` 内部 `for attempt in range(3)` retry（manager.py:656）| 单次失败即降级 | 对齐 ADR 012 |
| `_callbacks` / `on_cluster_assigned` callback 机制 | 删 | 现状死代码 |
| `_stats` 字段 | 删 | observability 归 evermem |
| state 原地 mutate | 值对象 in/out | 事务安全 |

→ 净化后核心算法 `cluster_by_geometry` 3 行 + `cluster_by_llm` 10 行，vs 现状 `_cluster_memcell_embedding` + `_cluster_memcell_llm` 共 ~185 行。

### 2.5 LLM 配置与注入

EverAlgo **不持有 scene 概念**——业务编排（哪个算子用哪个模型）归 evermem。EverAlgo `everalgo.llm` 子包仅承担：① **LLMClient Protocol 接口**；② **Provider 路由**（`LLMConfig.provider` → SDK 适配实现）；③ **全局注入 + scoped 替换**（DSPy `dspy.settings.lm` + `dspy.context(lm=...)` 同款）。

#### 子包结构

```
everalgo/llm/
├── __init__.py        # facade: chat / stream / use / configure + re-export types/errors
├── types.py           # ChatMessage / ChatResponse / Usage / ToolCall / Chunk
├── client.py          # LLMClient Protocol (@runtime_checkable) + LLMConfig dataclass
├── errors.py          # LLMError + 7 子类（混合多重继承让原生 SDK catch 仍有效）
├── routing.py         # Provider 路由：build_client(config) → LLMClient
└── providers/
    ├── openai_compat.py   # openai.AsyncOpenAI（含 OpenRouter / vLLM / DeepSeek / Azure 兼容）
    ├── anthropic.py       # anthropic.AsyncAnthropic
    └── bedrock.py         # boto3 AsyncBedrockRuntime
```

#### 启动期：单一全局 LLMClient（DSPy 同款）

```python
from everalgo.llm import LLMConfig

everalgo.configure(
    llm=LLMConfig(                                # Provider 路由层按 config.provider 构造
        provider="openai_compat",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_KEY"],
        model="openai/gpt-4.1-mini",
    ),
)

# 或直接传已构造的 LLMClient 实例（绕过 Provider 路由）
everalgo.configure(llm=my_custom_client)
```

env 注入（12-factor / k8s 友好）：

```
EVERALGO_LLM_PROVIDER=openai_compat
EVERALGO_LLM_BASE_URL=https://openrouter.ai/api/v1
EVERALGO_LLM_API_KEY=sk-...
EVERALGO_LLM_MODEL=openai/gpt-4.1-mini
```

#### 算法同学的日常调用（3 层注入解析）

EverAlgo 算子方法签名加可选 `llm: LLMClient | None = None` 参数（DSPy `predictor(..., lm=...)` 同款），内部用 `everalgo.llm.resolve(llm)` 单行封装 3 层 fallback + 未注入抛 `LLMNotConfiguredError`，避免每个用 LLM 的算子重复写 boilerplate：

```python
class EpisodeExtractor:
    async def aextract(
        self, memcell, *, llm: LLMClient | None = None,
    ) -> list[Episode]:
        client = everalgo.llm.resolve(llm)    # 1 行：3 层 fallback + 兜底抛异常
        return await client.chat(messages, ...)
```

辅助函数 `resolve()` 实现（仅算法库内一处定义，所有算子共用）：

```python
# everalgo/llm/__init__.py
def resolve(per_call: LLMClient | None = None) -> LLMClient:
    """3 层注入解析：per-call > scoped (ContextVar) > default (global)。
    任一层非 None 即返回；全 None 时抛 LLMNotConfiguredError。"""
    if per_call is not None:
        return per_call
    client = current()    # 查 ContextVar (scoped) → 查全局 default
    if client is None:
        raise LLMNotConfiguredError(
            "No LLM configured. Use everalgo.configure(llm=...), "
            "pass llm= per-call, or wrap in everalgo.llm.use(...)."
        )
    return client
```

#### 3 层注入路径（DSPy 同款）

| 层 | 用法 | 优先级 | 典型场景 |
|----|------|--------|----------|
| **per-call** | `aextract(..., llm=client)` 直接传参 | 最高 | evermem 单调用按 scene 注入（主路径）|
| **scoped** | `async with everalgo.llm.use(client):` contextmanager | 次高 | evermem pipeline 段批量同 client（减少重复传参）|
| **default** | 启动期 `everalgo.configure(llm=default)` | 兜底 | dev / 测试 / Jupyter / 简单脚本（一次走天下）|

实现机制：`everalgo.llm.use(client)` 用 `contextvars.ContextVar` set / reset（threading.local + async safe，DSPy `dspy.context` 同款）。

#### evermem 端做 scene 路由（业务编排层职责）

EverAlgo 完全无 scene 概念——evermem 持有 scene → client 映射，3 层注入按场景选用：

```python
# evermem 端 SceneRouter（业务编排层）
class SceneRouter:
    def __init__(self):
        self._clients = {
            "episode":  OpenAICompatClient(model="anthropic/claude-sonnet-4-6", ...),
            "profile":  OpenAICompatClient(model="openai/gpt-4.1-mini",         ...),
            "boundary": OpenAICompatClient(model="qwen/qwen3-235b",             ...),
            "rerank":   OpenAICompatClient(model="openai/gpt-4o-mini",          ...),
        }

    def get(self, scene: str) -> LLMClient:
        return self._clients[scene]


# evermem pipeline 调用 EverAlgo 算子（每场景按需选 per-call / scoped）

# 单调用按 scene → 用 per-call 简洁
async def aextract_episodes(memcell):
    return await EpisodeExtractor().aextract(memcell, llm=scene_router.get("episode"))


# 多调用同 client → 用 scoped wrap 减少重复传参
async def arank_all(rank_input):
    async with everalgo.llm.use(scene_router.get("rerank")):
        return await asyncio.gather(
            rank.episodic.arank(rank_input),
            rank.case.arank(rank_input),
            rank.skill.arank(rank_input),
        )
```

#### 内部路由伪代码（仅 Provider 路由，无 Scene 路由）

```python
# everalgo/llm/routing.py
def build_client(config: LLMConfig) -> LLMClient:
    """Provider 路由：config.provider → LLMClient 实现"""
    match config.provider:
        case "openai_compat":
            from .providers.openai_compat import build
            return build(config)
        case "anthropic":
            from .providers.anthropic import build
            return build(config)
        case "bedrock":
            from .providers.bedrock import build
            return build(config)
        case _:
            raise ValueError(f"Unknown provider: {config.provider}")
```

#### 高可用编排不属于开源版需求

EverAlgo 设计原则：保证"接上就能用"——单 client 调用 + SDK 默认 2 次重试兜底。**以下均不属于开源版需求**：
- multi-key 轮转
- 跨 provider fallback
- 长时间退避 / 冷却
- 配额降级
- 租户级路由
- 多 model load balance

这些是部署侧 reliability 关注点，**不影响算法本身的 SOTA 复现**。EverAlgo 调用失败直接抛 `LLMError`（7 子类含语义 + 混合多重继承让原生 SDK catch 仍有效），由 caller 决定如何处理（dev / 测试场景常见做法是直接传播）。

`LLMClient` Protocol（[ADR 011](decisions/011-protocol-vs-abc.md)）天然支持装饰器扩展——**部署方有需要时可自行实现**透明叠加，EverAlgo 设计文档不预设具体实施方式：

```python
# EverAlgo 默认调用形态（接上就能用）
client = openai_compat.build(LLMConfig(api_key=KEY, model="gpt-4.1-mini"))
result = await client.chat(messages, model="gpt-4.1-mini")

# 部署方按需扩展（示意，具体实现自定）
class MyFallbackClient:                            # 实现 LLMClient Protocol 即可
    def __init__(self, primary, fallbacks): ...
    async def chat(self, messages, *, model, **kw): ...
```

上层算子调用形态完全一致（`await client.chat(...)`），扩展对算法库透明。


> ✅ **设计自检**
> - **Why scene 路由不在 EverAlgo**：scene（episode / profile / boundary / rerank 等）是业务编排概念——"哪个算法步骤用哪个模型"是部署 / 业务 / 成本决策，**与算法本身无关**。**业界主流 LLM 库 4/4 不做 scene 路由（横跨 3 阵营无差别）**：算法库阵营 DSPy `dspy.settings.lm` + `dspy.context(lm=...)` / LlamaIndex `Settings.llm` + chain 框架阵营 LangChain（`prompt | model | parser` 组合时传 model）+ agent 框架阵营 AutoGen（用户构造 client 传 agent）。"算法库 / chain 框架 / agent 框架" 跨 3 阵营都不做 → 反向印证 scene 路由不属任何阵营 LLM 库的职责，归业务编排层。evermem 现状的 `LLMScene` enum 是把业务逻辑掺进了算法库，重构 EverAlgo 时剥离到 evermem 端 SceneRouter
> - **Why 3 层注入（per-call > scoped > default）**：DSPy `dspy.settings.configure` + `dspy.context` + `predictor(..., lm=...)` 同款 3 层模式实证——per-call 给 evermem 单调用 scene 注入（主路径）/ scoped 给 evermem pipeline 段批量切换（减少重复传参）/ default 给 dev / 测试 / Jupyter / 简单脚本兜底（生产 evermem 可不调 configure）。优先级 per-call > scoped > default 无歧义。EverAlgo 算子方法签名补可选 `llm: LLMClient | None = None` 参数 + 内部用 `everalgo.llm.resolve(llm)` 单行封装 3 层 fallback + 未注入抛 `LLMNotConfiguredError`（算法库内一处定义，所有算子共用，避免每个算子重复 7 行 boilerplate）
> - **Why Provider 路由仍在 EverAlgo**：Provider 路由（`LLMConfig.provider` → SDK 适配）是**实现层**职责（SDK 适配代码紧耦合算法库），与 Scene 路由的**业务层**职责不同；Letta `LLMClient.create` `match-case` 同款实现
> - **Why 算法层不加重试**：业界 OpenAI / Anthropic SDK 内置 `DEFAULT_MAX_RETRIES = 2`（hardcoded `_constants.py`，覆盖 connection / 408 / 409 / 429 / 5xx）已成熟兜底；DSPy 3 / LlamaIndex 3-10 / instructor 1 在 SDK 之上叠加 retry 是早期 SDK 不可靠的历史包袱，现代叠加只放大延迟，EverAlgo 取 LangChain Core 路线（核心不重试 + 依赖 provider SDK）
> - **行业依据**：DSPy `dspy.settings.lm` + `dspy.context` 全局 + scoped 模式；Letta `LLMClient.create` `match-case` 实现 Provider 路由；LangChain Core `BaseChatModel` 不内置 retry / `ChatOpenAI` 透传 `max_retries` 给 OpenAI SDK 的分层哲学
>
> **完整决策与 6 轮调研事实见 [ADR 012 LLM 抽象层架构](decisions/012-llm-stack-architecture.md)**（含 P1/P2/P3 三派 13 项目矩阵 + LiteLLM 工业级真实定位 + B 派 ABC 4 真实驱动 + 5 派 LLMError 错误层级分布 + 4/4 主流 LLM 库横跨 3 阵营不做 scene 路由实证 + evermem 现状代码迁移清单）

---

## 3. 待讨论清单（下一步逐条推进）

| # | 议题 | 备注 |
|---|------|------|
| T1 | 数据契约：`Message / RawData / MemCell / Episode / Foresight / AtomicFact / Profile / AgentCase / AgentSkill / ParsedContent / KnowledgeMemory` 的精确 schema | Pydantic v2 建议作为基类，需确认 |
| T2 | 算子完整清单（补齐 20 个）+ 每一个函数签名 | 需 BOSS 按子领域逐个对齐 |
| T3 | `everalgo.configure(...)` 的完整参数表 + env 映射规则 | 仅 LLM 相关（v0.39 起 prompt 不再走 configure，详见 T5）|
| T4 | `everalgo.llm` 模块的具体 API（`chat` / `stream` / `use()` contextmanager / `configure(llm=...)` / `LLMClient` Protocol / `LLMError` 7 子类 / Provider 路由 `build_client(config)`）| 注：scene 路由出 EverAlgo 归 evermem，详见 §2.5 |
| ~~T5~~ | ✅ **RESOLVED 2026-05-06**（v0.39）：Prompt 是 Python 字符串模块（`prompts/{en,zh}/<name>.py` module-level 常量），与算法库阵营 DSPy / LlamaIndex / instructor / mem0 / memsys_opensource 现状 5/5 一致。evermem 自定义路径：算子 per-call `prompt=` 参数（细粒度主路径）/ caller monkey-patch 模块常量（启动期粗粒度全局）。validator 仍保留（占位符 / 长度校验，`everalgo-core/prompts/validator.py`）。详见 §1.4 prompt 实现段 + §1.2 line 66 prompts 子包描述 |
| T6 | 测试/替换的惯例：monkeypatch / respx / fake 客户端 /`everalgo.testing` 模块是否提供 | |
| T7 | AgentSkill 是否废除聚类（来自原设计文档的决策，但有风险，见附录 A-1） | 需算法团队确认 |
| T8 | Workspace pipeline 是否硬编码"仅 Episode"，还是由 evermem 自由编排（见附录 A-2） | |
| T9 | Knowledge 的"全局共享、不含用户身份"在权限模型上的边界（见附录 A-4） | 需产品+安全确认 |
| T10 | 子领域间的依赖方向规则（能否从 `user_memory` 直接 import `parser`？共享类型如何放置） | |
| ~~T11~~ | ✅ **RESOLVED 2026-04-23**（v0.35 二次精化）：聚类作为独立工具性子包 `everalgo-clustering`（与 boundary / rank / parser 平级）；v0.35 接口形态定型为双公开函数 `cluster_by_geometry`（user_memory 调）+ `cluster_by_llm`（agent_memory 调）+ 值对象 `ClusterState`（详见 §2.4）。早期"`centroid` + `llm_direct` 两策略并列"表述作废 |
| T12 | Rank 层数据契约：`RankInput` schema（`sparse_candidates[] + dense_candidates[]` + 预取关联 ep→fact 等，与具体存储引擎解耦）、`RankOutput` schema、4 Ranker 各自的 payload 差异 | 与 evermem Recall 侧对齐 |
| T13 | Rank prompts 组织：4 Ranker 的 rerank prompt 放 `rank/prompts/rerank_{episodic,profile,case,skill}.md`？是否共享一个 prompt 还是拆细？ | 注：scene 概念出 EverAlgo 后，prompt 选择是算子内部决策（与 LLM 模型选择解耦） |
| T14 | `rank.{fusion, weight, rerank}` API 完整签名：fusion `rrf(es, milvus, k=60)` / `lr(es, milvus, weights)` / `cosine_to_lr_score(sim, alpha)` / `score_propagation(parents, children, alpha)`；weight `weighted_score(items, fields={"quality_score": 0.5})` / `multi_field_weighting(items, weights={"maturity": 0.6, "confidence": 0.4})`；rerank `rerank/arerank(items, prompt, top_k)` 参数细节 | |
| T15 | ep→fact 关联的传输 schema：evermem Recall 侧预取多少 fact per episode？以什么结构传入 Ranker？何时展开（recall 阶段 vs rank 阶段）？ | 跨 evermem / EverAlgo 接口协议，需双方对齐 |

---

## 附录 A：设计疑点（源自对原始 Confluence 设计的挑战）

以下 7 条源于对原设计文档的深入审视，是**待讨论清单**中多条议题的底层原因。

**A-1. AgentSkill 取消聚类的 skill drift 风险**
原文档称"LLM 的语义判断本身已包含模式识别逻辑，不需要额外聚类"。但：(a) LLM 同输入不同轮次可能输出不同判断，产生 skill drift；(b) 当 skill 库增长到千级，每次 prompt 要塞完整 skill 列表，context 成本超线性增长，反而比聚类开销更高。

**A-2. Workspace pipeline "仅 Episode" 的论据不充分**
Email / Calendar 天然有 deadline / commitment（Foresight 适用），Confluence 页面密集 AtomicFact（项目规范、接口契约）。真正的边界应是"数据是否有连续会话语义"，而非源类型。

**A-3. "EverAlgo 无状态" 与 "分布式锁保证顺序"自相矛盾**
分布式锁必依赖 Redis 等持久化组件。本设计已解耦：**锁的实现与调度完全由 evermem 持有**，EverAlgo 只暴露纯计算算子。

**A-4. Knowledge Library "全局共享、不含用户身份"的权限模型缺失**
用户上传的合同 / 录音有高度隐私性，"全局共享"的边界不清（跨用户单租户内？跨租户？）。若支持私有 Knowledge，"不含用户身份"的前提就破。待产品明确。

**A-5. 4 library 若独立 distribution 的版本地狱**
已在 §1.3 选择单 distribution + extras 规避。保留选项：未来若 Parser 需要与其他产品线独立演进，再分家为独立 distribution。

**A-6. MVP 记忆类型命名与现状断层**
原设计文档提"5 种 MVP：memcell / episodic / profile / proactivity / consolidation"。当前 evermem `MemoryType` enum 是 8 值（PROFILE / EPISODIC_MEMORY / FORESIGHT / ATOMIC_FACT / RAW_MESSAGE / AGENT_MEMORY / AGENT_CASE / AGENT_SKILL），无 `proactivity` / `consolidation`。命名需统一，proactivity ≈ Foresight 是明显的，consolidation 对应关系未明。

**A-7. PromptSlot 沙盒的具体实现形态**
原文档提"变量检查 → 沙盒试运行 → Schema 校验"三步，但"沙盒"具体是 shadow mode（LLM 实跑并比对）、dry run（不调 LLM 只做 schema）、还是产品化平台做的外部校验？本设计倾向**不在 EverAlgo 内置 LLM 沙盒**（会破坏无状态），只做 schema-level 校验 + 变量引用检查。

**A-8. ep→fact 关联关系的传递协议（已部分 resolved）**
「检索 EverAlgo」文档说 EverAlgo 无 DB 操作，但 `EpisodicRanker` 有 Hierarchical Expansion (ep→fact)。**BOSS 已确认（2026-04-23）：evermem Recall 阶段预取 ep→fact 关联关系传入 Ranker，Rank 层在内存里做 hierarchy 展开，无任何 DB 调用。** 但传输 schema（每个 episode 带多少 fact？数据结构形态？）待 T15 定稿。

**A-9. MongoDB backfill 在 md-based 架构下的对应**
「检索 EverAlgo」文档表格写 evermem Recall 含 "MongoDB backfill"，但新 evermem 设计（page 2157772920）去 Mongo 用 md 为 SoT。"backfill" 在新架构下应理解为"按 `md_path + entry_id` 从 md 文件回读 entry 原文"，两文档未同步修订，需对齐。

**A-10. AtomicFact Pre-fetch 的触发范围**
「检索 EverAlgo」文档表格说 evermem Recall 含 "AtomicFact Pre-fetch"。不清楚：仅 EpisodicRanker 的 ep→fact 展开需要（按 episode 预取关联 fact）？还是所有涉及 fact 的 Ranker 都需要？预取时机（与 ES/Milvus 召回并行 vs 串行）文档未说。

---

## 版本记录

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-04-22 | 初稿，沉淀定位 / 发布策略 / 代码风格 / 上游使用形态 |
| v0.2 | 2026-04-22 | 子领域拆分为 4 产出型 + 3 工具型（boundary/clustering/retrieval 抽出为公用子领域）；调用示例 import 路径同步更新；新增 T11 Clusterer 去留 |
| v0.3 | 2026-04-22 | §1.2 引入 facade + providers 分层（参考 litellm / SQLAlchemy / Django）；`llm` / `embed` 作为顶层 facade，`providers/{llm,embed}/<provider>/` 装具体实现；§2.1 configure 示例切到 `llm={default, scenes, fallback_to_default}` 新格式；新增 §2.5「LLM Scene Routing 机制」完整说明 |
| v0.4 | 2026-04-23 | 基于 evermem 设计（page 2157772920）+ 检索 EverAlgo（page 2139194126）重构：§1.1 加 Extract + Rank 双主轴；§1.2 子包划分整段重写 —— 产品性子包按 evermem 调用面 1:1 组织（5 产品 + 4 基础 + testing），删 `boundary/` / `clustering/` / `retrieval/` / `embed/` / `providers/embed/`，加 `rank/`（4 Ranker + fusion.py）和顶层扁平 `tokenize.py`；§1.2 命名原则补 scikit-learn / Django 行业实证（WebFetch 2026-04-22）；§2.1 调用示例对齐新子包；§2.3 算子分层改按双主轴维度；§2.4 聚类算子降级为 extractor 内部细节；§2.5 场景清单对齐新子包；新增 T12-T15 数据契约/rank 细节议题；新增附录 A-8/A-9/A-10 源自「检索 EverAlgo」的新疑点（A-8 ep→fact 预取协议由 BOSS 决 "evermem Recall 预取传入"） |
| v0.5 | 2026-04-23 | **Revert §1.2 的 "物理按调用面 1:1" 为 "物理按算法职责 + 外部 re-export 对齐契约"**（行业算法库主流模式：scikit-learn / transformers / litellm / django 均如此）。恢复 `boundary/` 独立工具性子包（3 MemCellExtractor + 共享 `_tokenize` / `_force_split` / LLM 边界 prompt），删顶层 `tokenize.py`。evermem 设计文档的契约名（v0.7 矫正前为 `everalgo.UserMemory.ConvMemCellExtractor`，矫正后为 `everalgo.user_memory.ConvMemCellExtractor`）通过 `user_memory/__init__.py` / `agent_memory/__init__.py` re-export 对齐。产品性子包变 4（parser / user_memory / agent_memory / knowledge），工具性子包 2（boundary / rank，clustering 待 T7/T11 决议）。§2.1 示例展示两条访问路径（外部契约路径 + 算法同学物理路径），§2.3 算子表回归 `boundary.*`，§3 T11 重新聚焦 clustering 物理形态。动因：v0.4 的机械对齐牺牲了算法同学迭代效率（改边界要动 3 文件 + 共享底层外露），与 §1 "算法同学大本营" 定位冲突。 |
| v0.6 | 2026-04-23 | **§1.3 整段重写：单 distribution + extras → 7 独立 distribution + namespace package**（PEP 420，参照 llama-index 模式）。BOSS 决议——集成方需"升级 A 不动 B"。拆分：`everalgo-core` + `everalgo-{boundary, rank, parser, user-memory, agent-memory, knowledge}`，命名全 dash，import 共享 `everalgo.*`，testing 并入 core（参照 numpy.testing / pandas.testing / torch.testing 调研结论），不加 meta package。**§1.2 加正文段"为什么 re-export 而不是 Client 类 facade"**——Python 生态算法库 vs 网络 SDK 二分泾渭分明，大公司原厂 SDK 100% 用 Client 类（OpenAI/Anthropic/Google），大公司算法库 100% 用 re-export（Meta PyTorch、HuggingFace transformers、scikit-learn），EverAlgo 按 §1.1 定位明确属算法库阵营，与该阵营所有库一致选 re-export（WebFetch 2026-04-23 核验 7 仓库）。§1.2 目录速览以 distribution 框分组标注归属。 |
| v0.7 | 2026-04-23 | **PEP 8 命名矫正**：evermem 设计文档原文使用的 `everalgo.UserMemory.ConvMemCellExtractor` / `everalgo.AgentMemory.*` / `everalgo.Parser` / `everalgo.Knowledge.*` / `everalgo.Rank.*` 是 PascalCase namespace（Java/.NET 风格），违反 PEP 8 模块命名（应 lowercase）。本文档矫正为 `everalgo.user_memory.ConvMemCellExtractor` / `everalgo.agent_memory.*` / `everalgo.parser` / `everalgo.knowledge.*` / `everalgo.rank.*`，与 transformers / pytorch / scikit-learn / llama-index / openai-sdk / dspy 等所有主流 Python AI 库 100% 一致（"lowercase namespace + PascalCase class"）。evermem 文档作者侧待同步修订。`testing/` 内容矫正为「assertions + fake_llm 两件套」（对标 numpy.testing / torch.testing 纯断言风格），删除 `fixtures` 子目录（pytest 文化术语，主流算法库 testing 子包均无；预设样本未来若提供独立 `everalgo.examples` 子包，按 sklearn.datasets 模式）。§1.1 双主轴表精简：Extract 输入收敛为 `MemCell` 等结构化输入单元，Rank 输入与具体存储引擎解耦（`sparse_candidates[] + dense_candidates[]` + 预取关联）。 |
| v0.8 | 2026-04-23 | **`providers/` 收缩进 `llm/` 内部**：v0.3-v0.7 用 `everalgo/llm/` + 顶层平级 `everalgo/providers/llm/<provider>/` 的拆法，v0.5 阶段已核查发现错（明星 LLM 库 litellm / instructor / dspy / llama-index 全部用"LLM 抽象 + providers 收在一个根子包内"模式），但 v0.6 转 distribution 拆分时未回收。本次矫正为 `everalgo/llm/providers/<provider>/`，与 instructor 模式对齐；命名 `providers` 避开 litellm 的 `llms/` 在 EverAlgo 上重复（顶层已是 `llm/`，子目录 `llms` 套娃尴尬）。§1.2 目录速览框、§1.2 自检"Why providers 内嵌"那条、§1.3 everalgo-core 内容描述同步更新；顶层不再有 `providers/` 子包。§1.2 末尾统计句矫正为 "10 个 subpackage / 7 个 PyPI distribution"（之前误写 "10 个 / 单一 distribution" 与 "9 个 / 7 个" 两处过时）。基础设施段从"4 个子包"改为"3 个"（`llm/providers` 不是独立子包，并入 `llm` 行）。 |
| v0.9 | 2026-04-23 | **§1.3 加版本兼容策略 + 仓库管理两个新小节**（BOSS 指令"只参考 HuggingFace"）：① 版本兼容——HuggingFace 全家桶实证（transformers `huggingface-hub>=1.5.0,<2.0` / datasets `>=0.25.0,<2.0` / accelerate `>=0.21.0` 无 upper），EverAlgo 采纳"兄弟包不互依赖 + 下游对 base 包宽松约束 + core 严守 SemVer + 不强制同步 release" 4 条策略；diamond dependency 自然消解。② 仓库管理——multi-repo（**7 个独立 GitLab 仓**，模式参照 HuggingFace 4 个 GitHub 仓；平台不同模式相同），不选 monorepo（langchain `libs/` / llama-index 顶层）。修正 §1.3 pyproject 草案 `everalgo-core>=0.1,<0.2` → `>=0.1.0,<2.0.0`（仿 HuggingFace transformers 对 hub）。自检"行业依据"按维度重组：版本管理 + 仓库管理 → HuggingFace；namespace package → llama-index；langchain / haystack / HuggingFace A 风格 namespace 不参照原因明示。 |
| v0.10 | 2026-04-23 | **§1.3 顶部加「行业参照矩阵」表**，作为整章总览。回应读者认知挑战："为什么命名参照 llama-index、仓库参照 HuggingFace 不一致？" —— 矩阵把 5 个维度（命名前缀 / import namespace / 仓库形态 / 版本约束 / 兄弟包互依赖）× 3 家明星库（HuggingFace / llama-index / langchain）做对比，标明 EverAlgo 在每维度的选择 + 选择依据（追溯到 evermem 契约 / 升级 A 不动 B / 算法库定位三条硬约束）。结论："不存在主参照概念，按维度独立选最优是工程常态"。同时清理 §1.3 自检"行业依据"重复段（已被矩阵覆盖）。版本兼容策略小节顶加 SemVer 简明定义（MAJOR.MINOR.PATCH 含义 + `0.x.x` vs `1.0.0` 边界 + 链 semver.org），避免后续术语跳读。仓库管理小节 GitHub URL 全改为 GitLab（公司内部平台），同时明示"行业实证 = HuggingFace/GitHub，EverAlgo 落地 = GitLab，模式相同平台不同"区分。 |
| v0.11 | 2026-04-23 | **反转 v0.9 multi-repo 决策为 monorepo + uv workspace**（参照 LangChain `libs/` + LlamaIndex 顶层多包，AI 圈"core + N 紧密联动包"业务结构主流；HuggingFace multi-repo 仅适用业务独立场景如 transformers/datasets/accelerate，与 EverAlgo 紧密联动结构错配）。承认我此前把 BOSS"只参照 HuggingFace（仅针对版本兼容策略）"越界扩展到仓库形态决策的错误。具体改动：① 参照矩阵第 3 行（仓库形态）从 HuggingFace → langchain / llama-index，依据列改为"AI 圈 core + N 紧密联动包业务结构主流；EverAlgo 业务结构同构"；② §1.3 仓库管理小节整段重写——单仓 `<gitlab>/<group>/everalgo`，仓内 `packages/everalgo-*/` 各含独立 `pyproject.toml` + 独立 SemVer + 独立 PyPI 发布（仿 LangChain `libs/partners/openai/`），明示"monorepo 是仓库形态，与发布粒度正交，'升级 A 不动 B' 完全成立"；③ 自检 Why monorepo 用 LangChain / LlamaIndex 实证 + 不选 multi-repo 列 HuggingFace 模式错配理由；④ §1.3 安装样例分两端：生产端用户 `pip install everalgo-X` 单 dist，开发端算法同学 `git clone everalgo && uv sync --all-packages` 仿 LangChain / LlamaIndex / Apache Airflow workspace 模式；⑤ "总览"句改：仓库形态参照库由 HuggingFace 替换为 langchain / llama-index。 |
| v0.12 | 2026-04-23 | **clustering 升级为独立工具性子包**（BOSS 决议）。原 v0.5 "暂不独立 / 视 T7 / T11 决议"作废。改动：① §1.2 工具性子包表 2 → **3**（boundary / rank / **clustering**），含 `centroid` + `llm_direct` 两种聚类策略；② §1.2 子包统计 10 → **11 个 subpackage**（4 产品 + 3 工具 + 3 基础 + 1 测试），distribution 7 → **8**；③ §1.2 目录速览框加 `everalgo-clustering` 框（state-in/state-out 接口注释）；④ §1.2 自检"Why clustering 暂不独立"改写为"Why clustering 独立"——profile 合并和 skill 聚合都需聚类、共享实现入独立子包、迭代友好；⑤ §1.3 拆分清单加 `everalgo-clustering` 行；user-memory / agent-memory 依赖 distribution 加 `clustering`；⑥ §1.3 依赖关系图重画——加 clustering 节点，user/agent_memory 同时依赖 boundary + clustering；⑦ §1.3 monorepo 仓内目录加 `packages/everalgo-clustering/`；⑧ §1.3 安装样例 user-memory / agent-memory 自动拉依赖列表加 `everalgo-clustering`；⑨ §1.3 pyproject 草案 user-memory `dependencies` 加 `everalgo-clustering>=0.1.0,<2.0.0`（注释 "profile 合并相似 atomic_fact 用"）；⑩ §1.3 参照矩阵业务结构同构描述 6 → 7 紧密依赖产品/工具包；⑪ §2.4 聚类算子段重写：从"内部私有模块" → "独立工具性子包"，明示两种策略 + state-in/state-out + 策略层面取舍归 T7；⑫ §3 T11 标 ✅ RESOLVED；策略层 centroid vs llm_direct 取舍仍归 T7。 |
| v0.13 | 2026-04-23 | **文档结构改革：详细论证拆出 ADR 文档**（BOSS 方法论纠正——选型不光看大家怎么做，还要从优劣分析 + 需求适配度倒推；这类详细论证不应拥塞主文档）。新建 `everalgo/docs/decisions/` 目录 + 9 个 ADR 文档（Architecture Decision Record）：① ADR 001 仓库形态 monorepo + uv workspace；② ADR 002 多 distribution + 独立 SemVer；③ ADR 003 PEP 420 namespace package；④ ADR 004 LLM providers 内嵌 llm/；⑤ ADR 005 testing 公开子包；⑥ ADR 006 clustering 独立工具性子包；⑦ ADR 007 版本兼容策略 HuggingFace 风；⑧ ADR 008 re-export 式 facade 而非 Client 类；⑨ ADR 009 命名规范 llama-index 风。每个 ADR 用统一 6 段结构：状态 / 背景 / 候选方案 / 客观优劣分析（不带 EverAlgo 滤镜）/ 对 EverAlgo 适配度评估（逐条优劣评估）/ 决策 + 行业实证印证。`decisions/README.md` 索引 + 论证哲学说明（避免 cargo cult）。design.md §1.2 / §1.3 自检条目改写为"简短结论 + ADR 链接"——长论证从主文档移走，主流程清晰；读者按需点链接深究"为什么不那样"。版本日志加本条记录方法论纠正与文档结构改革范围。 |
| v0.14 | 2026-04-23 | **§1.2 进一步瘦身**（BOSS 反馈"§1.2 里面的内容也可以适当抽出去"）。删除 §1.2 内 4 段冗长论述：① 整段"为什么 re-export 而不是 Client 类 facade"（13 行表格 + 4 阵营论述）→ 浓缩为单句指 ADR 008；② 整段"主流算法库的物理布局（5 个最知名实证）"（13 行 5 表实证）→ 单句指 ADR 008；③ testing/ 行业实证 + 两件套内容（11 行）→ 测试辅助一行加 ADR 005 链接；④ "外部契约对齐机制 re-export 代码示例"（28 行 `__init__.py` 完整代码）→ 4 行概念说明 + 链接 ADR 008 实施细节。design.md 行数 727 → 670（减 57 行）。主文档主流程清晰度提升，读者按需深究点 ADR 链接。 |
| v0.15 | 2026-04-23 | **§1.4 sync/async 双接口反转 + ADR 010 创建**（BOSS 两条质疑：① "肯定不用 DI"——§1.4 不必再对比 DI ② "算子为什么是 async 的呢"——质疑算子默认 async 设计）。WebFetch 4 个明星 AI 库实证：litellm `completion()` + `acompletion()`、instructor `Instructor` + `AsyncInstructor`、llama-index `query()` + `aquery()`、langchain Runnable `invoke` + `ainvoke` —— **100% 双接口模式，无单 async 反例**。EverAlgo 反转之前"算子全 async"误判为：**I/O 算子提供 sync + async 双接口**（`extract` / `aextract` 等，async 名 `a` 前缀仿 litellm/llama-index/langchain），**纯计算算子保持同步**。改动：① 新增 [ADR 010](decisions/010-sync-async-dual-interface.md) sync/async 双接口模式（含明星 4 库实证 + 实施降本路径基类派生 / `asgiref.sync` + 反方案 async-only / sync-only / anyio 评估）；② `decisions/README.md` 索引加 ADR 010 行；③ §1.4 标题去掉"禁用 DI"（已无对比意义）；§1.4 第 1 条改"I/O 算子双接口 + 纯计算同步"；§1.4 自检删 DI 对比 + 加 "Why I/O 双接口" / "Why 纯计算同步" / "Why 不用 Client 类" 三条（指 ADR 010 / ADR 008）；④ §2.3 算子分层表加 sync/async 接口形态列，全部 I/O-bound 算子改为双接口（`{extract, aextract}` 等），加双接口实现规范注 + ADR 010 链接。 |
| v0.16 | 2026-04-23 | **ADR 010 实施细节段重写 + 实施推荐反转**（BOSS 决定性约束信息：主用户 evermem = FastAPI 异步服务）。改动：① 扩展行业实证从 4 项目到 9 项目（加 httpx / OpenAI SDK / Anthropic SDK / redis-py / SQLAlchemy 5 个），区分双类（持状态客户端）vs 双方法（无状态算子）两条命名路线，明确 EverAlgo 归后者；② 加"决定性约束：主用户 evermem = FastAPI 异步服务" 段——async 主战场约束 async 接口必须 native；③ 加 B3 sync-first + thread pool 性能瓶颈精确分析（FastAPI 100 QPS / LLM 1s 场景：native async ~100 RPS vs B3 ~32 RPS，差 ~3x），thread pool 调大代价（每 thread 8MB 内核栈 + context switch overhead），结论 B3 不适配 EverAlgo 主用户 FastAPI；④ B1（手写两份）vs B2（async-first + asgiref）覆盖范围 vs 维护成本 trade-off 表，决策 B2（理由：evermem 100% async + sync 次要 + 单实现降本）；⑤ 文档明示 sync 接口边界："仅限非 event loop 环境调用"（与 litellm `acompletion()` / dspy 同样约定）；⑥ 实施样例代码（`async_to_sync` 单算子写法 + `DualInterface` 基类自动派生写法）；⑦ 退路：B1 何时回滚的 3 条触发条件；⑧ §1.4 / §2.3 实施规范注同步更新（async 必须 native + sync 限非 event loop）。修正之前 v0.15 推荐"手写两份"的过度保守判断 —— evermem 不调 sync 接口故"覆盖全场景"对主用户零额外收益。 |
| v0.17 | 2026-04-28 | **§1.4 review 修复 6 处与 ADR 010 v0.16 决议不一致**：① 第 1 条措辞从抽象"I/O 算子双接口"改为"async-first + sync 桥接"明示主用户 evermem = FastAPI 主战场，加主路径 / sync 桥接两个具体例子 + sync 接口边界声明；② 第 3 条示范从 `complete(...)` (sync) 改为 `await acomplete(...)` (async 主路径)；③ 第 4 条 contextmanager 从只 `with use(client)` 扩展为 `async with` / `with` 双形态；④ 自检 "Why I/O 算子双接口" 删除"Jupyter sync 顺手"v0.13 旧表述（与 v0.16 决议矛盾）+ 删除"基类派生"措辞（langchain B3 写法，不是 EverAlgo 选的 B2）；⑤ 自检改写为 "Why async-first + asgiref 桥接 sync"，含 evermem FastAPI 约束 + B3 性能数字（3x 差距：~100 vs ~32 RPS）+ B2 asgiref 单实现 + sync 边界；⑥ 整段把 ADR 010 v0.16 关键约定（sync 仅限非 event loop）从 ADR 链接深处提到 §1.4 主体明示。 |
| v0.18 | 2026-04-28 | **"纯计算算子用同步函数"补社区共识权威实证**（BOSS 质疑 "这个有依据吗，社区公认？"）。WebFetch 3 处官方文档：① **Python 官方 asyncio 文档**（[asyncio-dev "Running Blocking Code"](https://docs.python.org/3/library/asyncio-dev.html)）明确 "Blocking (CPU-bound) code should not be called directly [in async] ... would be delayed by 1 second"，推荐 `run_in_executor` + `ProcessPoolExecutor`；② **FastAPI 官方文档** "for CPU bound operations ... use normal `def`"；③ **NumPy 官方文档** "NumPy targets compute-bound operations (not I/O-bound), where async provides minimal benefit"。延伸：numpy/scipy/sklearn/pandas/pytorch/jax 全 sync API 无反例。改动：① §1.4 第 2 条加 3 文档实证 + 重计算 caveat（≤10ms 安全 / 10-100ms 边缘 / >100ms 需 `run_in_executor`）；② ADR 010 §"纯计算算子保持单一同步" 段加 3 处官方文档表格 + 计算时长判断标准 + 重计算未来场景指引（保持 sync def，由 caller `run_in_executor` 包装而非改写 async def 内部 sync 阻塞 anti-pattern）。论证从无出处论断升级为有 3 处权威实证。 |
| v0.19 | 2026-04-28 | **"纯计算算子不实现两个版本"补 9 项目实证 + langchain 反例分析**（BOSS 质疑 "对于这种函数，不需要实现两个版本了？大家都是这样？"）。严格核查：① **8 家主流项目 100% 纯计算 sync only**（numpy / pandas / sklearn / pytorch / scipy / litellm / llama-index / OpenAI SDK / httpx）② **唯一反例 langchain LCEL**——`PromptTemplate` / `OutputParser` 等纯计算 Runnable 节点被强制提供 `ainvoke`，但默认通过 `run_in_executor` 派生（B3 thread pool 性能代价），是为 LCEL chain `invoke / ainvoke` 接口统一接受的代价。EverAlgo 非 chain 框架（算子被 evermem 单调用，不组装 async chain），无 langchain 统一性需求 → 按主流 8 家惯例 sync only。改动：① §1.4 第 2 条加 9 项目实证 + langchain 反例分析 + 主流 vs chain 框架的场景区分；② ADR 010 §"纯计算算子保持单一同步" 段加 9 项目对照表 + langchain 特殊原因解释 + EverAlgo 非 chain 框架结论。修正之前 "社区共识" 论断的口径——精确为"主流 8 家共识 + langchain chain 框架反例"，避免读者疑惑"langchain 不就有 ainvoke 吗"。 |
| v0.20 | 2026-04-28 | **§1.4 删除"依赖住在模块里"行**（BOSS：措辞模糊；非标准术语，新读者要猜）。该条想表达的"模块级全局依赖访问，不用 DI / Client / 参数传依赖"已由 §1.4 自检 "Why 模块级函数 + 全局配置（不用 Client 类）" → ADR 008 + 配置三层覆盖一行 + memory feedback_algo_lib_no_di 共同覆盖，无需主体重复。 |
| v0.21 | 2026-04-28 | **§1.4 删除"配置三层覆盖"行**（BOSS：措辞含糊"什么意思"）。该条想表达的"环境变量 < 全局 setter < contextmanager"三层优先级机制已在 §2.5 LLM Scene Routing 机制完整详述（含示例代码 / env 变量映射规则 / contextmanager 用法 / 内部路由伪代码）。§1.4 是代码风格总览，避免与 §2.5 重复，删除该行让 §2.5 唯一讲述。 |
| v0.22 | 2026-04-28 | **§1.4 "Protocol 作为类型注解"补 PEP 544 / typing 官方文档出处**（BOSS 质疑 "有依据吗"）。WebFetch 三处权威源核证 ✅ 共识：① PEP 544 原文 "it's _not necessary_ to subclass explicitly"；② Python typing 官方 "structural subtyping (static duck-typing)"；③ mypy 官方 "static equivalent of duck typing"；工业实证 SQLAlchemy `util/typing.py` 对内置 `dict` / `_GenericAlias` 用 Protocol 检查结构兼容（不可能继承）。补出处 + 加 `@runtime_checkable` + `isinstance` caveat（仅查属性存在性 / EverAlgo 不依赖此用法）。 |
| v0.23 | 2026-04-28 | **新增 ADR 011 + §1.4 Protocol 行重写**（BOSS 质疑 "AI 明星算法库项目也是这么做的吗"，挑战 SDK 阵营定位不同不能作先例）。AI 生态 Protocol 用法严格核查（10 项目源码扫描）：① **同定位 "有外部调用能力的算法库" 阵营**实证 —— DSPy `CodeInterpreter` / LlamaIndex `VectorStore`/`GraphStore` / instructor `*Handler` 全 `@runtime_checkable Protocol`，与 EverAlgo 同推理路径；② **反例 LangChain** —— `BaseLLM`/`BaseChatModel`/`BaseRetriever` 全 `ABC + abstractmethod`（8 Protocol / 34 ABC / 64 abstractmethod 计数），但 LangChain 持 chain 状态 + lifecycle hooks，与 EverAlgo 无状态算子（H4）场景不同；③ **SDK 阵营出局** —— OpenAI/Anthropic SDK 虽 Protocol 主导但定位纯网络代理（无算法），stainless 自动生成器动机与算法库无关，不引用以避免 cargo cult。改动：① 新建 [ADR 011 Protocol vs ABC](decisions/011-protocol-vs-abc.md)，6 段结构（含同定位 4 项目实证表 + LangChain 反例分析 + SDK 不引用说明）；② `decisions/README.md` 索引加 ADR 011；③ §1.4 第 4 行 Protocol 措辞改为强调 "EverAlgo 算子无状态（H4）"驱动 + 指向 DSPy/LlamaIndex/instructor 同模式，链 ADR 011；④ §1.4 自检加 "Why Protocol（不用 ABC）" 条目链 ADR 011。 |
| v0.24 | 2026-04-28 | **§1.4 测试三件套修正：respx 出局、`unittest.mock.patch` 入主位、补 DSPy 实证**（BOSS 质疑 "有依据吗"）。同定位算法库测试实证核查：① **monkeypatch** ✅ 完全主流（pytest 官方原文 "safely set/delete an attribute, dictionary item or environment variable" 正是为"覆盖全局"设计）；② **contextmanager scoped 替换** ✅ 主流（DSPy `with dspy.context(lm=...)` 是 EverAlgo `with everalgo.llm.use(client)` 的同款 threading.local 实证）；③ **respx ⚠️ 有前提**——只 mock httpx，同定位 4 项目里 DSPy / LlamaIndex / instructor 都不用，仅 openai-python 主用。EverAlgo `LLMClient` 是用户实现 Protocol（不强制底层 httpx），respx 列在主线给读者"必须配 httpx"的隐性绑定误导，**出局**；④ **`unittest.mock.patch` 入主位**——DSPy / instructor / openai-python 实际通用 mock 兜底（DSPy `mock.patch("litellm.completion")` / LlamaIndex 自定义 MockLLM + patch dispatcher）。改动：§1.4 第 5 行 `monkeypatch / respx / contextmanager` → `monkeypatch / unittest.mock.patch / contextmanager`，并加 DSPy `with dspy.context(lm=...)` 同款实证。 |
| v0.25 | 2026-04-28 | **§2.1 场景代码 + §2.2 I/O 形态修正：去除过度设计的 AsyncIterator + 修方法名一致性**（BOSS 质疑 "运行期五个典型场景 为什么是 async for 这样的"）。三处不一致：① 场景 A/B/C 用 `async for memcell in ConvMemCellExtractor().extract(...)` —— `extract` 是 sync 方法名（按 ADR 010 双接口规范应 `aextract`/`adetect`），`async for` 配 sync 方法语法不通；② 边界检测返回 `AsyncIterator[MemCell]` 是过度设计——LLM 切分点 structured output 一次性返回，下游 episode/foresight/fact 各 Extractor 对单 MemCell 独立执行不是 pipeline 重叠，`async for` 无收益；③ 场景 E `await rank.episodic(rank_input)` 缺 `arank` 方法名（按 §2.3 应 `rank.episodic.arank`）。修正：① 边界算子返回类型 `AsyncIterator[MemCell]` → `list[MemCell]`，与 LlamaIndex `NodeParser.get_nodes_from_documents() -> List[BaseNode]` / LangChain `TextSplitter.split_documents() -> List[Document]` 同款（切分类算子主流不流式）；② 场景 A/B/C/D/E 全部方法名补 `a` 前缀（`adetect` / `aextract` / `arank` / `aparse`），与 §1.4 + §2.3 + ADR 010 一致；③ `async for memcell in xxx` → `memcells = await xxx.adetect(...); for memcell in memcells:` 形式，list 直接 for 消费；④ §2.2 加同行实证 + 未来若需 LLM streaming 加 `astream_detect()` 互补不替换 `adetect()`（litellm `acompletion(stream=True)` 模式）；⑤ 场景 E `rank.profile.rank` 用同步形式（按 §2.3 算子表 profile 是同步纯计算 cosine）。 |
| v0.26 | 2026-04-28 | **`Conv` → `Chat` 命名矫正**（BOSS 质疑 "ConvMemCellExtractor 这个 conv 是什么，明星项目怎么命名的"）。两层驱动：① 业界对齐：6 项目核查（OpenAI SDK `ChatCompletion` / LlamaIndex `ChatMessage`/`SimpleChatEngine` / HuggingFace `apply_chat_template`/`ChatType` / DSPy `ChatAdapter` / LangChain 新 API `ChatPromptTemplate`）—— **5/6 用 Chat**，零项目用 `Conv` 作 Conversation 缩写；② 歧义消除：PyTorch `nn.{Conv1d, Conv2d, Conv3d, ConvTranspose*d, LazyConv*}` 12 个高频类强占用 `Conv` 命名空间作 Convolution 缩写，AI 算法库读者看 `ConvMemCellExtractor` 第一反应是 "Convolutional Memory Cell"（联想 ConvLSTM），与"对话边界检测"语义南辕北辙。memsys_opensource 现状代码核查（Explore agent 实读）：现状 `ConvMemCellExtractor`（基类）+ `AgentMemCellExtractor`（继承 Conv ~90% 复用）2 套实现；`Workspace*` 完全未落地；现状 enum 用全词 `RawDataType.CONVERSATION`，类名却用 `Conv` 缩写——本身就有 Conv 缩写问题。改动 4 文件 ~23 处替换：① design.md 11 处（§1.2 facade 双路径示例 ×3、§1.2 子包结构 ASCII 图 ×2、§1.3 re-export 多 distribution 示例 ×2、§2.1 场景 A 调用代码、§1.2 算法同学物理路径 import、§1.2 boundary 子包内容表、§2.3 算子表 boundary 行）；② [ADR 003 namespace](decisions/003-namespace-package-pep420.md) H1 契约示例 ×1；③ [ADR 008 facade](decisions/008-re-export-vs-client-facade.md) 6 处（背景契约引用、A 方案描述、实施细节代码 ×2、两条访问路径示例 ×2）；④ [ADR 009 命名规范](decisions/009-naming-convention-llama-index-style.md) 5 处（H1、PascalCase 反例、CapWords 示例、矫正表 + 加二层说明 PEP 8 lowercase + Conv 歧义消除双重驱动）。**注**：首轮 grep 用 `boundary\.conv\b` 模式，遗漏了 §2.3 算子表 `boundary.{conv, workspace, agent}` 这种含花括号写法和 §1.2 ASCII 树形图 `{conv, workspace, agent}.py` / `re-export Conv/WorkspaceMemCellExtractor` 共 3 处，BOSS 询问场景 E 时连带核查到，已补齐。同步成本：memsys_opensource 现状代码（`ConvMemCellExtractor` / `RawDataType.CONVERSATION` / `CONV_BATCH_BOUNDARY_DETECTION_PROMPT` / `conv_memcell_extractor.py`）落地 EverAlgo 时一并重命名；evermem 文档作者侧（zhanglibin）需同步矫正（与 v0.7 PEP 8 矫正同样处理）。Anthropic SDK 用 `Message` 不用 `Chat` 是单家偏好（5/6 主流仍是 Chat），可忽略。 |
| v0.27 | 2026-04-28 | **§1.2 + §2.3 rank 子包双层抽象（4 业务 facade + 算法工具）**（BOSS 反问 "rank 按业务场景分这个合理吗，不同场景有不同操作吗"）。先核查 4 Ranker 真实算法差异度（Confluence「检索 EverAlgo」zhanglibin v3）：① **Episodic 真独有** —— hierarchical ep→fact 展开 + score_propagation；② **Profile 真独有** —— cosine + threshold + 去重，**完全无 fusion 无 rerank**；③ **Case / Skill 高度同构** —— 都是 `fusion → 加权 → (可选) rerank`，仅加权字段不同（case `quality_score` / skill `maturity + confidence`）+ skill 无 rerank。业界对比：LangChain `BaseRetriever` / LlamaIndex `BaseNodePostprocessor` / DSPy `Retrieve` 都按算法策略分类，**业界 3/3 主流不按业务记忆类型分**——EverAlgo 按业务分是少数派。合理性来源：业务参数（字段映射 / scene / 加权策略）封装在 EverAlgo Ranker facade 内部，evermem 调用对齐 memory_type 分支无需懂 rank 细节，对应现状 `_search_episodic_memory` / `_search_profile` 等 4 method 分支结构。改进：抽双层抽象——`rank/{episodic, profile, case, skill}` 是业务 facade（evermem 调用面）+ `rank/{fusion, weight, rerank}` 是算法工具（算法同学迭代面，4 facade 内部组合调用 + 算法同学新增 ranker 时直接复用），消除 Case/Skill 同构代码冗余；类比 LangChain `BaseRetriever` 接口 + 实现 / LlamaIndex `BaseNodePostprocessor` 接口 + 实现双层抽象。改动：① §1.2 子包结构 ASCII 图 rank 子包拆 `{episodic, profile, case, skill}.py` + `{fusion, weight, rerank}.py`；② §1.2 自检加 "Why 4 业务 facade + 算法工具双层" 段（含与 LangChain/LlamaIndex 双层抽象类比 + 业界对比说明）；③ §2.3 算子表 `rank.{...}` 行重组：`Rank facade` 行（4 业务 facade，episodic/case/skill 双接口 + profile 单 sync）+ `算法工具` 行（fusion / weight / rerank 三类）；④ T14 议题扩展为 `rank.{fusion, weight, rerank}` 完整 API 签名定稿（含 weight `weighted_score` / `multi_field_weighting`，rerank `rerank/arerank` scene 参数）。 |
| v0.28 | 2026-04-28 | **§1.4 加 "EverAlgo 命名强契约" 段**（BOSS 反馈 "让使用 everalgo 的用户来区分 await/no await 也太不友好了吧 大家怎么做的"）。业界 4 种实现核查：① **X1 thread pool 派生**（`async def arank: return await to_thread(rank, ...)`）—— LangChain LCEL / LlamaIndex `BaseNodePostprocessor` / litellm `acompletion` 主流，但 `asyncio.to_thread` 官方文档明确"primarily intended for IO-bound functions"，毫秒级纯 Python 计算用 thread pool 负优化（thread switch 10us-1ms > 计算本身），LangChain 用 X1 是 chain 框架兜底（任意 Runnable 自动 async），不为性能优化（v0.19 已论证 EverAlgo 不为 chain 统一性付此代价）；② **X2 async def 直执行**（`async def arank: return rank(...)`）—— FastAPI 文档允许 "async def even if no await inside"，开销 μs 级，但**4 个 AI 算法库零实证**（DSPy/LlamaIndex/LangChain/litellm 全不用），cargo cult 创新风险 + 读代码人误以为真异步；③ **X3 纯 sync only**（EverAlgo 当前 + DSPy 同款）—— DSPy 纯计算 metric / Adapter 无 `a*` 版本，仅含 LLM 调用层暴露 `acall`/`aforward`。结论：**X3 是技术最优**，BOSS 担心的"用户区分 await 不友好"由命名约定解决——**`a` 前缀 = async 是强契约**，用户看名字即知，无需记忆。改动：① §1.4 标题之下加 callout 段（命名强契约定义 + 同 DSPy/litellm/instructor 命名引证 + 场景 E 实例对照）；② 5 条主体不动；③ 自检不动（命名契约是规则不是设计决策，含义已在 callout 自洽）。 |
| v0.29 | 2026-04-28 | **§2.2 / §2.3 / §2.5 错误处理与跨 Provider fallback 责任划分明确**（BOSS 反问 "错误直接抛异常 evermem 负责重试 / 降级 别的算法库也是这样吗 IO 异常重试也不管？"）。业界 9 项目重试矩阵核查：① **OpenAI / Anthropic SDK** 默认 `DEFAULT_MAX_RETRIES = 2`（hardcoded `_constants.py`，覆盖 connection / 408 / 409 / 429 / 5xx + `x-should-retry` header）—— **行业事实存在**；② **DSPy / LlamaIndex / instructor 100% 在 SDK 之上叠加 retry**（DSPy 3 / LlamaIndex-OpenAI 3 / LlamaIndex-Anthropic 10 / instructor 1）—— 早期 SDK 不可靠的历史包袱；③ **LangChain Core 反其道而行**：`BaseChatModel` 不内置 retry，但 `ChatOpenAI` 透传 `max_retries` 给 OpenAI SDK，**SDK 那 2 次仍生效**；④ **真正"裸抛 IO 异常完全不管"零先例**——最接近的 LangChain Core 也仍依赖 SDK 默认兜底。EverAlgo 当前措辞"错误直接抛异常 evermem 负责重试"字面反主流，需精确化：① 区分**两层 fallback**——配置 fallback（启动期 scene 未声明用 default 配置，归 EverAlgo）vs 运行时 fallback（调用失败跨 Provider 切换，归 evermem）；② 明示底层 SDK 默认 2 次重试本来就在（不显式 `max_retries=0` 即生效）；③ evermem 通过 LLMClient Protocol 装饰器实现跨 Provider fallback。改动：① §2.2 I/O 形态约定 "错误直接抛异常" 行扩为 3 条精确措辞（依赖 SDK 默认 2 次 + 跨 Provider 归 evermem + 算法层不叠 retry 的论据）；② §2.3 末尾错误处理段同步精确化；③ §2.1 场景 E 启动期配置注释 / §2.5 启动期配置注释从 "scene 调用失败降级 default" 改为 "配置 fallback：scene 未声明时用 default 配置（启动期，**不是运行时降级**）"；④ §2.5 加 "跨 Provider 运行时 fallback：归 evermem" 完整段，含 X1（FallbackLLMClient 装饰器）+ X2（litellm.Router 适配 LLMClient Protocol）两条实施路径完整代码示例；⑤ §2.5 自检从 "Why fallback 开关在 default 层" 重写为 "Why fallback_to_default 是配置 fallback 不是运行时 fallback" + "Why 算法层不加 retry"（含 SDK 默认 2 次 + DSPy/LlamaIndex 叠加历史包袱论证 + LangChain Core 同款分层哲学引证）。对齐 LangChain Core 哲学（核心不重试 + 依赖 provider SDK 内置）。 |
| v0.30 | 2026-04-28 | **§2.5 整段重写：scene 路由剥离出 EverAlgo + LLM 调用栈架构定型（P2 各家原生 SDK + Protocol + 双层路由 + 7 LLMError 子类）**（BOSS 4 轮反问驱动："现在调 LLM 用什么 / LiteLLM 不是 toy 吗 / B 派 4/4 ABC 一定有原因 / scene 路由在 provider 路由之上 / LLMError 只有 Letta 有吗 / 场景路由是不是不应该让 everalgo 做"）。多轮调研事实集成：① **现状代码**：memsys_opensource 用 aiohttp 自实现 OpenAI compat（`openai_provider.py`）+ 自重试 5 次 + 库内 `FallbackLLMProvider` decorator + `LLMScene` enum 分发，**业界 13 项目零先例 C 派**；② **业界 13 项目 LLM 底层矩阵**：B 派 7（LlamaIndex/LangChain/AutoGen/Letta/Ragas/Instructor/mem0 各家原生 SDK）/ A 派 LiteLLM 强绑 2（DSPy/Cognee）/ D 派多适配器 2（CrewAI 原生优先 + LiteLLM 兜底 / smolagents）—— **B 派事实标准**；③ **LiteLLM 不是 toy**（45k stars / 261M 月下载与 OpenAI SDK 同量级 / DSPy 主路径），**但 2026-03 供应链投毒 + proxy CVE 真实风险**；与 enterprise EnterprisePipelineRouter 职责重叠 → 不强绑；④ **B 派 100% ABC 真实驱动 4 条**（concrete mixin / 多继承胶水 / 实例化早失败 / 生态级 framework 需求）—— EverAlgo 一条不命中（Protocol 仍稳健，加 decorator/contextmanager + `@runtime_checkable` 加固）；⑤ **scene/provider 两层路由分离**——scene 是业务编排概念（业界 4/4 算法库不做），provider 是 SDK 适配实现层（Letta `LLMClient.create` `match-case` 同款）；⑥ **LLMError 业界 5 派分布**：A SDK 自带完整层级（OpenAI 20 / Anthropic 17）/ B 算法库统一 LLMError（仅 Letta 13）/ C 平铺 1-2 个（mem0/CrewAI/DSPy）/ D 不定义（LlamaIndex/AutoGen）/ E 通用层级（LangChain core/Instructor）—— EverAlgo 取 Letta 模式精简到 7 子类 + LangChain partner 混合多重继承策略（让 SDK 原生 catch + 语义 catch 双有效）。**关键决策**：scene 路由完全剥离出 EverAlgo，由 evermem 端 SceneRouter 持有 + `everalgo.llm.use(client)` contextmanager scoped 替换调用（DSPy `dspy.context(lm=...)` 同款）。改动：① §1.3 everalgo-core 子包 `llm` facade 描述去 scene 路由 + 加 LLMClient Protocol / LLMError 7 子类；② §2.1 启动期配置示例从 `llm={default, scenes, fallback_to_default}` 简化为 `llm=LLMConfig(...)` 单一全局；③ **§2.5 整段重写**：标题"LLM Scene Routing 机制"→"LLM 配置与注入"；删除场景清单 9 scene 表 / scene 路由示例 / `EVERALGO_LLM_SCENE_*` env 变量；新增 everalgo/llm/ 子包结构（types/client/errors/routing/providers）；启动期单一 LLMClient 注入（DSPy 同款）；evermem 端 SceneRouter 完整代码示例 + `async with everalgo.llm.use(scene_router.get(...))` scoped 切换；内部路由伪代码改为仅 Provider 路由 `build_client(config)` `match-case`；自检 5 条重写（Why scene 不在 EverAlgo + Why 全局+contextmanager + Why Provider 路由仍在 + Why 算法层不加 retry + 行业依据）；④ T4 议题更新：`complete/stream_json` → `chat/stream`；T13 prompt 选择改为算子内部决策（解耦 LLM 模型选择）；§2.3 算子表 rank.rerank scene 参数改为 prompt 参数。memsys_opensource 现状代码迁移清单：`OpenAIProvider`(aiohttp 自实现 5 次重试) → `everalgo.llm.providers.openai_compat`（`openai.AsyncOpenAI` + ApiKeyRotator 薄 wrapper 业务保留）；`_MAX_RETRIES=5` → 删除（依赖 SDK 默认 2 次）；`FallbackLLMProvider`（库内）→ 移到 evermem 端；`LLMScene` enum + `_get_provider_for_scene` → 移到 evermem 端 SceneRouter。 |
| v0.31 | 2026-04-28 | **新建 [ADR 012 LLM 抽象层架构](decisions/012-llm-stack-architecture.md)**（BOSS 反问 "路由的决策文档在哪"——v0.30 改动只在 changelog 流水账，未沉淀独立 ADR）。ADR 012 6 段结构整合 6 轮调研事实：① 5 维度候选（底层 SDK / 抽象基类 / 路由分层 / 错误层级 / 重试责任）；② 客观优劣分析含 P1/P2/P3 三派 + LiteLLM 工业级真实定位（45k stars / 261M 月下载 / 2026-03 供应链事件）；③ EverAlgo 适配度逐条评估；④ 5 决策（D1 P2 各家原生 SDK / D2 Protocol 维持 / D3 双层路由分离 scene 出 + provider 留 / D4 LLMError 7 子类混合多重继承 / D5 算法层不加 retry 依赖 SDK 默认 + evermem 跨 Provider fallback）；⑤ 实施细节（子包结构 + 核心接口签名）；⑥ 行业实证印证（13 项目矩阵 + LLMError 5 派分布 + B 派 ABC 4 真实驱动 + scene 路由 4/4 不做实证 + memsys_opensource 现状迁移清单）；⑦ 6 个后续演化触发条件。`decisions/README.md` 索引加 ADR 012；§2.5 自检末尾加 ADR 012 链接（指向完整决策与调研事实）。 |
| v0.32 | 2026-04-28 | **§2.1 + §2.5 LLM 注入改为 3 层（per-call / scoped / default）DSPy 同款**（BOSS 反问 "如果 evermem 管 scene 路由 那 everalgo.configure 的意义是什么"）。问题本质：scene 完全归 evermem 时，若 evermem 用 contextmanager / per-call 显式注入每个调用，全局 default 永远不会被触发——`configure(llm=...)` 失去意义。深层调研：DSPy 已有 3 层注入实证（`dspy.settings.configure(lm=...)` 全局 + `dspy.context(lm=...)` scoped + `predictor(..., lm=...)` per-call），优先级 per-call > scoped > default。生产 evermem 用 per-call / scoped 显式注入；dev / 测试 / Jupyter / 简单脚本用 configure 一次走天下；3 层并存解决"全局 default 何时触发"的疑问——**default 是为非 evermem 用户场景**而存在。改动：① §2.1 启动期 + 运行期场景代码完整重写——启动期 configure 注释为"可选兜底"+ evermem 端持有 SceneRouter 显式展示；场景 A/B/C/D 5 处用 `aextract(memcell, llm=scene_router.get(...))` per-call 主路径；场景 E 用 `async with everalgo.llm.use(scene_router.get("rerank"))` scoped wrap 批量切换（4 ranker 同 client）；场景 E `rank.profile.rank` / `rank.fusion.rrf` 保持 sync 纯计算无 LLM 参数。② §2.5 算法同学日常调用展示算子方法签名加可选 `llm: LLMClient | None = None`+ `everalgo.llm.current()` 内部按优先级查找 + `LLMNotConfiguredError` 兜底；新增 "3 层注入路径" 表（per-call / scoped / default 用法 + 优先级 + 典型场景）；evermem 端 SceneRouter 示例改为 per-call + scoped 两种风格并存（单调用按 scene → per-call；多调用同 client → scoped wrap）；contextvars.ContextVar 实现机制说明（DSPy `dspy.context` 同款 threading.local + async safe）。③ §2.5 自检 "Why 全局 + contextmanager" 重写为 "Why 3 层注入（per-call > scoped > default）"——含 DSPy 实证 + 生产 / dev / 测试 用户场景区分 + LLMNotConfiguredError 兜底。 |
| v0.33 | 2026-04-28 | **multi-key 轮转 + 跨 Provider fallback 开源版不做**（BOSS "key 轮转 / fallback 属于企业级 现在先不考虑" + "不是归 enterprise，是开源版不做，不属于开源版需求"——避免 opensource 设计文档预设 enterprise 行为越界）。原则：opensource 设计只描述开源版做什么 + 不做什么，**不预设 enterprise 怎么做**（那是 enterprise 自己的事，opensource 文档不染指）。改动 14 处措辞从"归 enterprise"修正为"开源版不做 / 不属于开源版需求"：① §2.2 错误处理段；② §2.3 末尾错误处理；③ §2.5 跨 Provider fallback 段标题"高可用编排归 enterprise" → "高可用编排不属于开源版需求"+ 整段措辞精确化（删除"enterprise 落地时如何如何"具体说明 + 改为"部署方有需要时可自行实现"，不预设 `EnterpriseFallbackClient` 等具体类名）；④ §2.5 装饰器示例代码 `EnterpriseFallbackClient` → `MyFallbackClient`（去除 enterprise 命名预设）；⑤ ADR 012 EverAlgo 适配度评估段；⑥ ADR 012 D5 决策标题与内容；⑦ ADR 012 memsys_opensource 现状代码迁移清单：`ApiKeyRotator` 行 / `FallbackLLMProvider` 行措辞从"归 enterprise"改为"开源版不做"+ 注 "部署方有需要时可自行实现 LLMClient 装饰器叠加"。删除所有 X1/X2 详细 fallback 代码示例 + `EnterpriseFallbackClient` / `MultiKeyLLMClient` 等预设 enterprise 类名引用。同时简化错误处理段——只描述 opensource 直接抛 `LLMError`（7 子类含语义 + 混合多重继承让原生 SDK catch 仍有效），由 caller 决定如何处理；保留 `LLMClient` Protocol 装饰器扩展性简短示意作为部署方扩展指引，不预设具体落地形态。 |
| v0.34 | 2026-04-29 | **§2.4 聚类算子整段重写 + §2.3 算子表 cluster_id 关系列 + ADR 006 论据精确化**（BOSS 反问 "centroid + llm_direct 是当前实现吗" / "cluster 和 atomicfact 没关系吧 你说话要有依据" / "你这样写没人看得懂"）。深入读 memsys_opensource 现状代码（cluster_manager/manager.py 794 行 + memory_extractor/*.py + mem_memorize.py 三阶段编排）核证 5 个 extractor 与 cluster_id 的真实关系：① **AgentSkillExtractor 直接强类型依赖**（agent_skill_extractor.py:427/476/783/795 等多处 `cluster_id: str` 参数）；② **ProfileExtractor 间接数据依赖**（profile_extractor.py:89 `cluster_episodes: list[Dict]`，编排层 mem_memorize.py:485-501 按 cluster_id 反查 eventid_to_cluster + fetch memcells 后传入）；③ **AtomicFactExtractor / EpisodeExtractor / ForesightExtractor 零依赖**（grep 全文 0 个 cluster_id 引用）。修正 §2.4 早期错误描述："Profile 合并 AtomicFact" → 错（Profile 是 LLM 增量编辑 episode，不合并 AtomicFact）；"centroid + llm_direct 两策略并列" → 错（实际是 1 个 K-means 状态机 + 2 条相似度路径，共享 centroid 增量更新 + 时间窗口）；"AgentSkill / Profile / AtomicFact 都 import clustering" → 错（聚类是 cluster_id 生产者，下游算子是 cluster_id 消费者，编排在 caller 层）。改动：① **§2.4 整段重写**——人话版：开头一句话讲清楚 cluster 输出 cluster_id + 下游消费表（AgentSkill 直接 / Profile 间接 / 其他无关）+ 编排顺序伪代码 + 4 条自检；②【§2.3 算子表精确化**——拆分 `user_memory.{episode, foresight, atomic_fact, profile}` 一行为两行（前 3 个独立 / profile 接 `cluster_episodes`）；拆分 `agent_memory.{case, skill}` 为 case 独立 / skill **直接接 `cluster_id: str`**；clustering 行从 `{centroid, llm_direct}.{assign, aassign}` 改为 `IncrementalClusterer.{cluster, acluster_with_llm}` 单类双入口；③ **ADR 006 论据精确化**：背景段加 "v0.34 论据精确化注"（基于 memsys_opensource 现状核证），明示"Profile 不合并 AtomicFact" / "聚类不是两策略并列而是状态机 + 两路径" 两处修正；独立子包决议保持（论据反而更稳：多消费者前置依赖）。算子抽象：`IncrementalClusterer` + `ClusterState` dataclass（state in/out 显式传递，caller 持久化）+ sync `cluster()` / async `acluster_with_llm()` 双入口；下游算子签名：`AgentSkillExtractor.aextract(case, *, cluster_id, existing_skill, llm)` / `ProfileExtractor.aextract(memcell, *, cluster_episodes, old_profile, llm)` / `AtomicFact / Episode / Foresight` 各自 `aextract(memcell, *, llm)`。 |
| v0.35 | 2026-04-30 | **§2.4 cluster 算子第二轮重构（双函数 + 值对象 ClusterState 3 字段精简）**（BOSS 一点一点 step-by-step 推进 + 多轮挑战："为什么不拆两个方法 / state 必须有吗 / state 由 OS 还是 Core 管 / `update_state` 看着奇怪 / LLM 决策入参向量也给 / embed 进 everalgo 怎么用 / rank 进 embed 不进的判定线"）。v0.34 设计的 `IncrementalClusterer` 单类双入口 + `centroid+llm_direct` 二策略表述 → 第二轮全部推翻。新设计：① **拆双公开函数**——`cluster_by_geometry`（user_memory 调）/ `cluster_by_llm`（agent_memory 调）平级独立，不再单类内 has_case 分发，对齐 sklearn `cluster.{KMeans,DBSCAN,OPTICS}` 行业实证 + 包分层（user/agent_memory 各自 ClusterState 物理隔离，删 `case_cluster_ids`）；② **`ClusterState` 精简为 3 字段**（centroids / counts / last_ts）—— online incremental K-means 算法本质 3 项累积信息，删 v0.34 现状 10 字段中 7 个历史包袱（`event_ids` / `timestamps` / `vectors` / `cluster_ids` / `eventid_to_cluster` / `case_cluster_ids` / `next_cluster_idx`）；③ **值对象 in/out**——frozen dataclass + `state.assign(cid, vector, ts)` 类型方法（Python 标准库 `frozenset.union` / `Path.with_suffix` 同模式），不原地 mutate，事务安全；④ **责任分层明确化**——EverAlgo 定义类型 + 演化方法（assign / from_dict / to_dict）+ 算法本质（候选检索 + 决策 + LLM 含 prompt 内置 + 失败降级），evermem 决定实例创建 / 持久化 / 加锁 / 时机；⑤ **embed 不进 EverAlgo 决议二次确认**——核证 design.md:158 已立 + 给完整理由（embed 无算法 IP 载体 / model 选择属业务决策 / sklearn FAISS DSPy 行业对齐 / 与 LLM 含 prompt 是 IP 载体形成对比）+ 给假设性 everalgo-embed 内置形态 + 客观利弊 + 不内置作为最终决议；⑥ **rank 进 EverAlgo vs embed 不进的判定线**——是否承载算法 IP（prompt / 公式 / 策略 / 实现）：rank 含 4 ranker + fusion 公式（RRF）属 IP / embed 仅 SDK 转包属基础设施服务；该判定线可推广 review 其他子包；⑦ **LLM 决策入参不含向量**——LLM 是文本模型，吃 text + 标量（query_text + similarity 标量 + recent_texts），raw float 数组对 LLM 无意义；候选含 cluster_id + similarity（标量），不传 vector；⑧ **fetch_previews callback 保留**——caller 持有完整事件存储，不该让算法库重复维护 `eventid_to_cluster`；候选簇是算子内 Top-K 召回后才确定，caller 不能提前预取所有簇文本（O(N) 浪费）；callback 是该场景最优解（且仅此一处 callback，删除 `_callbacks` / `on_cluster_assigned` 死代码）；⑨ **LLM 失败降级保留在算子内**——`LLM 失败 → 几何 top-1 + threshold` 是 cluster 算法 IP 的一部分（与基础设施 retry 不同）；删 `_call_llm_for_clustering` 内部 `for attempt in range(3)` retry（对齐 ADR 012）；⑩ **Step-by-step 全流程对照现状代码核证**——cluster 完整闭环 7 步（evermem 选 clustering_text → 算 vector → load state → 调算子 → 算子内候选检索 + 决策 + state 演化 → save state → 触发下游 ProfileExtractor / SkillExtractor），每步标 EverAlgo vs evermem 边界 + 现状 mem_memorize.py / manager.py 行号对应。改动：① **§2.4 整段重写**（41 行 → 174 行，含公开 API 完整签名 + 包分层映射 + 责任边界 + 算子内部流程伪代码 + 9 条自检 + 与现状代码差异表 11 项）；② **§2.3 line 537 同步**——clustering 行从 `IncrementalClusterer.{cluster, acluster_with_llm}` 改为 `cluster_by_geometry / cluster_by_llm` 双公开函数。净化收益：核心算法代码量 `cluster_by_geometry` 3 行 + `cluster_by_llm` 10 行，vs 现状 `_cluster_memcell_embedding` + `_cluster_memcell_llm` 共 ~185 行（删 `_get_embedding` / `_extract_text` / `_callbacks` / `_stats` / 内部 retry / `case_cluster_ids` 维护 / 死字段）。 |
| v0.36 | 2026-05-06 | **§1.2 + §1.3 parser 归类工具性 + 同层兄弟互依赖原则精化**（BOSS 反问"§1.3 兄弟包互依赖是什么意思"——回答时发现 §1.2 与 §1.3 line 234 描述硬冲突）。问题本质：§1.2 line 44 把 parser 列为"产品性子包（4）"，但 §1.3 line 192 拆分清单显示 `everalgo-knowledge dependencies = ["core", "parser"]`——按 §1.2 归类 parser + knowledge 都是产品性兄弟，knowledge 依赖 parser **直接违反** line 234"产品性兄弟互不在 install_requires"原则。语义层面 parser 不产出记忆类型（不像 user_memory 产出 Episode / agent_memory 产出 Case），只产出 ParsedContent 中间产物，本质是被 knowledge 消费的"中间转换工具"，归工具性更符合"产品包产出记忆类型 / 工具包做中间转换"边界。改动：① **§1.2 line 44 "产品性子包（4）" → "产品性子包（3）"**（删除 parser 行）；② **§1.2 line 53 "工具性子包（3）" → "工具性子包（4）"**（加入 parser 行，描述："多模态原始文件 → ParsedContent；消费者 knowledge / evermem 环节 1"）；③ §1.2 line 53 工具性子包小标题加注"不产出记忆类型，是中间转换工具"语义边界明示；④ §1.2 line 59 clustering 行描述同步更新到 v0.35 形态（双公开函数 + ClusterState 3 字段，链接 §2.4 / ADR 006）；⑤ §1.2 line 114-117 everalgo-clustering 目录速览框从旧 `{centroid, llm_direct}.py` 改为 `_algorithm.py`（双公开函数实现 + 私有原子）；⑥ §1.2 line 146 子包统计"4 产品性 + 3 工具性" → "3 产品性 + 4 工具性"；⑦ **§1.3 line 175 行业参照矩阵末列 "EverAlgo 产品性 4 包业务独立" → "同层兄弟（产品性 3 包 + 工具性 4 包）各自横向独立"**；⑧ **§1.3 line 234 第 1 条原则"核心兄弟包之间不互相依赖——产品性 4 包... 互相不在 install_requires" → "同层兄弟包之间不互相依赖——产品性 3 包之间 + 工具性 4 包之间各自横向独立 + 跨层依赖（产品包依赖工具包）不属兄弟互依赖是合理的拓扑下行依赖"**；⑨ §1.2 line 157 "Why clustering 独立工具性子包" 自检条目 stale 表述"profile 合并相似 atomic_fact + skill 聚合相似 case" → "user_memory.profile 按 episode 簇上下文做增量编辑 + agent_memory.skill 按 case 簇聚合"（与 v0.34 ADR 006 论据精确化对齐）；⑩ §1.3 line 363 pyproject 草案 everalgo-clustering 注释 "profile 合并相似 atomic_fact 用" → "episode 簇用（profile 增量编辑上下文）"；⑪ §3 line 903 T11 RESOLVED 注追加 v0.35 二次精化说明（双公开函数形态 + ClusterState；早期 `centroid + llm_direct` 两策略并列表述作废）。原则提炼：**同层兄弟禁兄弟互依赖是设计铁律**（HuggingFace 实证 + 满足"升级 A 不动 B"硬约束 + 防 diamond dependency），跨层依赖（产品 → 工具 → base）是合理拓扑。changelog 历史 line 951 / 962 中"4 产出型 + 3 工具型"等措辞作为历史事实保留不改。 |
| v0.37 | 2026-05-06 | **§1.3 line 380 "Why 不加 meta package" 自检条目错误论证修正**（BOSS 反问"如果要全部安装是什么命令"——回答时发现现状无 meta 需手列 4 dist 这件事本身没问题，但**否定 meta 的论证"meta 通常 pin 整套版本与独立升级意图相反"是错的**）。事实核查：LlamaIndex `llama-index` meta + LangChain `langchain` meta 都用宽松约束（`>=X.Y,<next_major`），不 pin 死版本，"升级 A 不动 B"（H2）在 meta + 宽松约束模式下完全兑现——meta 不必然 pin，原论证基于错误前提。改动：line 380 自检条目重写——结论"不加 meta"维持，但论证理由替换为 4 条真实依据：① 主用户 evermem 按场景单装即可，"全装"是低频场景；② 全装手列 4 个顶层 dist 一行命令搞定（user_memory + agent_memory + knowledge + rank 自动拉齐 8 个 dist，rank 因不被产品包依赖必须显式列）；③ meta 包要持续维护对各 dist 的兼容性约束（即便宽松约束，新增 dist / breaking 升级时仍要发新 meta），边际收益低于维护成本；④ HuggingFace transformers/datasets/accelerate 同模式不提供 meta。原 "meta 必 pin 整套版本" 论据明示标注"不构成反对 meta 的依据"，避免读者误以为 meta 包都会丢失独立升级能力。提炼方法论：**结论可以维持，但论据必须真实**——错的论据即便指向对的结论，也会污染未来基于该论据的衍生决策（如未来 BOSS 看到 "meta 必 pin" 会误以为加 meta 必须 pin 全 dist 版本）。 |
| v0.38 | 2026-05-06 | **§1.4 line 397 prompt 存储格式从"md 资源文件"改为"Python 字符串模块"**（BOSS 反问"prompt 用 md 存有依据吗，明星项目怎么做的，md 还是 TOML 还是 YAML"）。10 家明星项目 prompt 存储格式调研（WebSearch 2026-05-06）：① **算法库阵营 5/5 全用 Python 字符串硬编码**——DSPy（Signature class 动态生成 prompt 不存文件）/ LlamaIndex（PromptTemplate Python 类 f-string）/ instructor（Pydantic docstring + Jinja2 模板）/ mem0（Python 字符串）/ memsys_opensource 现状（`memory_layer/prompts/{en,zh}/*.py` 模块）；② **模型层** HuggingFace transformers 用 Jinja2（chat_template.jinja 存 tokenizer_config.json，1 家）；③ **端到端框架阵营 3/3 用外置文件**——LangChain（JSON / YAML / Python 三选，推荐 JSON / YAML）/ CrewAI（YAML 首推 agents.yaml + tasks.yaml）/ Semantic Kernel（Jinja2 + 自家模板）；④ **`.md` 在算法库阵营 0 实证**——仅见于工具 user config（Claude Code CLAUDE.md / cursor rules），那是用户配置不是算法 prompt；⑤ memsys_opensource 现状代码核证：`src/memory_layer/prompts/{en,zh}/{cluster, agent, atomic_fact, conv, episode_mem, foresight, group_profile_merge, profile}_prompts.py` 全部 Python 文件，与 line 397 当前"md 资源文件"原则**直接矛盾**。改动：line 397 重写为"Prompt 是 Python 字符串模块（如 prompts/en/cluster_decision.py 内 `CLUSTER_DECISION_PROMPT = "..."`），不外置 .md / .yaml / .toml；与算法库阵营 DSPy / LlamaIndex / instructor / mem0 / memsys_opensource 现状 5/5 一致；多语言通过子模块组织（prompts/en/ + prompts/zh/）；改 prompt = 改 .py 字符串。端到端框架阵营 LangChain / CrewAI / Semantic Kernel 外置 YAML / Jinja2 不适配 EverAlgo 算法库定位"。原"明星实证"提法不准——明星 = 端到端框架；EverAlgo 是算法库，应对齐算法库阵营。提炼方法论：**任何"明星实证"声明必须先指明阵营**（算法库 vs 端到端框架 vs 模型层 vs 工具配置），不能笼统说"明星项目都这样"——不同阵营对同一问题的选择往往相反（参见 v0.35 embed 不进 EverAlgo 决议同样的"按阵营对齐"逻辑）。 |
| v0.39 | 2026-05-06 | **prompt 实现整体改为 Python 字符串模块（多处连锁同步）**（BOSS 反问"prompt 用 .py 文件，evermem 自定义怎么办" → 给出 per-call + monkey-patch 双路径方案 → BOSS"先把 prompt 改成 .py 实现吧，整体改一下"）。改动 6 处：① **§1.2 line 66 prompts 子包描述**——"Prompt 加载 / 三层 override / validator 机制（具体 prompt 文件就近放各子包 prompts/）" → "Prompt validator 机制（占位符 / 长度校验）+ 多语言子模块组织约定；具体 prompt 字符串就近放各子包 `prompts/{en,zh}/<name>.py` 内作为 module-level 常量；evermem 自定义路径：算子 per-call `prompt=` 参数（细粒度，主路径）/ caller monkey-patch 模块常量（启动期粗粒度全局）"；② **§1.2 line 95 everalgo-core 目录速览**——`prompts/ {loader, validator}.py` → `prompts/ validator.py`（删 loader.py，Python 模块直接 import 不需文件加载逻辑）+ 加注释"各子包 prompts/{en,zh}/<name>.py 自带 module 常量"；③ **§1.2 line 160 自检 "Why prompts 不进 providers"**——"prompts 机制（三层 override / 渲染 / validator）是算法资源不是可替换 provider；具体 prompt 就近放各子包 prompts/" → "prompts 是算法 IP（与算法绑定 / 调优 prompt = 调算法），不是可替换 provider；具体 prompt 字符串就近放各子包 `prompts/{en,zh}/<name>.py` 作为 Python 模块常量；自定义路径见 §1.4 prompt 实现段"；④ **§1.4 line 397 后追加 evermem 自定义 prompt 双路径段**——明示两条路径（per-call `prompt=` 参数为细粒度主路径 + caller monkey-patch 模块常量为粗粒度全局）+ 行业实证（LlamaIndex `update_prompts` / HuggingFace `tokenizer.chat_template = "..."` 同款）+ 明示不引入的复杂机制（prompt_dir / configure(prompts={...}) 全局 default 层 / scoped contextmanager / setter API 都是过度设计，per-call + monkey-patch 已覆盖 100% 场景）；⑤ **§3 T3 议题精化**——"涉及 LLM / prompt / rerank 多组配置" → "仅 LLM 相关（v0.39 起 prompt 不再走 configure，详见 T5）"；⑥ **§3 T5 议题标 RESOLVED**——"Prompt 资源组织：目录布局、命名空间、override 机制、sandbox validator 范围" → ✅ RESOLVED 注（Python 字符串模块 + per-call + monkey-patch 双路径，validator 保留）。10 家明星项目调研按定位聚类的关键发现（v0.38 已记录）：算法库阵营 5/5 全用 Python 字符串硬编码；端到端框架阵营 3/3 用外置文件；HuggingFace 模型层用 Jinja2；.md 在算法库阵营 0 实证。EverAlgo 算法库定位 → 算法库阵营对齐。一致性：本次改动后 design.md 全文 prompt 表述统一为"Python 字符串模块"语义，删除 "prompt_dir" / "loader.py" / "三层 override" 等基于"外置文件资源"假设的 v0.x 早期遗留。 |
| v0.40 | 2026-05-06 | **§2.4 编排顺序段拆为 user_memory / agent_memory 双支显式伪代码**（BOSS 反问"§2.4 编排顺序里面怎么没有 IncrementalClusterer 呢"——根因：v0.35 把 IncrementalClusterer 拆成 `cluster_by_geometry` / `cluster_by_llm` 双函数，但编排顺序段仍用通配符 `cluster_by_*` 占位，让读者无法直观看出实际调哪个函数）。改动：① 编排顺序段标题下方加引导句"按 user_memory / agent_memory 包分层各自编排，分两支并行（两支用各自独立的 ClusterState 物理隔离；每支各自调对应的双函数之一）"；② Phase 1 拆为 Phase 1（user_memory 支调 `cluster_by_geometry`）+ Phase 1'（agent_memory 支调 `cluster_by_llm`）两段独立伪代码块，分别明示 `text` 来源 / `vector` 算法 / 锁 key（`trigger_clustering:{user_id}` vs `trigger_clustering:{agent_id}`）/ 算子调用签名 / state_store 持久化；③ Phase 2 同样按"独立路径 vs 依赖 cluster_id"二分，各 Extractor 调用签名明示（AtomicFact/Foresight 独立 / Profile 接 cluster_episodes / Skill 直接接 cluster_id）；④ 段尾加注 "关于 v0.34 IncrementalClusterer 单类双入口：v0.35 已拆为双公开函数（详见上方公开 API 段 + 自检 Why 拆双函数）；编排顺序里不再出现 IncrementalClusterer 类名"——明示给后来读者避免混淆。删除原 `cluster_by_*` 通配符占位写法。提炼方法论：**伪代码不该用通配符占位**——若两条路径调用不同函数，应该按路径分别展开，否则读者无法对照实际签名理解；通配符只在多次调用同一函数时才合理用。 |
| v0.41 | 2026-05-06 | **§2.4 公开 API 块每个入参加注释解释**（BOSS 反问"§2.4 公开 API 里面，注释解释入参"——原 API 代码块仅函数 docstring + 极少注释，读者要看入参用途必须查后文 step-by-step / 算子内部流程伪代码段）。改动：原 API 代码块（54 行）扩为带详细行级注释版（72 行），覆盖 5 个类型 / 函数的全部入参 + 返回值：① **ClusterState 三字段**——centroids（每簇中心向量；cosine 决策的对照向量）/ counts（每簇事件数；centroid 增量公式 (C*n+v)/(n+1) 的 n）/ last_ts（每簇最后更新时间；时间窗约束用），加 class docstring "online incremental K-means 必需的 3 项信息"；② **ClusterState.assign** —— cluster_id（None=新建分配 ID/非 None=归入）/ vector（caller 算好；centroid 更新或新簇初始化）/ timestamp（更新 last_ts = max(prev, ts)）+ 返回值注释（frozen 值对象；事务安全）；③ **ClusterConfig 四字段** —— threshold（几何决策阈值；几何路径用 / LLM 路径失败降级用 + 算法可解释默认 vs 生产值 caller 自调说明）/ time_window_days（仅 cluster_by_geometry 用）/ k_candidates（仅 cluster_by_llm Top-K 召回）/ llm_skip_threshold（仅 fast path 用）；④ **Candidate** —— cluster_id + similarity（_find_candidates 输出 / _decide_by_* 输入）；⑤ **cluster_by_geometry 入参** —— vector（caller 算 EverAlgo 不做 embed）/ timestamp（caller 已 parse）/ state（caller 持久化 + load）/ config（阈值族）+ 返回值；⑥ **cluster_by_llm 入参** —— 加 query_text（拼 prompt 用，典型为 task_intent）/ llm（按 LLMScene 路由）/ fetch_previews（callback 双行注释明示入参 / 返回值结构 + N 由 caller 决定）。每行注释精简（< 80 字符），行末对齐。读者收益：单看 API 代码块即可理解每个入参的用途 / 来源 / 用法，不必跳到 step-by-step 段或与现状代码对照表去查。 |
| v0.42 | 2026-05-06 | **§2.4 cluster_by_llm 入参 `fetch_previews` callback → `cluster_previews: dict` 直接传**（BOSS 反问"fetch_previews 不能直接传查好的文本吗，为什么要传一个回调函数"——根因：早期 callback 论据"caller 不能提前预取所有簇文本（O(N) 浪费）"基于 N >> K 假设；EverAlgo 实际场景单 owner_id 簇数 N ≤ 50，`k_candidates = 30`，IO 比仅 ~1.6 倍，不构成"严重浪费"）。4 候选方案对比（A. callback / B. 直接传 dict / C. caller 拼 Layer 1 / D. 两次算子调用）→ 方案 B 综合最优（无反向依赖 + 函数签名干净 + IO 浪费可控 + caller 端可缓存 + 对齐 sklearn / FAISS / BIRCH 算法库阵营 caller 提供完整数据模式）。改动 5 处：① **§2.4 公开 API 块 cluster_by_llm 入参签名**——`fetch_previews: Callable[[list[ClusterId]], Awaitable[dict[ClusterId, list[str]]]]` → `cluster_previews: dict[ClusterId, list[str]]`，注释明示 key/value 结构 + "算子 Top-K 召回后从此 dict 取候选对应项"用法；② **§2.4 责任边界表 caller 第 ③ 项**——"实现 fetch_previews callback（caller 自家事件存储反查）" → "批量预取 cluster_previews（caller 遍历 state.centroids.keys() 自家事件存储反查最近文本，可缓存）"；③ **§2.4 算子内部流程 cluster_by_llm 5 步伪代码**——`previews = await fetch_previews([c.cluster_id for c in candidates])` → `previews = {c.cluster_id: cluster_previews.get(c.cluster_id, []) for c in candidates}`（从已传入 dict 取，无 await callback）；④ **§2.4 编排顺序 Phase 1' agent_memory 支**——cluster_by_llm 调用前加一步"批量预取所有簇最近文本"显式伪代码（`cluster_previews = await case_repo.fetch_recent_intents_by_clusters(list(state.centroids.keys()), max_per_cluster=5)`），调用入参 `fetch_previews=fp` → `cluster_previews=cluster_previews`；⑤ **§2.4 设计自检 "Why fetch_previews 是 callback 而非 state 字段" → "Why cluster_previews 直接传 dict 而非 callback"**——重写论证：明示 v0.42 修正 + 实际场景 N/K 数值 + 三条优势（纯数据 / 函数签名干净 / 可缓存）+ 对齐 sklearn KMeans.fit(X) / FAISS index.add(vectors) / BIRCH 算法库阵营。提炼方法论：**架构选择不是论证一次定终身**——早期论据基于的场景假设可能不成立，要敢于在新证据下推翻自己的设计；callback 反向依赖在算法库定位下是 code smell，能用纯数据传递就别用 callback。 |
| v0.43 | 2026-05-06 | **§2.5 LLM 3 层注入解析从 7 行 boilerplate 抽为 `everalgo.llm.resolve(llm)` 单行**（BOSS 反问"每一个用 LLM 的函数都要写这么一段吗"——根因：v0.32 引入 3 层注入时，算子内代码示例直接展开 `llm or current() + if None raise LLMNotConfiguredError` 7 行；20 个用 LLM 的算子各自重复 = 严重 code smell + 一致性维护负担）。改动：① **§2.5 算法同学日常调用代码示例**——`EpisodeExtractor.aextract` 内 7 行 boilerplate（`client = llm or everalgo.llm.current()` + `if client is None: raise LLMNotConfiguredError(...)`）→ 1 行 `client = everalgo.llm.resolve(llm)`；引导句加说明"内部用 `everalgo.llm.resolve(llm)` 单行封装 3 层 fallback + 未注入抛 LLMNotConfiguredError，避免每个用 LLM 的算子重复写 boilerplate"；② **新增 `resolve()` 实现伪代码段**——明示 3 层 fallback 逻辑（per_call → ContextVar scoped → 全局 default）+ 全 None 时抛 `LLMNotConfiguredError`；明示"算法库内一处定义，所有算子共用"；③ **§2.5 自检条目同步精化**——"EverAlgo 算子方法签名补可选 llm 参数 + everalgo.llm.current() 内部按优先级查找" → "+ 内部用 everalgo.llm.resolve(llm) 单行封装 3 层 fallback + 未注入抛 LLMNotConfiguredError（算法库内一处定义，所有算子共用，避免每个算子重复 7 行 boilerplate）"。设计原则：**算法库内的横切公共逻辑（如 LLM 注入解析、错误兜底）应抽为辅助函数，而非每个算子重复展开**；DSPy / LlamaIndex / instructor 同款（DSPy `Predict.forward` 内部从 `dspy.settings.lm` 单行取 / LlamaIndex `Service.llm` 单行取）。提炼方法论：**示例代码不该展示 boilerplate** ——文档示例如果让读者觉得"每个 caller 都要重写一遍"，要么是 API 设计有缺陷（缺辅助函数），要么是文档示例展开不当（应直接给最佳实践版本）；boilerplate 应封装在算法库内，文档示例只展示 caller 实际写的代码。 |
| v0.44 | 2026-05-06 | **全文术语统一：`EverAlgo opensource` → `EverAlgo` / 独立 `opensource` → `evermem`**（BOSS"把所有 EverAlgo opensource 改成 EverAlgo，所有 opensource 改成 evermem"）。改动 7 处：① **3 处 `EverAlgo opensource` → `EverAlgo`**——line 519（错误处理段"EverAlgo opensource 库自身不加额外重试层"）/ line 546（§2.3 末尾错误处理段）/ line 922（§2.5 高可用编排段"EverAlgo opensource 调用失败直接抛 LLMError"），简化品牌名（EverAlgo 本身即开源算法库，不需 opensource 定语）；② **4 处独立 `opensource` → `evermem`**——line 914（"opensource 设计原则：保证'接上就能用'" → "evermem 设计原则"）/ line 924（"opensource 设计文档不预设具体实施方式" → "evermem 设计文档"）/ line 927（"# opensource 调用形态" → "# evermem 调用形态"）/ §2.5 上文一处。**保留不改**：① **`memsys_opensource` 项目代号**（指现状代码仓库的 GitLab/GitHub 真实名称，改成 `memsys_evermem` 会破坏与现实仓库的对应——按 negative lookbehind `(?<!memsys_)opensource` 匹配跳过）；② **changelog 历史行**（v0.30/v0.32/v0.33/v0.34/v0.38 等记录里的"opensource"是历史事实记录，不动）；③ 中文"开源版"措辞（BOSS 指令仅针对英文 opensource 一词，line 912 标题"高可用编排不属于开源版需求" / line 914 / 916 等中文"开源版"保留）。提炼方法论：**全局术语统一时务必区分项目代号与定位描述词**——项目代号（如 memsys_opensource = GitLab 仓库名）是固定 reference，全局替换会破坏与外部系统的对应；定位描述词（如"opensource 设计原则"中的 opensource）是可重命名的 alias，按 BOSS 意图替换；changelog 历史是事实记录原则上不动。 |
| v0.45 | 2026-05-06 | **v0.44 字面替换 3 处语义错位精化（evermem → EverAlgo 回滚）**（v0.44 按 BOSS"所有 opensource → evermem"字面执行后发现 3 处实际主语应是 EverAlgo 不是 evermem，BOSS 确认"改"）。改动：① **§2.5 line 914**——"evermem 设计原则：保证'接上就能用'——单 client 调用 + SDK 默认 2 次重试兜底" → "EverAlgo 设计原则：保证'接上就能用'..."（描述的是 EverAlgo 算法库的简化策略，不是 evermem web 服务的设计原则）；② **§2.5 line 924**——"evermem 设计文档不预设具体实施方式" → "EverAlgo 设计文档不预设具体实施方式"（本文档 line 1 自述"EverAlgo 架构设计文档"，不是 evermem 设计文档）；③ **§2.5 line 927 代码块注释**——"# evermem 调用形态（接上就能用）" → "# EverAlgo 默认调用形态（接上就能用）"（更精确：这是 EverAlgo 提供给 caller 的最简默认调用形态，不绑死特定 caller 名）。提炼方法论：**字面执行 + 事后语义 review** 的工作流——对术语全局替换这种容易出语义错位的批量改动，先按字面执行得到"草稿"，再逐处 review 主语是否被改错；不要因为"BOSS 说所有都改"就跳过语义 review。v0.44 + v0.45 合在一起完成"opensource 术语统一"全部改动。 |
| v0.46 | 2026-05-06 | **§1.3 仓库管理段"参照"出处分层修正**（BOSS 抓出"pydantic-ai 是唯一 uv workspace 多 dist 实证"是孤证 + design.md 主体段措辞混淆 monorepo 形态与 uv workspace 工具）。4 项目实测调研（LangChain / LlamaIndex / Apache Airflow / Dagster 的 dev workflow，看根 pyproject.toml / CONTRIBUTING / Makefile / scripts/）：① **uv workspace 不是孤证**——Apache Airflow 是同形态大型实证（100+ workspace members + PEP 420 namespace `airflow.providers.*` + 单 `uv.lock` + `uv sync --all-packages`），ADR 001 line 160 早已列入；② **LangChain / LlamaIndex 自身不用 workspace**——LangChain `libs/<pkg>` 各自独立 venv + 独立 `uv.lock`（[libs/Makefile](https://raw.githubusercontent.com/langchain-ai/langchain/master/libs/Makefile)）；LlamaIndex 子包独立 venv + Pants 编排；它们只作"monorepo + 多 dist"形态实证，**不**作 uv workspace 工具实证；③ **其他路线**：Dagster 用 `python scripts/install_dev_python_modules.py` 拼超长 `uv pip install -e A -e B ...` 一把梭进单 venv（独立顶层名，无 namespace 共享），不适配 EverAlgo 同 namespace 需求；LlamaIndex 接受跨包隔离（改上游需手动重装），与 EverAlgo"算法库快速迭代"目标冲突。改动 design.md 主体段 3 处：① **line 249 标题**——"仓库管理：monorepo + uv workspace（参照 LangChain / LlamaIndex）" → "仓库管理：monorepo（参照 LangChain / LlamaIndex）+ uv workspace（参照 Apache Airflow / pydantic-ai）"；② **line 251 段正文**——拆为"形态"和"工具"两层独立举证：monorepo 参照 LangChain / LlamaIndex（明示"它们自身不用 uv workspace"）；uv workspace 参照 Apache Airflow（100+ workspace members 大型实证）+ pydantic-ai（同 namespace 多 dist 同形态）；③ **line 345 注释**——"workspace 模式（仿 LangChain / LlamaIndex / Apache Airflow）" → "workspace 模式（仿 Apache Airflow / pydantic-ai；LangChain / LlamaIndex 是 monorepo 形态参照，不用 workspace）"；④ **§1.3 自检 "Why monorepo + uv workspace" 条**——加"两层论据"明示 monorepo 形态 vs uv workspace 工具的不同实证来源。提炼方法论：**"参照 X / Y"措辞要分层引用** —— monorepo 是仓库形态决策，uv workspace 是工具选型决策，两件事独立举证。把"形态参照库"和"工具参照库"混在同一句"参照 LangChain / LlamaIndex 用 uv workspace"会让读者误以为参照库也用 uv workspace（实际不用）；孤证（仅 pydantic-ai 实证）也不能立——必须翻出第二个同形态实证（Apache Airflow）证据才稳。BOSS 这次抓的是"措辞容易让读者把'形态实证'当成'工具实证'"的精确语义偏差。 |
| v0.47 | 2026-05-06 | **§2.5 自检 + ADR 012 阵营归类不一致修正（疑点 1）**（v0.46 后扫 design.md 同模式论据，发现 line 945 / 951 + ADR 012 line 124 / 168 / 286 把 LangChain / AutoGen 列入"业界算法库 4/4 实证"，但 design.md 其他章节一致归阵营外——line 400 v0.19 "唯一反例 langchain LCEL（chain 框架，非算法库）"/ line 402 v0.39 "端到端框架阵营 LangChain / CrewAI / Semantic Kernel"/ ADR 011 v0.23 "反例 LangChain — 持 chain 状态 + lifecycle hooks 与 EverAlgo 无状态算子场景不同"/ AutoGen 是 agent 框架同理）。**阵营归类不一致即论据 cherry-picking** —— 同一项目在不同章节按需归阵营是论据强度的弱点。改动：① **design.md line 945 自检 "Why scene 路由不在 EverAlgo"**——"业界算法库 4/4 实证：DSPy / LlamaIndex / LangChain / AutoGen 全部不做 scene 路由" → "**业界主流 LLM 库 4/4 不做 scene 路由（横跨 3 阵营无差别）**：算法库阵营 DSPy + LlamaIndex / chain 框架阵营 LangChain / agent 框架阵营 AutoGen。3 阵营都不做 → 反向印证 scene 路由不属任何阵营 LLM 库的职责"；② **design.md line 951 ADR 012 引用** "4/4 算法库不做 scene 路由实证" → "4/4 主流 LLM 库横跨 3 阵营不做 scene 路由实证"；③ **ADR 012 line 124**——"业界算法库 4/4 实证 (...) 不做 scene 路由" → "业界主流 LLM 库 4/4 横跨 3 阵营都不做 scene 路由 (DSPy + LlamaIndex 算法库 / LangChain chain 框架 / AutoGen agent 框架)，3 阵营都不做 → 反向印证不属任何阵营 LLM 库的职责"；④ **ADR 012 line 168 决策摘要**——"业界算法库 4/4 不做 scene 路由" → "业界主流 LLM 库 4/4 横跨 3 阵营 (算法库 / chain 框架 / agent 框架) 都不做 scene 路由"；⑤ **ADR 012 line 286 段标题 + 表格**——"### Scene 路由不在算法库（4/4 实证）" → "### Scene 路由不在 LLM 库（4/4 横跨 3 阵营实证）"，表格加"阵营"列分类标注 DSPy/LlamaIndex=算法库 / LangChain=chain 框架 / AutoGen=agent 框架，段尾加"3 阵营无差别"小结。**论点反加强**：从"算法库都不做"到"跨 3 阵营都不做"，证据范围更广（不限于算法库阵营），更证明 scene 路由是业务编排职责而非任何 LLM 库的职责。提炼方法论：**阵营归类必须全文一致** —— 同一项目（如 LangChain）在不同章节按论点需要换阵营归类是论据 cherry-picking，会被读者抓出"阵营贴标签是为了凑数"。正确做法是给项目固定阵营标签，论据描述按"是否横跨多阵营"展开（横跨 = 论点更强）。BOSS 抓的是"阵营归类"语义偏差，与 v0.46 抓的"形态参照库 vs 工具参照库"语义偏差是同一类问题——**论据出处的精确性必须经得起跨章节自洽检查**。 |
| v0.48 | 2026-05-06 | **§1.3 PEP 420 实证修正：拆 uv workspace 工具实证 vs PEP 420 实证两层（v0.46 余 bug）**（v0.46 修了"形态参照库 vs 工具参照库"两层，但残留一个 bug：把 Apache Airflow 同时列为"uv workspace 工具实证"+"PEP 420 namespace 大型实证"——实测 Airflow 的 `airflow/__init__.py` 末行是 `__path__ = pkgutil.extend_path(__path__, __name__)` **pkgutil-style legacy namespace**，不是 PEP 420 native；pydantic-ai 实测是 3 个**独立** namespace（`pydantic_ai` / `pydantic_evals` / `pydantic_graph` 各自有 `__init__.py`），不是同 namespace 多 dist。BOSS 抓出"顶层空一层这个做法规范吗，有依据吗"问题后系统核证）。**新增工业实证（实测各项目 `<namespace>/__init__.py` 状态）**：① **google-cloud-*** 100+ dist 共享 `google.cloud.*`（python-storage / api-core 等仓库的 `google/__init__.py` 与 `google/cloud/__init__.py` 实测均 HTTP 404）—— 工业级最大 PEP 420 实证；② **sphinxcontrib-*** 6 个 PyPA 官方分发的 Sphinx 扩展（applehelp / htmlhelp / qthelp / serializinghtml / devhelp / jsmath）共享 `sphinxcontrib.*` PEP 420 native；③ **PyPA 官方示例** [sample-namespace-packages](https://github.com/pypa/sample-namespace-packages) `native/` 子目录；④ **PyPA 官方文档**[packaging-namespace-packages.html](https://packaging.python.org/en/latest/guides/packaging-namespace-packages/#native-namespace-packages) 直接推荐 PEP 420 native namespace 用于"Py3-only + pip-only"项目（EverAlgo 满足两条：`requires-python=">=3.12"` + uv/pip 安装）。改动 3 文件：① **design.md §1.3 line 249-254 仓库管理段**——从"monorepo + uv workspace 两层"扩为"monorepo + uv workspace + PEP 420 namespace 三层独立举证"，每层附实证 + 反例区分（明示 Airflow 是 pkgutil-style 不是 PEP 420，仅作 uv workspace 工具实证）；② **design.md §1.3 自检 "Why monorepo + uv workspace"** → "Why monorepo + uv workspace + PEP 420 namespace"，3 维度独立溯源；③ **AGENTS.md §2 第 2 段 namespace 描述** 加 PyPA 官方推荐链接 + google-cloud-* / sphinxcontrib-* 工业实证；4 段 uv workspace 描述明示 Airflow / pydantic-ai 仅作 uv workspace 实证、namespace 实现不是 PEP 420；④ **plan 文件依据表** 新增"PEP 420 native namespace 多 dist 共享"行（PyPA 官方 + google-cloud-* + sphinxcontrib-* + PyPA 示例 + 反例区分），uv workspace 行注明"仅作 workspace 实证 not PEP 420"。**EverAlgo 实施未受影响**：之前的实施本身就是 PEP 420 native（`packages/*/src/everalgo/` 顶层无 `__init__.py`），与 PyPA 官方推荐 1:1，且 `everalgo.__path__` 实测包含 8 个目录（步骤 11 自检）—— 错的只是论据出处，方案本身正确。提炼方法论：**"namespace 实现"和"workspace 工具"是两个独立维度**——同一项目可能在 workspace 维度是好榜样、namespace 维度是反例（如 Airflow），不能把"业内大型项目"笼统当作各维度都实证。验证方法是**实测每个候选项目对应 `<namespace>/__init__.py` 文件**：404 = PEP 420 native；含 `extend_path` / `declare_namespace` = pkgutil/setuptools legacy（不是 PEP 420）。BOSS 这次抓的是 v0.46 修复时维度合并不彻底——**论据维度划分要尽可能细，每条出处只承担一个维度的实证责任**。 |
