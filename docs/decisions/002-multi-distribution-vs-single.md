# ADR 002: 发布粒度 — 多 distribution + 独立 SemVer

## 状态

✅ **Accepted** — 2026-04-23（v0.6 提议；v0.9-v0.12 细化）

## 背景

EverAlgo 包含多个算法子包（parser / user_memory / agent_memory / knowledge / boundary / rank / clustering 等）。**发布粒度**决定用户在 PyPI 端如何安装、如何升级、如何控制依赖。

相关硬约束：
- **H2** 用户独立升级 A 不动 B（BOSS 反复强调）
- **H3** 算法同学迭代速度
- **H7** 可识别系列前缀

## 候选方案

| 方案 | 描述 |
|------|------|
| **A. 单 distribution + extras** | `everalgo` 一个 PyPI 包，重依赖（whisper / paddleocr 等）放 `[parser]` extras，用户 `pip install everalgo[parser]` 选装 |
| **B. 多 distribution + 独立版本号** | 8 个独立 PyPI 包：`everalgo-core` + 7 个产品/工具包，各自独立 SemVer，用户按需 `pip install everalgo-user-memory` |
| C. 多 distribution + 整套 meta-package | B + 顶层 `everalgo` meta-package 依赖整套 |

## 客观优劣分析

### 单 distribution + extras 优势

| 维度 | 说明 |
|------|------|
| 跨子包重构成本零 | 单 dist 内任意改动一个 commit 完成 |
| 版本一致性强 | 用户装到的所有子模块永远是同一个版本 |
| 用户安装命令最简 | `pip install everalgo` 一行搞定 |
| PyPI 维护成本最低 | 一份 `pyproject.toml`、一份 release 流程 |
| 文档单点 | 一份 README、一份 CHANGELOG |

### 单 distribution + extras 劣势

| 维度 | 说明 |
|------|------|
| 用户无法独立升级子模块 | 升级 user_memory 必须同时升 agent_memory（绑定整体版本）|
| 重依赖打包复杂 | parser 重依赖（whisper / paddleocr 数百 MB）即使在 extras 也共享 dist 元数据 |
| 子模块版本演化耦合 | 一个子模块 breaking 改动强迫整 dist major bump |

### 多 distribution + 独立版本号 优势

| 维度 | 说明 |
|------|------|
| 用户独立升级子模块 | `pip install -U everalgo-user-memory` 不动其他包 |
| 子模块独立演化节奏 | 各子包独立 SemVer，breaking 改动只影响自己 + 直接下游 |
| 用户按需安装 | 不需要 parser 时不装 everalgo-parser，避免下载重依赖 |
| 子模块责任归属清晰 | 各 dist 独立 issue / changelog / release tag |

### 多 distribution + 独立版本号 劣势

| 维度 | 说明 |
|------|------|
| 跨子包重构成本高 | 需多 dist 同步 release（**与仓库形态相关**：multi-repo 加重；monorepo 大幅缓解，见 [ADR 001](001-multi-repo-vs-monorepo.md)）|
| diamond dependency 风险 | 多个下游包对共同 base 包约束区间不一致时安装冲突（**有解**，见 [ADR 007](007-version-compatibility-strategy.md)）|
| PyPI 维护成本 | N 个 `pyproject.toml`、N 套 release 流程（**monorepo + 自动化** 可缓解）|
| 文档分散 | N 个 README / CHANGELOG |

### 多 distribution + meta-package 优势/劣势

新增优势：用户 `pip install everalgo` 一键装齐套（学习曲线低）

新增劣势：meta 通常 pin 整套版本（`everalgo==1.0` → `everalgo-user-memory==X` + `everalgo-core==Y` + ...），**与"独立升级"意图反向**

## 对 EverAlgo 适配度评估

### 单 distribution 优势对 EverAlgo 的适配度

| 优势 | 适配度 |
|------|--------|
| 跨子包重构成本零 | ✅ 受益（H6 v0.x 演化阶段需要）—— **但 monorepo + 多 dist 也能达到（[ADR 001](001-multi-repo-vs-monorepo.md)）**，不是单 dist 独占 |
| 版本一致性强 | ⚠️ 不在意（用户对子模块版本一致性无强需求） |
| 安装命令最简 | ⚠️ 不在意（`pip install everalgo-user-memory` 也能自动拉依赖）|
| PyPI 维护成本低 | ⚠️ 可 mitigate（monorepo + 共享 release 自动化）|

### 单 distribution 劣势对 EverAlgo 的适配度

| 劣势 | 适配度 |
|------|--------|
| **用户无法独立升级子模块** | ❌ **致命**——直接违反 H2 硬约束 |
| 重依赖打包复杂 | ❌ 介意（parser 重依赖但 user_memory 不需要）|
| 子模块版本演化耦合 | ❌ 介意（H6 演化阶段不希望子模块互相绑死）|

### 多 distribution 优势对 EverAlgo 的适配度

| 优势 | 适配度 |
|------|--------|
| 用户独立升级子模块 | ✅ **强需要**（H2 直接命中） |
| 子模块独立演化节奏 | ✅ 强需要（H6 演化阶段独立 breaking 不互相绑死） |
| 按需安装 | ✅ 强需要（parser 重依赖与 user_memory 解耦）|
| 责任归属清晰 | ✅ 受益 |

### 多 distribution 劣势对 EverAlgo 的适配度

| 劣势 | 适配度 |
|------|--------|
| 跨子包重构成本高 | ⚠️ **可 mitigate**——monorepo + uv workspace 让跨 dist 重构变成单 commit（见 [ADR 001](001-multi-repo-vs-monorepo.md)）|
| diamond dependency 风险 | ⚠️ 可 mitigate——HuggingFace 风宽松约束 + 严守 SemVer（见 [ADR 007](007-version-compatibility-strategy.md)）|
| PyPI 维护成本 | ⚠️ 可 mitigate——monorepo 共享 release 自动化 |
| 文档分散 | ⚠️ 可 mitigate——主仓 README 索引各包，单包内自带 README |

### meta-package 评估

EverAlgo 有"独立升级 A 不动 B"硬约束，meta 与之反向，**直接排除**。

## 决策

**选 多 distribution + 独立版本号（方案 B），不加 meta-package**。

逐条统计：
- 单 dist 致命劣势 1 条（违反 H2），介意劣势 2 条
- 多 dist 强需要 3 条 + 受益 1 条
- 多 dist 劣势全部可 mitigate（依赖 ADR 001 monorepo + ADR 007 版本兼容）

## 实施细节

8 个 distribution 拆分：

| Distribution | 内容 | 依赖 |
|--------------|------|------|
| `everalgo-core` | types / llm / prompts / config / protocols / testing | — |
| `everalgo-parser` | 多模态解析 | core |
| `everalgo-boundary` | 3 MemCellExtractor + 共享 tokenize | core |
| `everalgo-rank` | 4 Ranker + fusion | core |
| `everalgo-clustering` | centroid + llm_direct | core |
| `everalgo-user-memory` | Episode / Foresight / AtomicFact / Profile | core, boundary, clustering |
| `everalgo-agent-memory` | Case / Skill | core, boundary, clustering |
| `everalgo-knowledge` | KnowledgeExtractor | core, parser |

详见 design.md §1.3 拆分清单。

## 行业实证印证

同推理路径下，AI 圈核心 + 紧密联动子包结构都用多 dist + 独立版本号：

| 项目 | distribution 数 | base 包 | 独立 SemVer |
|------|---------------|---------|-----------|
| LangChain | 60+ | `langchain-core` | ✅ |
| LlamaIndex | 150+ | `llama-index-core` | ✅ |
| Apache Airflow | 60+ | apache-airflow | ✅ |
| HuggingFace 全家桶 | 4+ | `huggingface_hub` | ✅ |

**关键不冲突**：HuggingFace 在仓库形态选 multi-repo，但发布粒度同样是多 dist + 独立 SemVer。仓库形态（[ADR 001](001-multi-repo-vs-monorepo.md)）和发布粒度是正交维度。

## 后续演化触发条件

1. **某子包业务上消亡**：如废弃 knowledge → 该 dist 标 deprecated，不影响其他
2. **新增产品形态**：如 `everalgo-graph-memory`（图记忆）→ 加新 dist，不影响现有
3. **EverAlgo 整体被纳入更大生态**：如 EverOS 决定把 everalgo 收回单 dist 内嵌 → 重新评估（届时 H2 是否仍硬约束）

## 相关 ADR

- [ADR 001 仓库形态](001-multi-repo-vs-monorepo.md) — monorepo 让多 dist 跨包重构成本归零
- [ADR 007 版本兼容策略](007-version-compatibility-strategy.md) — 宽松约束化解 diamond dependency
