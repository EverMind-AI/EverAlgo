# ADR 001: 仓库形态 — monorepo + uv workspace

## 状态

✅ **Accepted** — 2026-04-23（v0.11 反转 v0.9 multi-repo 决策；v0.12 适配 8 distribution）

## 背景

EverCore 拆为 8 个独立 PyPI distribution（`evercore-core` + 4 产品性 + 3 工具性，详见 [ADR 002](002-multi-distribution-vs-single.md)）。**仓库形态**是另一独立维度——这 8 个 distribution 应该放在 1 个 git 仓还是 8 个 git 仓？

相关硬约束：
- **H3** 算法同学迭代速度（§1.1 核心定位）
- **H5** 跨包紧密联动（core / boundary / clustering 改动频繁影响产品包）
- **H6** v0.x 演化阶段（类型契约 / Protocol 仍在快速演化）
- **H2** 用户独立升级 A 不动 B（与仓库形态正交，由发布粒度保证）

## 候选方案

| 方案 | 描述 | 代表项目 |
|------|------|---------|
| **A. multi-repo** | 8 个独立 git 仓库（`<gitlab>/<group>/evercore-core` / `evercore-parser` / ...）| HuggingFace 全家桶（transformers / datasets / accelerate / huggingface_hub 4 仓）|
| **B. monorepo + uv workspace** | 单一 git 仓库 `<gitlab>/<group>/evercore`，仓内 `packages/evercore-*/` 各包独立 `pyproject.toml` | LangChain `libs/{core, partners/*, ...}/` / LlamaIndex 顶层 `llama-index-*/` 多目录 / Apache Airflow `providers/` |
| C. monorepo 单 distribution | 单仓单 dist + extras（v0.5 旧方案）| numpy / pandas |

C 已被 [ADR 002](002-multi-distribution-vs-single.md) 排除（违反 H2 升级 A 不动 B），本 ADR 只比 A vs B。

## 客观优劣分析（普世，不带 EverCore 滤镜）

### multi-repo 优势

| 维度 | 说明 |
|------|------|
| 物理隔离强 | 仓权限误改无法跨仓污染 |
| 单仓 clone 体积小 | 只拉所需包，磁盘 / 带宽节省 |
| 不同 release cadence | 每仓独立 maintainer / 独立 release 节奏 |
| CI 资源天然隔离 | 改 A 仓只触发 A 仓 pipeline |
| 对外贡献门槛低 | fork 单仓即可贡献 |
| Issue 责任归属清晰 | bug 报告自然落到具体仓 |

### multi-repo 劣势

| 维度 | 说明 |
|------|------|
| 跨包重构破坏原子性 | 跨包改动需多仓 PR 同步合并，无法保证一致性 |
| 跨包联调繁琐 | sibling editable install 路径协调复杂；行业无标准最佳实践（见下文 HuggingFace CONTRIBUTING 实证）|
| 共享配置重复维护 | 8 套 lint / pre-commit / CI 模板 |
| 新人上手成本高 | N 仓 clone + N 个 README 并行阅读 |
| 跨仓 issue 关联弱 | 影响多包的 bug 难以归并讨论 |
| 全仓 grep / refactor 工具失效 | IDE 跨仓搜索 / 重构受限 |

### monorepo + uv workspace 优势

| 维度 | 说明 |
|------|------|
| 跨包重构原子性 | 单 commit 可跨 N 包同步修改，类型契约演化保证一致 |
| 工具链 / 共享配置一处管理 | lint / pre-commit / CI 模板统一 |
| 跨包导航 / grep / refactor 强 | IDE 全仓搜索 / 重构原生支持 |
| 新人上手快 | 1 仓 clone，1 个 README 入口 |
| workspace 工具一键 sync | `uv sync --all-packages` 装齐所有包 editable install 到共享 venv |
| 跨包 issue 自然关联 | 单 issue tracker，跨包 bug 易归并 |

### monorepo + uv workspace 劣势

| 维度 | 说明 |
|------|------|
| 单仓体积大 | 含全部包代码 + lockfile |
| 物理隔离弱 | 仓权限污染所有包 |
| CI 资源开销需配置 | 必须 path-based trigger 才隔离 |
| 不同 maintainer / cadence 难协调 | 同 release 节奏假设 |
| 对外贡献门槛高 | 必须克隆全仓 |

## 对 EverCore 适配度评估

### multi-repo 优势对 EverCore 的适配度

| 优势 | 适配度 | 理由 |
|------|--------|------|
| 物理隔离强 | ⚠️ **用不上** | EverCore 同算法团队内部维护，无误改恶意 |
| 单仓 clone 体积小 | ⚠️ 不在意 | 算法库代码本就轻量（文本 + prompts，数十 MB），8 包合一也不大 |
| 不同 release cadence | ⚠️ 用不上 | 8 包同算法团队、同节奏维护（H5 紧密联动） |
| CI 资源天然隔离 | ⚠️ 可 mitigate | GitLab `rules:changes:` path-based trigger 等价，一次配置长期 0 维护 |
| 对外贡献门槛低 | ⚠️ 用不上 | 内部项目（公司 GitLab），无第三方贡献者预期 |
| Issue 责任归属清晰 | ⚠️ 不在意 | 单 issue tracker 用 label / component 字段同样能区分 |

### multi-repo 劣势对 EverCore 的适配度

| 劣势 | 适配度 | 理由 |
|------|--------|------|
| 跨包重构破坏原子性 | ❌ **强烈介意** | H6 v0.x 演化阶段 + H5 紧密联动 → 跨包重构高频 |
| 跨包联调繁琐 | ❌ **强烈介意** | H3 算法同学迭代速度核心；行业无标准最佳实践（HuggingFace 三家 CONTRIBUTING 都没文档化此事，因为他们 4 包业务独立才不需要）|
| 共享配置重复维护 | ❌ 介意 | 同算法团队同节奏，8 套配置是浪费 |
| 新人上手成本高 | ❌ 介意 | 算法同学是核心读者（H3）|
| 跨仓 issue 关联弱 | ⚠️ 不在意 | 内部项目，issue 量级不大 |
| 全仓 grep / refactor 失效 | ❌ 介意 | H6 演化阶段需要频繁跨包重构 |

### monorepo 优势对 EverCore 的适配度

| 优势 | 适配度 | 理由 |
|------|--------|------|
| 跨包重构原子性 | ✅ **强需要** | H6 v0.x 演化阶段 + H5 紧密联动 |
| 工具链 / 共享配置一处管理 | ✅ 强需要 | 同算法团队节奏，统一配置低维护成本 |
| 跨包导航 / grep / refactor 强 | ✅ 强需要 | H6 演化阶段日常需要 |
| 新人上手快 | ✅ 强需要 | H3 算法同学核心读者 |
| workspace 一键 sync | ✅ 强需要 | H3 算法同学迭代速度 |
| 跨包 issue 自然关联 | ✅ 受益 | 跨包 bug 归并讨论方便 |

### monorepo 劣势对 EverCore 的适配度

| 劣势 | 适配度 | 理由 |
|------|--------|------|
| 单仓体积大 | ⚠️ 不在意 | 算法库轻量 |
| 物理隔离弱 | ⚠️ 不在意 | 同团队信任内部 |
| CI 资源开销需配置 | ⚠️ 可 mitigate | GitLab `rules:changes:` 一次配置成本 |
| 不同 maintainer / cadence 难协调 | ⚠️ 用不上 | 8 包同节奏（H5）|
| 对外贡献门槛高 | ⚠️ 用不上 | 内部项目 |

## 决策

**选 monorepo + uv workspace**。

逐条统计：
- monorepo 优势 EverCore **强需要 6 条 + 受益 1 条**
- monorepo 劣势 EverCore **不在意 / 用不上 / 可 mitigate 全部 5 条**
- multi-repo 优势 EverCore **用不上 / 不在意 5 条 + 可 mitigate 1 条**
- multi-repo 劣势 EverCore **强烈介意 2 条 + 介意 3 条**

monorepo 完胜——所有强需要 EverCore 都获得，所有劣势都不在意或可 mitigate；multi-repo 优势 EverCore 几乎都用不上，劣势却强烈介意。

## 实施细节

仓内目录结构：

```
<gitlab>/<group>/evercore/
├── pyproject.toml              # workspace 根（uv workspace 配置）
├── uv.lock                     # 整 workspace 共享 lockfile
├── .gitlab-ci.yml              # 单仓 CI（path-based trigger）
├── packages/
│   ├── evercore-core/          # 各包独立 pyproject.toml + 独立 SemVer
│   ├── evercore-parser/
│   ├── evercore-boundary/
│   ├── evercore-rank/
│   ├── evercore-clustering/
│   ├── evercore-user-memory/
│   ├── evercore-agent-memory/
│   └── evercore-knowledge/
└── docs/
```

每个 `packages/evercore-*/` 子目录独立含 `pyproject.toml`、独立 SemVer、独立 PyPI 发布——**仓库形态与发布粒度正交**。"升级 A 不动 B"（H2）由 PyPI 端 dist 独立保证，与仓内 layout 无关。

## 行业实证印证

同推理路径下，AI 圈业务结构同构（core + N 紧密联动包）的项目都收敛到 monorepo：

| 项目 | 业务结构 | 仓库形态 | 文档实证 |
|------|---------|---------|---------|
| **LangChain** | `langchain-core` + 60+ partners | monorepo `libs/` | github.com/langchain-ai/langchain/tree/master/libs |
| **LlamaIndex** | `llama-index-core` + 150+ integrations | monorepo 顶层多包 | github.com/run-llama/llama_index |
| **Apache Airflow** | airflow core + 60+ providers | monorepo + uv workspace | github.com/apache/airflow（CONTRIBUTING 明确推荐 `uv sync --all-packages`）|
| **Dagster** | dagster + 60+ integrations | monorepo `python_modules/` | github.com/dagster-io/dagster |
| **Prefect** | prefect + 18+ integrations | monorepo `src/integrations/` | github.com/PrefectHQ/prefect/tree/main/src/integrations |
| HuggingFace | 4 包**业务独立** | multi-repo | 不构成本论证反例——业务独立 ≠ 紧密联动 |

**关键澄清**：LangChain / LlamaIndex 选 monorepo **不是因为"潮流"，而是同样的需求驱动**。HuggingFace 选 multi-repo 是因为 transformers / datasets / accelerate 业务独立、跨包改动天然低频——根本不同推理路径，不构成反例。

## 后续演化触发条件

何时需要重新评估本决策：

1. **EverCore 1.0 进入维护期**：v0.x → 1.x 后跨包重构频率显著下降，monorepo 跨包原子性优势可能不再核心；但其他维度（新人上手 / 工具链）仍倾向 monorepo
2. **业务上某包独立成产品**：如 `evercore-rank` 独立成检索产品被多团队消费、节奏与 EverCore 主线脱钩，可能要拆出去独立仓
3. **算法库膨胀超过 1GB**：如自带大模型 / 重 fixture，`git clone` 体验受损时再评估

## 相关 ADR

- [ADR 002 多 distribution + 独立 SemVer](002-multi-distribution-vs-single.md) — 发布粒度（与仓库形态正交）
- [ADR 007 版本兼容策略](007-version-compatibility-strategy.md) — 宽松 SemVer 约束在仓库形态下的兼容性保证
