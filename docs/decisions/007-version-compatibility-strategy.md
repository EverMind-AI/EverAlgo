# ADR 007: 版本兼容策略 — 宽松约束 + 严守 SemVer + 不强制同步 release

## 状态

✅ **Accepted** — 2026-04-23（v0.9 制定，v0.10 加 SemVer 前置定义）

## 背景

[ADR 002](002-multi-distribution-vs-single.md) 决定多 distribution 后，多个下游 dist 都依赖 `evercore-core`。当不同下游对 core 的版本约束不一致时，可能触发 **diamond dependency**（依赖地狱）：

```
evercore-user-memory 0.5 要 evercore-core>=0.1,<0.2
evercore-agent-memory 0.6 要 evercore-core>=0.2,<0.3
→ pip install 两者 → 解析失败
```

需要**版本兼容策略**避免此问题。

相关硬约束：
- **H2** 用户独立升级 A 不动 B（diamond 冲突会强制用户升级整套）
- 不应增加用户认知负担

## 前置概念：SemVer

[Semantic Versioning](https://semver.org)（语义化版本）：版本号 `MAJOR.MINOR.PATCH`：
- **MAJOR**：不兼容变更（breaking）
- **MINOR**：向后兼容的新功能
- **PATCH**：向后兼容的 bug 修复
- `0.x.x` 视作不稳定，任何变更可能 breaking
- `1.0.0` 是稳定 API 承诺起点

## 候选方案

| 方案 | 描述 |
|------|------|
| **A. 紧约束**（v0.5 草案误入） | 下游每个 dist 在 `pyproject.toml` 写 `evercore-core>=0.1,<0.2`（每个 minor 段独立锁定）|
| **B. 宽松约束 + 严守 SemVer + 不强制同步**（HuggingFace 模式）| 下游写 `evercore-core>=0.1.0,<2.0.0`（跨整 major 段）；core 严守 SemVer 兑现兼容承诺；各 dist 独立 release 不强制同步 |
| C. monorepo 强制同步 release（langchain 风部分实现）| 每次 core bump 时所有 dist 同步发新版 + 同步约束 |
| D. core 不演化 / 长期 frozen | 避免 breaking 但 EverCore v0.x 演化阶段不可行 |

## 客观优劣分析

### A. 紧约束 优势

| 维度 | 说明 |
|------|------|
| 版本兼容性可见 | pin upper bound 强制版本一致 |
| 下游意外升级 core 风险低 | 紧约束阻止跨段安装 |

### A. 紧约束 劣势

| 维度 | 说明 |
|------|------|
| **diamond 频发** | 任两下游对 core 的紧约束区间不重叠时立即冲突 |
| **每次 core bump 需所有下游更新约束** | 否则用户装不到一起 |
| 与 H2 矛盾 | 用户升级 user-memory 时若 core 跨了 minor 段，agent-memory 必须同时升 |

### B. 宽松约束 + 严守 SemVer + 不强制同步 优势

| 维度 | 说明 |
|------|------|
| **diamond 自然消解** | 跨整 major 段约束区间天然重叠，pip 总能找到公共版本 |
| 各下游独立演化 | 不强制同步 release，符合 H2 |
| 与主流 AI 多 dist 项目对齐 | HuggingFace / llama-index 都是这个模式 |
| 用户使用透明 | pip 自动 resolve，无认知负担 |

### B. 宽松约束 + 严守 SemVer + 不强制同步 劣势

| 维度 | 说明 |
|------|------|
| **依赖 core 守 SemVer 承诺** | core 一旦在 minor / patch 不小心 breaking，下游全栈出问题 |
| 0.x 阶段心智负担 | SemVer 0.x 视作不稳定，但 EverCore 仍要"实际尽量保持兼容"，是隐式承诺 |
| upper bound 选择有点 arbitrary | 写 `<2.0` 还是 `<3.0` 还是无 upper，需团队约定 |

### C. monorepo 强制同步 release 优势/劣势

新优势：版本一致性最强

新劣势：
- 直接违反 H2 "升级 A 不动 B"——同步 release 把所有 dist 锁定到同一节奏
- 跨包 release 工作量大，每次变更都要全套发版

### D. core 不演化

劣势：违反 H6 v0.x 演化阶段（类型契约 / Protocol 仍在快速变）

## 对 EverCore 适配度评估

### A 劣势对 EverCore 的适配度

| 劣势 | 适配度 |
|------|--------|
| diamond 频发 | ❌ **强烈介意**（用户体验受损） |
| 每次 core bump 全栈跟更新 | ❌ 强烈介意（H2 反向） |
| 与 H2 矛盾 | ❌ **致命** |

### B 优势对 EverCore 的适配度

| 优势 | 适配度 |
|------|--------|
| diamond 自然消解 | ✅ **强需要**（H2 直接命中）|
| 各下游独立演化 | ✅ 强需要（H2）|
| 与主流模式对齐 | ✅ 受益 |
| 用户使用透明 | ✅ 强需要（用户认知零负担）|

### B 劣势对 EverCore 的适配度

| 劣势 | 适配度 |
|------|--------|
| 依赖 core 守 SemVer | ⚠️ 可 mitigate（团队约定 + CI 兼容性测试 + 严守纪律）|
| 0.x 心智负担 | ⚠️ 可 mitigate（文档明示"0.x 阶段尽量兼容、breaking 集中偶数 minor"）|
| upper bound arbitrary | ⚠️ 可 mitigate（约定 `<2.0` 跨整 major，与 HuggingFace transformers 对 hub 写法对齐）|

### C / D 评估

C：直接违反 H2 → 排除

D：违反 H6 → 排除

## 决策

**选 B：宽松约束 + 严守 SemVer + 不强制同步 release（HuggingFace 模式）**。

### 4 条具体策略

1. **核心兄弟包之间不互相依赖**——产品性 4 包（user_memory / agent_memory / knowledge / parser）**互相不在 `install_requires`**（仿 HuggingFace transformers / datasets / accelerate 互不在 install_requires 互依赖）
2. **下游对 core 用宽松约束**：`evercore-core>=X.Y,<2.0`（仿 transformers 对 hub）或 `>=X.Y`（仿 accelerate 对 hub）
3. **`evercore-core` 严守 SemVer**：minor + patch 必须向后兼容；breaking 集中 major bump（0.x 阶段尽量兼容、breaking 集中偶数 minor）
4. **不强制同步 release**：每个 distribution 独立演进

### 约束写法约定

```toml
# evercore-user-memory pyproject.toml
dependencies = [
  "evercore-core>=0.1.0,<2.0.0",        # 跨整 major 段
  "evercore-boundary>=0.1.0,<2.0.0",
  "evercore-clustering>=0.1.0,<2.0.0",
]
```

### 用户侧 lockfile 兜底

用 `uv lock` / `poetry lock` 锁一组兼容版本到 `uv.lock`；升级时局部 `uv lock --upgrade-package evercore-user-memory`。

## 行业实证印证

HuggingFace 全家桶版本约束实证（WebFetch 2026-04-23 核验 4 个 setup.py）：

| 包 | 对 huggingface_hub 约束 | 兄弟包间互依赖 |
|----|---------------------|-----------------|
| **transformers** | `huggingface-hub>=1.5.0,<2.0` | datasets / accelerate **不在 install_requires**（仅 extras）|
| **datasets** | `huggingface-hub>=0.25.0,<2.0` | 不依赖 transformers / accelerate |
| **accelerate** | `huggingface_hub>=0.21.0`（**无 upper**）| 不依赖 transformers / datasets |
| **huggingface_hub** | — | — |

HuggingFace 千万级用户量级实证此模式有效——4 包独立演化、跨大量版本组合无 diamond 冲突。

EverCore 套用此模式：`evercore-core` 担当 EverCore 中的 `huggingface_hub` 角色（共同 base），产品性兄弟包不互依赖。

## 后续演化触发条件

1. **EverCore 1.0.0 发布**：进入稳定期后约束写法不变（仍 `<2.0`），但实质风险降低（minor/patch 严格兼容）
2. **某次 core breaking 影响多个下游**：触发版本兼容性测试机制（CI 跑跨 core 版本的下游兼容矩阵）
3. **用户报告 diamond 实际发生**：检查是否某下游 over-pinned；调整约束写法

## 相关 ADR

- [ADR 002 多 distribution](002-multi-distribution-vs-single.md) — diamond dependency 是多 dist 模式的内生风险
- [ADR 001 monorepo](001-multi-repo-vs-monorepo.md) — monorepo 让 core 演化时跨 dist 同步约束更新成本低（不强制同步 release，但**有需要时**便于一次 MR 同步多 dist）
