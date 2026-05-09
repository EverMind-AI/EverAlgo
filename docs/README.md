# 📚 EverCore 文档

EverCore 是 **EverOS（AI 记忆管理系统）** 依赖的算法库，无状态，提供 Extract / Rank 双主轴算法 IP；EverOS 负责所有工程载体职责（API / 持久化 / 编排 / scene 路由 / 记忆生命周期）。文档按读者目标分区。

## 🗺️ 读者地图

### 我是新同学，想快速搞懂 EverCore

→ 从 [`concepts/architecture.md`](concepts/architecture.md) 开始（约 30 分钟读完）

涵盖：分层、定位、双主轴、子包结构、命名契约、调用形态、状态边界。每个抽象都配可运行代码示例，每个 why 都链到对应 ADR，按需深入。

### 我是算法同学，要新增或修改算子

→ 先读 [`concepts/architecture.md`](concepts/architecture.md) 第 4-7 节（子包 / 命名 / 调用 / 状态边界）

→ 再翻 `how-to/`（待补）：`add-extractor.md` / `add-ranker.md` / `customize-prompt.md`

→ 决策溯源在 [`decisions/`](decisions/) 12 篇 ADR

### 我是 EverOS 工程同学，要调用 EverCore

→ `getting-started/01-installation.md`（待补）+ `02-first-extraction.md`（待补）

→ API 精确签名查 `reference/`（待补，自动生成）

→ 调用形态全景见 [`concepts/architecture.md`](concepts/architecture.md) §6

### 我是框架设计者 / 想了解某个决策为什么这样定

→ 决策记录 [`decisions/`](decisions/)（12 篇 ADR，每篇含候选方案 + 行业实证 + 适配度评估）

ADR 是 EverCore 设计决策的单一事实来源。`concepts/architecture.md` 中每个 why 都链到对应 ADR。

---

## 📂 目录结构

```
docs/
├── README.md                    # 本文件，文档入口 + 读者地图
│
├── concepts/                    # 概念解释（理解导向）
│   └── architecture.md          # ★ 架构总览（onboarding 抓手）
│
└── decisions/                   # 架构决策记录（ADR）
    ├── README.md                # 12 篇 ADR 索引
    └── 001-…012-…
```

已建占位（含写作时机说明）：

```
└── reference/                   # API 参考（占位，schema 定稿后填充）
    └── README.md
```

待补区：

```
├── getting-started/             # 入门教程（学习导向）
└── how-to/                      # 操作指南（任务导向）
```

---

## ✍️ 写作纪律（给文档贡献者）

文档体系按 [Diátaxis](https://diataxis.fr) 四象限分区——**读者目标决定写法**，不要混写：

| 象限 | 读者目标 | 写作要点 | 不要做 |
|------|---------|---------|-------|
| **Tutorials**（`getting-started/`）| 学习 | 一步步带，承诺成功 | 不解释 why、不展开理论 |
| **How-to**（`how-to/`）| 任务 | 配方式，可拷贝粘贴 | 不展开理论 |
| **Reference**（`reference/`）| 查找 | 完整、严谨、机器可生成 | 不写教程式叙述 |
| **Concepts**（`concepts/`）| 理解 | 写 why 和原理 | 不写 step-by-step |

附加纪律：

- **每个 why 链到 ADR，不在概念文档展开**——避免双份维护
- **每个抽象配可运行代码片段**（pytorch notes 同款）
- **设计评审稿冻结，概念文档跟代码改**——避免决策稿与产品文档双份过期
- **`concepts/architecture.md` 硬上限 600 行**——超出说明在写 why，应砍掉链接到 ADR
