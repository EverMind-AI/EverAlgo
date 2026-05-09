# ADR 006: clustering 形态 — 独立工具性子包 `evercore-clustering`

## 状态

✅ **Accepted** — 2026-04-23（v0.12 升级；v0.5 暂不独立的判断作废）

## 背景

EverCore 涉及聚类的链路（基于 memsys_opensource 现状代码核证，2026-04-29）：

- **直接消费 `cluster_id`**：`AgentSkillExtractor.aextract(case, cluster_id=...)` —— 同 cluster 下的多个 `AgentCase` 聚合为一个 `AgentSkill`（agent_skill_extractor.py:427 等多处直接接 `cluster_id: str`）
- **间接消费 `cluster_id`**：`ProfileExtractor.aextract(memcell, cluster_episodes=...)` —— 算子签名不接 `cluster_id`，编排层按 `cluster_id` 反查 `eventid_to_cluster` + fetch memcells 后传 `cluster_episodes` 列表给算子作上下文
- **不消费 cluster**：`AtomicFactExtractor` / `EpisodeExtractor` / `ForesightExtractor` 走独立路径，单条 MemCell 提取，与聚类零关系

**v0.34 论据精确化注**：本 ADR 早期描述"Profile 合并 AtomicFact / AgentSkill 合并 AgentCase 都用聚类，两种策略 centroid + llm_direct"基于 §2.4 早期设计预设；现状代码核证后修正——

1. **Profile 不合并 AtomicFact**：Profile 是按 episode 历史 LLM 增量编辑（add/update/delete），与 AtomicFact 无关
2. **聚类不是"两策略并列"**：现状 `ClusterManager` 是**一个 K-means 状态机 + 两条相似度路径**（共享 centroid 增量更新 + 时间窗口）—— sync 路径用 cosine + 时间窗口；async 路径用 embedding 召回 top-K + 阈值跳过 + LLM 决策。两路径不是独立策略，分两个类需重复 state 管理代码违反 DRY

→ 独立子包决议**仍合理**（论据反而更稳：clustering 是多消费者前置依赖，独立性更强），但候选方案表与论据描述需对齐现状。

**clustering 形态**决定这部分代码住哪里、如何被调用。

相关硬约束：
- **H3** 算法同学迭代速度（聚类策略迭代 / 切换 centroid ↔ llm_direct 要顺手）
- **H5** 跨包紧密联动（聚类是 user_memory 和 agent_memory 共用工具）

## 候选方案

| 方案 | 描述 |
|------|------|
| **A. 独立工具性子包 `evercore-clustering`** | 与 boundary / rank 平级，含 `centroid` + `llm_direct` 两策略；user_memory / agent_memory 在 `pyproject.toml` `dependencies` 里 require `evercore-clustering` |
| B. extractor 内部私有模块 | `agent_memory/_cluster.py` + `user_memory/_cluster.py`（私有命名）各自维护一份；或仅一个有，另一个反向 import |
| C. evercore-core 内部 | 放 `evercore/clustering/` 在 `evercore-core` dist 里 |
| D. 废除聚类（LLM 直判完全替代）| 不要聚类算子，全部走 LLM 判断"是新 skill 还是补充已有"|

## 客观优劣分析

### A. 独立工具性子包 优势

| 维度 | 说明 |
|------|------|
| 算法同学一处迭代 | 改聚类策略只动 `evercore/clustering/` 一个目录 |
| 复用避免重复 | user_memory 和 agent_memory 共享同一份实现，不会出现两份不一致代码 |
| 与 boundary / rank 模式对齐 | 同样是"被多个产品包消费的横切算子"，子包形态对称 |
| 独立 SemVer 演化 | 聚类策略 breaking 改动只 bump `evercore-clustering`，不污染其他 dist |
| 用户可单独按需选装 | 不需聚类的场景可跳过（虽然 user/agent_memory 强制依赖，但场景上独立认知）|

### A. 独立工具性子包 劣势

| 维度 | 说明 |
|------|------|
| 多一个 dist 维护 | dist 数 7 → 8，多一份 `pyproject.toml` / release 流程（**monorepo 下**单仓维护，开销小）|
| 用户多一个依赖关系 | user-memory / agent-memory 自动拉 evercore-clustering，安装链更长 |
| 增加 import 表项 | 算法同学读 user_memory 代码看到 `from evercore.clustering import ...` |

### B. extractor 内部私有模块 优势

| 维度 | 说明 |
|------|------|
| 不暴露顶层 | EverOS 调用拓扑无聚类实体，不暴露顶层 fits "外部契约最小"原则 |
| 实现完全隐藏 | user_memory / agent_memory 各自管自己的聚类，无跨包暴露 |

### B. extractor 内部私有模块 劣势

| 维度 | 说明 |
|------|------|
| **代码重复** | user_memory 和 agent_memory 两份 `_cluster.py`，必有偏差 / 维护不一致 |
| **算法同学迭代痛点** | 改聚类策略要改两处，且哪一处先改的问题协调成本高 |
| **反向依赖坏 pattern** | 如果只一处实现，另一处 import 它，user_memory 和 agent_memory 互相 import 形成耦合（违反产品包业务独立原则）|
| 与 boundary / rank 模式不对齐 | boundary 是公共子包共享，clustering 反而藏起来 |

### C. evercore-core 内部 优势/劣势

新优势：不增加 dist 数量

新劣势：
- core 是基础设施（types / llm / prompts / config / protocols / testing），**不该**含算法实现
- clustering 是算法（centroid 距离计算 + LLM 调用），与 core "纯基础"定位冲突
- core 一次升级 = 所有产品包必跟，违反"独立升级 A 不动 B"（[ADR 002](002-multi-distribution-vs-single.md)）

### D. 废除聚类 优势/劣势

新优势：实现简化、无聚类算子

新劣势：
- LLM 直判每次塞完整 skill 列表 → context 成本超线性增长（千级 skill 时不可行）
- LLM 同输入不同轮次可能输出不同判断 → skill drift（设计文档 A-1 疑点已指出）
- profile 合并相似 atomic_fact 用 LLM 直判同样有 drift 问题
- 这是策略选择问题（centroid vs llm_direct），不是结构问题——只选一个策略 = 失去灵活性

## 对 EverCore 适配度评估

### A 优势对 EverCore 的适配度

| 优势 | 适配度 |
|------|--------|
| 算法同学一处迭代 | ✅ **强需要**（H3）|
| 复用避免重复 | ✅ 强需要（H5 user/agent_memory 共享聚类逻辑）|
| 与 boundary / rank 模式对齐 | ✅ 受益（一致性）|
| 独立 SemVer 演化 | ✅ 受益（[ADR 002](002-multi-distribution-vs-single.md) 同 spirit）|
| 用户按需认知 | ✅ 受益 |

### A 劣势对 EverCore 的适配度

| 劣势 | 适配度 |
|------|--------|
| 多一个 dist 维护 | ⚠️ 可 mitigate（[ADR 001](001-multi-repo-vs-monorepo.md) monorepo 下开销小）|
| 安装链更长 | ⚠️ 不在意（pip 自动拉，用户透明）|
| import 表项 +1 | ⚠️ 不在意（算法同学日常多个 evercore-* import 习以为常）|

### B 劣势对 EverCore 的适配度

| 劣势 | 适配度 |
|------|--------|
| 代码重复 | ❌ **强烈介意**（违反 H3 + H5）|
| 算法同学迭代痛点 | ❌ 强烈介意（H3 直接命中）|
| 反向依赖坏 pattern | ❌ 强烈介意（破坏 user_memory 和 agent_memory 业务独立 [ADR 002](002-multi-distribution-vs-single.md)）|
| 与 boundary / rank 模式不对齐 | ❌ 介意（一致性受损）|

### C / D 评估

C：违反 core "纯基础"定位 + 违反 H2 升级 A 不动 B → 排除

D：是**策略层选择**（centroid 还是 llm_direct），不是结构层问题；保留 A 方案后两种策略都可共存，由 user_memory / agent_memory 各自选用 → 不解决结构问题，且失去灵活性

## 决策

**选 A：独立工具性子包 `evercore-clustering`**。

实施细节：

```
evercore/clustering/
├── __init__.py            # 暴露 cluster() / classify_or_create() 等公开 API
├── centroid.py            # centroid-based 算法（state-in/state-out 接口）
├── llm_direct.py          # LLM 直判语义归并
└── prompts/               # llm_direct 用的 prompt 模板
```

接口约定（state-in / state-out，caller 持久化）：

```python
async def assign(
    item: AtomicFact | AgentCase,
    state: ClusterState,
    strategy: Literal["centroid", "llm_direct"] = "centroid",
) -> tuple[ClusterId, ClusterState]:
    ...
```

distribution 关系：
- `evercore-clustering` dependencies: `evercore-core`
- `evercore-user-memory` dependencies: `evercore-core, evercore-boundary, evercore-clustering`
- `evercore-agent-memory` dependencies: `evercore-core, evercore-boundary, evercore-clustering`

**centroid vs llm_direct 策略选择**留给 user_memory 和 agent_memory 各自决定（profile 合并 / skill 聚合可能选不同），见 design.md §3 T7 议题。

## 行业实证印证

`scikit-learn` 的 `sklearn.cluster` 子包是行业标准：

```python
sklearn/cluster/
├── _kmeans.py / _hierarchical.py / _dbscan.py / _spectral.py / ...
├── __init__.py            # 暴露 KMeans / DBSCAN / 等
```

与 EverCore `evercore.clustering` 同模式——独立子包、多种聚类算法、用户按需选用。

`scikit-learn` 把 `cluster/` 作为**26 个并列顶层子包之一**（design.md §1.2 命名原则实证），EverCore 把 `clustering/` 作为**3 个工具性子包之一**（与 boundary / rank 平级）—— 同样的"算法族独立子包"定位。

## 后续演化触发条件

1. **T7 决议**：若 BOSS 确定"全用 llm_direct 废 centroid"，则 `centroid.py` 删除，`evercore-clustering` 仍作为 dist 存在（只剩 llm_direct）；若决定"全用 centroid 废 llm_direct"反之
2. **聚类算子用户量增长**：第三方算法引入新策略（如 HDBSCAN / ward），加新模块到 `evercore/clustering/`
3. **聚类不再被任何产品包消费**：极端情况若 user_memory / agent_memory 都决定不聚类，`evercore-clustering` 标 deprecated（不影响其他 dist）

## 相关 ADR

- [ADR 002 多 distribution](002-multi-distribution-vs-single.md) — clustering 独立 dist 的依据
- design.md §3 T7（AgentSkill 是否废聚类）— 策略层取舍，与本 ADR 结构层选择正交
