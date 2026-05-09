# ADR 005: testing 暴露形态 — 公开子包 `evercore.testing`，assertions + fake_llm 两件套

## 状态

✅ **Accepted** — 2026-04-23（v0.7 矫正 v0.6 模糊"fake_llm + fixtures"为"assertions + fake_llm"）

## 背景

EverCore 是算法库，下游消费方（EverOS 集成测、算法同学单测、第三方）都需要写测试调用 EverCore。**testing 暴露形态**决定 EverCore 给下游写测试提供哪些辅助、如何打包发布。

相关硬约束：
- **H3** 算法同学迭代速度（写单测要顺手）
- 算法库 vs SDK 分类（[ADR 008](008-re-export-vs-client-facade.md)）：EverCore 是算法库

## 候选方案

| 方案 | 描述 |
|------|------|
| **A. 公开子包 `evercore.testing`，assertions + fake_llm 两件套** | `evercore-core` dist 内含 `evercore/testing/` 子包；`assertions.py` 提供领域专用断言（`assert_memcell_equal` 等），`fake_llm.py` 提供可编程 fake LLM 客户端；下游 `from evercore.testing import ...` 直接用 |
| B. 同 A 但加 `fixtures/` 子目录 | A + `evercore/testing/fixtures/` 含预设样本（sample_messages / sample_memcell 等）|
| C. 独立 distribution `evercore-testing` | 拆出 testing 成独立 dist，dev-only 安装 |
| D. 不暴露 testing 子包 | 下游自己 mock，不提供官方测试辅助 |

## 客观优劣分析

### A. 公开 `evercore.testing`，两件套 优势

| 维度 | 说明 |
|------|------|
| 下游写单测低门槛 | `from evercore.testing import fake_llm` 一行即用 |
| 与算法库主流模式对位 | numpy / pandas / pytorch 都是 testing 子包内含 assertions（行业实证） |
| 与 core 同节奏演化 | core 加新数据类型时直接在 `assertions.py` 加对应断言函数 |
| 安装零成本 | 与 core 一并装，用户不需额外 dep 声明 |

### A. 公开 `evercore.testing`，两件套 劣势

| 维度 | 说明 |
|------|------|
| `evercore-core` dist 体积 +少量 | testing 子包代码增加 dist 体积（实际几 KB，可忽略）|
| 生产环境也装了 testing | 用户生产部署时 `evercore-core` 内含 testing 子包（实际不会 import 不影响）|

### B. 加 fixtures 优势/劣势

新优势：预设样本省去构造样板

新劣势：
- **主流算法库无此实证**：numpy.testing / pandas.testing / torch.testing / langchain-tests **都不放预设样本**
- "fixtures" 命名是 pytest 文化术语，不是 testing 子包标准命名（pytest 自己用 `conftest.py` 管 fixture）
- 预设样本与领域细节耦合，更新频率高 → 污染 testing 子包稳定性

### C. 独立 dist `evercore-testing` 优势/劣势

新优势：testing 与 core 版本独立演化

新劣势：
- 与 core 紧密绑定（testing 依赖 core 的数据类型）→ 独立 dist 必然要求 testing pin core 版本，反而带来 diamond dependency 风险
- 主流算法库无此实证：numpy / pandas / pytorch 全部并入 core，无独立 testing dist
- 用户多一个 `pip install evercore-testing` 步骤，门槛上升

### D. 不暴露 testing 优势/劣势

优势：dist 最小

劣势：
- 下游写单测要自己 mock LLM、自己写断言函数，重复劳动 + 不一致
- 算法库形态下与社区主流相悖

## 对 EverCore 适配度评估

### A 优势对 EverCore 的适配度

| 优势 | 适配度 |
|------|--------|
| 下游单测低门槛 | ✅ **强需要**（H3 算法同学迭代速度；EverOS 集成测也受益）|
| 与主流算法库模式对位 | ✅ 受益（行业一致性，新人无学习成本）|
| 与 core 同节奏演化 | ✅ 受益（演化阶段同步更新）|
| 安装零成本 | ✅ 受益 |

### A 劣势对 EverCore 的适配度

| 劣势 | 适配度 |
|------|--------|
| dist 体积 +少量 | ⚠️ 不在意（几 KB） |
| 生产环境也装了 testing | ⚠️ 不在意（不 import 不影响）|

### B 评估（加 fixtures）

新优势"预设样本省样板"——⚠️ 可 mitigate（算法同学自己组织 fixture 更贴合具体测试场景）

新劣势"无主流实证 + 命名争议 + 稳定性污染"——❌ 介意（违反 §1.2 命名规范溯源 + 演化压力）

**不选 B**——预设样本若未来必要，独立 `evercore.examples` 子包按 sklearn.datasets 模式承接，不混入 testing。

### C 评估（独立 dist）

新劣势"diamond risk + 主流无实证 + 用户门槛"——❌ 强烈介意

**不选 C**

### D 评估（不暴露）

新劣势"重复劳动 + 与主流相悖"——❌ 强烈介意（违反 H3）

**不选 D**

## 决策

**选 A：公开子包 `evercore.testing`，assertions + fake_llm 两件套**。

`evercore.testing/` 内容：

| 子模块 | 内容 |
|--------|------|
| `evercore.testing.assertions` | `assert_memcell_equal` / `assert_episode_equal` / `assert_rank_output_close(top_k, atol)` 等针对 evercore 数据类型的断言函数 |
| `evercore.testing.fake_llm` | 可编程 fake LLM 客户端（按 scene / prompt 模板返回预设响应），下游 monkeypatch 替换真实 LLM |

**显式不放**：
- 预设样本数据（fixtures / samples）
- pytest plugin / conftest 共享

未来若需预设样本，独立 `evercore.examples` 子包（仿 `sklearn.datasets`），不混入 testing。

## 行业实证印证

主流算法库 100% 公开 testing 子包内容均为 "纯断言 + 测试工具"（WebFetch 2026-04-23 核验）：

| 库 | testing 子包内容 | 公开 API 形态 |
|----|----------------|--------------|
| **numpy** | `assert_allclose` / `assert_array_equal` | `numpy.testing` |
| **pandas** | `assert_frame_equal` / `assert_series_equal` | `pandas.testing` |
| **pytorch** | `assert_close` / `make_tensor` / `FileCheck` | `torch.testing` |
| **scikit-learn** | `check_estimator`（给第三方估算器）| `sklearn.utils.estimator_checks` |
| **langchain-tests** | 测试基础设施 / utils | 独立 `langchain-tests` dist（langchain 是少数选 C 的，但 EverCore 适配度低）|

**关键**：numpy / pandas / pytorch / sklearn **都不放预设样本到 testing 子包**。预设样本独立 namespace（如 `sklearn.datasets`）是更主流模式。

EverCore 选 A + 拒绝 fixtures，与 numpy / pandas / pytorch 路线一致。

## 后续演化触发条件

1. **下游反馈需要预设样本**（EverOS 集成测想直接拿 sample_memcell 跑）→ 独立 `evercore.examples` 子包，参照 sklearn.datasets，**不**回头加到 testing
2. **fake_llm 复杂度膨胀**（多 provider mock / streaming mock / batch mock）→ 拆 `evercore.testing.fake_llm/` 子目录而非单文件
3. **第三方写 evercore extension 时需要更复杂测试基础设施**（如 langchain-tests `standard-tests` 那种）→ 重新评估 C 独立 dist 方案

## 相关 ADR

- [ADR 002 多 distribution](002-multi-distribution-vs-single.md) — testing 并入 evercore-core 的依据
- [ADR 008 算法库 vs SDK 分类](008-re-export-vs-client-facade.md) — 算法库需要的 testing 工具与 SDK 不同
