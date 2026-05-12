# EverAlgo Architecture Decision Records (ADR)

本目录收录 EverAlgo 设计中的关键架构决策。每个 ADR 独立讨论一个决策，包含完整的优劣分析、对 EverAlgo 的适配度评估、决策结论与行业实证印证。

主设计文档 [`../design.md`](../design.md) 中的"设计自检"条目都链接到这里的对应 ADR 取详细论据。

## ADR 索引

| 编号 | 标题 | 状态 | 主决策 |
|------|------|------|--------|
| [001](001-multi-repo-vs-monorepo.md) | 仓库形态 | ✅ Accepted | monorepo + uv workspace |
| [002](002-multi-distribution-vs-single.md) | 发布粒度 | ✅ Accepted | 多 distribution + 独立 SemVer |
| [003](003-namespace-package-pep420.md) | namespace 模式 | ✅ Accepted | PEP 420 共享 `everalgo.*` namespace（B 风格）|
| [004](004-providers-nested-in-llm.md) | LLM provider 物理位置 | ✅ Accepted | 内嵌于 `llm/providers/` 子包 |
| [005](005-testing-as-public-subpackage.md) | testing 暴露形态 | ✅ Accepted | 公开子包 `everalgo.testing`，`assertions + fake_llm` 两件套 |
| [006](006-clustering-independent-subpackage.md) | clustering 形态 | ✅ Accepted | 独立工具性子包 `everalgo-clustering` |
| [007](007-version-compatibility-strategy.md) | 版本兼容策略 | ✅ Accepted | 宽松约束 + 严守 SemVer + 不强制同步 release |
| [008](008-re-export-vs-client-facade.md) | facade 实现模式 | ✅ Accepted | re-export 式 facade（非 Client 类）|
| [009](009-naming-convention-llama-index-style.md) | 命名规范 | ✅ Accepted | dash distribution + 共享 dot namespace + lowercase 模块 |
| [010](010-sync-async-dual-interface.md) | I/O 算子接口 | ✅ Accepted | sync + async 双接口（`a*` 前缀），明星 AI 库 100% 模式 |
| [011](011-protocol-vs-abc.md) | 接口约定 | ✅ Accepted | Protocol structural over ABC（无状态 → DSPy/LlamaIndex 插件接口同模式）|
| [012](012-llm-stack-architecture.md) | LLM 抽象层架构 | ✅ Accepted | 各家原生 SDK + Protocol + 双层路由（Scene 出 EverAlgo / Provider 在 EverAlgo） + LLMError 7 子类混合多重继承 + 算法层不加 retry |

## ADR 写作模板

每个 ADR 用统一 6 段结构：

1. **状态**：Accepted / Superseded / Deprecated + 决议日期
2. **背景**：决策面对的问题、约束、相关 ADR
3. **候选方案**：列出全部候选（不预设倾向）
4. **客观优劣分析**：每个方案的普世优劣（不带 EverAlgo 滤镜）
5. **对 EverAlgo 适配度评估**：逐条优势/劣势对 EverAlgo 的影响（强需要 / 不在意 / 强烈介意 / 可 mitigate / 用不上）
6. **决策与行业实证印证**：综合结论 + 同推理路径下行业实证作为旁证

## 论证哲学

- **客观优劣不挑论据**——必须列两边完整优劣
- **适配度逐条评估**——对每条优势"用得上吗" + 每条劣势"在意吗"独立判断
- **行业实证只作辅证**——用以说明同推理路径下别人也得到相同结论，**不作为主导论据**（避免 cargo cult）
- **追溯 EverAlgo 硬约束**：H1 evermem 文档契约 / H2 升级 A 不动 B / H3 算法同学迭代速度 / H4 无状态接口 / H5 跨包紧密联动 / H6 v0.x 演化阶段 / H7 可识别系列前缀

## 何时新增 ADR

- 新增重大架构决策
- 推翻现有决策（旧 ADR 改 Superseded，新 ADR 引用旧）
- 决策范围扩展（旧 ADR 状态保持，新 ADR 在补充范围内）

不必每个微决策都写 ADR——只对**对未来读者会有"为什么不那样"疑问的决策**写。
