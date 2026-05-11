# 📐 API Reference（占位）

EverAlgo 对外接口定义的归属位置。**当前未填充**——理由见下方"写作时机"。

## 应包含的 4 类内容

| 文件 | 内容 | 写法 |
|------|------|------|
| `data-contracts.md` | 核心 schema 字段表：`MemCell / Episode / Foresight / AtomicFact / Profile / AgentCase / AgentSkill / RankInput / RankOutput / ParsedContent / KnowledgeMemory`（每字段：类型、必填、语义、约束、示例值） | **手写**（工程算法两侧对齐表）|
| `api/`（目录） | 每个算子的精确函数签名 + 参数说明 + 返回值 + raises（`Extractor.aextract` / `Ranker.arank` / `cluster_by_*` / `LLMClient.chat` 等）| **自动生成**（mkdocstrings 或 Sphinx autodoc 抽 docstring + type hints）|
| `exceptions.md` | `LLMError` 7 子类语义 + 错误码 + caller 处置建议 | **手写**（稳定信息）|
| `configuration.md` | `LLMConfig` 字段 + `EVERALGO_LLM_*` env 映射规则 | **手写** |

## 写作纪律

- **API 签名 100% 自动生成**，不手写——明星项目（sklearn / pytorch / HuggingFace / FastAPI）实证：手写必腐烂
- **数据契约手写字段表**，但要与代码 dataclass / Pydantic model 字段一一对应；schema 改动 → 文档同步改
- **每个字段配示例值**，工程同学和算法同学都能在不读代码的前提下对齐
- **不写算子使用教程** —— 教程归 `getting-started/`，操作配方归 `how-to/`，本目录只放参考信息

## 写作时机（为什么现在还不写）

| 触发条件 | 应建文件 |
|---------|---------|
| 数据契约 schema 拍板（即设计稿 T1 完成，11 个核心 schema 字段定稿）| `data-contracts.md` |
| 首批算子代码落地 + docstring 完善（约 v0.5 阶段）| `api/`（同时引入 mkdocstrings 工具链）|
| `LLMError` 7 子类定稿 | `exceptions.md` |
| `LLMConfig` 完整字段表定稿 | `configuration.md` |

**当前阶段**：8 个 distribution 仅有空 `__init__.py` 脚手架，无任何 dataclass / Protocol / 函数签名落地；schema 仍在评审中。**此时写 reference 文档等于猜测**，1-2 周内必被推翻并造成双份维护。

待 schema 拍板 + 首批算子代码落地后再填充本目录。
