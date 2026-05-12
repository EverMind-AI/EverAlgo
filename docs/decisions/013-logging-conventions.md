# ADR 013: 日志规范 — Library logging 契约 + 默认安全过滤 + ruff 规则强制

## 状态

✅ **Accepted** — 2026-05-12

## 背景

EverAlgo 跨两种性质：
- **LLM I/O 路径**（boundary / clustering / parser / user_memory / agent_memory / knowledge 的 LLM 调用以及 `everalgo.llm` 自身）— 需要诊断日志：provider 路由、token 计数、retry、fallback
- **纯算法路径**（`rank.fusion.rrf`、`boundary._tokenize.count_tokens`、`clustering._algorithm.cluster_by_geometry` 中的几何距离）— 不调网络 I/O，无日志价值

同时存在 3 个硬约束：
1. **secret 泄露风险**：LLM provider SDK 在 `DEBUG` 级别会 log `Authorization: Bearer sk-...` header，库不主动过滤 = 默认不安全
2. **PEP 420 native namespace（ADR 003）**：`packages/*/src/everalgo/` 不能有 `__init__.py`，没有「namespace 根集中入口」可借
3. **多 distribution（ADR 002）**：8 个 distribution / 11 个公开子包，规范需要在每个 distribution 上一致生效

相关 ADR：
- [ADR 003](003-namespace-package-pep420.md) PEP 420 namespace
- [ADR 004](004-providers-nested-in-llm.md) providers nested in `llm`
- [ADR 012](012-llm-stack-architecture.md) LLM stack architecture（算法层不加 retry → 出错只 raise，依赖 logger 上下文）

相关硬约束：
- **H1** evermem 文档契约（evermem 是日志消费方，需要稳定 logger 命名）
- **H4** 无状态接口（不能内置 logging state）
- **H6** v0.x 演化阶段（不引入「未来可能需要」的复杂度）

## 候选方案

### 维度 1：日志技术路线（算法库做不做日志、用什么）

| 方案 | 描述 | 代表 |
|------|------|------|
| A. 完全不用 logging | 错误抛 exception，告警走 `warnings.warn`，性能外包 profiler | numpy / scipy / pandas / instructor |
| B. 全程 logging | 所有诊断走 stdlib `logging` | requests / urllib3 / openai-python / anthropic-sdk-python / dspy / litellm |
| **C. 混合** | I/O 路径 logging；用户行为 / 弃用走 warnings；纯算法 exception | sklearn（callback + verbose + Warning）/ EverAlgo |

### 维度 2：library 配置策略（library 在 import 时挂什么）

| 方案 | 描述 | 代表 |
|------|------|------|
| A. 不挂任何 handler | 完全交 root logger 处理 / Py3.2+ lastResort 兜底 | sphinxcontrib-applehelp `__init__.py:42` |
| **B. 每个子包挂 NullHandler** | 每个公开子包子 logger 上 `addHandler(NullHandler())` | google-auth `__init__.py:34` / google-cloud-spanner-driver `__init__.py:52-53` / requests / urllib3 |
| C. 集中在 namespace 根挂 | namespace 根 `__init__.py` 一次挂 | **PEP 420 禁止**（破坏 namespace portion 拼接） |

### 维度 3：Secret 过滤（API key / Bearer 等敏感值）

| 方案 | 描述 | 代表 |
|------|------|------|
| A. 不过滤 | 文档警告用户自己注意 | requests / urllib3（无 Filter）|
| B. 提供 Filter，opt-in | 用户显式 attach | 无大项目采用此路 |
| **C. 默认装载** | `import` 时无条件 attach 到 LLM logger | openai-python `_utils/_logs.py:33-42` / anthropic-sdk-python `_utils/_logs.py` |

### 维度 4：规范执行（如何防止规范被违反）

| 方案 | 描述 |
|------|------|
| A. 人工 code review | 依赖 reviewer 记得检查 |
| **B. ruff lint 规则强制** | `G` + `LOG` + `TRY` 规则集，pre-commit + CI 双层 |

## 客观优劣分析

### 维度 1：A vs B vs C 优劣

| 方案 | 优势 | 劣势 |
|------|------|------|
| A 不用 logging | 零仪式；用户用 cProfile / line_profiler / IPython %timeit 看耗时（sklearn 官方推这套）；exception message 详尽 | I/O 调用 retry / fallback 无诊断手段 → debug 困难 |
| B 全程 logging | 任何节点都有日志 | 算法库内部循环 / 几何距离这种非 I/O 计算打日志没意义且影响性能 |
| C 混合 | 各得其所；warnings 触发用户警觉而非淹没日志；exception 让错误立刻被发现 | 边界判断需明确（哪条信息走 warning、哪条走 logger）|

### 维度 2：A vs B vs C 优劣

| 方案 | 优势 | 劣势 |
|------|------|------|
| A 不挂 handler | 代码最少；Py3.12 lastResort 已替代 NullHandler 的功能价值 | 偏离 IO 类库行业惯例（requests/urllib3/openai 都挂）；放弃「显式标识 library logger」的规范性表达 |
| B 子包各挂 | 工业标准，9/10 主流 IO 库都这么做；多 distribution 时每个发布包自包含 | 11 处 boilerplate（每个子包 `__init__.py` 加一行） |
| C 集中 namespace 根 | 单点 setup | **PEP 420 不允许**——`everalgo/__init__.py` 一旦存在，其他 7 个 distribution 的 namespace portion 无法被 Python finder 发现 |

### 维度 3：A vs B vs C 优劣

| 方案 | 优势 | 劣势 |
|------|------|------|
| A 不过滤 | 实现最简 | **默认不安全**——用户开 DEBUG 立即泄露 API key 到 stderr |
| B opt-in | 安全责任移交用户 | 用户不知道有这个 Filter 的话等于没装 |
| C 默认装 | 默认安全（fail-safe） | 一个无声 mutate `record.args` 的副作用，需文档化 |

### 维度 4：A vs B 优劣

| 方案 | 优势 | 劣势 |
|------|------|------|
| A 人工 review | 灵活，可视上下文判断 | 规模化后人会忘；老员工知道、新员工踩坑 |
| B lint 强制 | 0 遗漏；规则代码化即文档；新贡献者立刻拿到反馈 | 偶尔需要 `# noqa` 标注合法例外 |

## 对 EverAlgo 适配度评估

### 维度 1：C（混合）适配度

| 评估项 | 结论 |
|------|------|
| LLM I/O 路径需诊断（H1 evermem 要看 boundary cut / LLM retry / fallback） | ✅ **强需要** logger 路径 |
| 算法内部循环（rrf 融合、tokenize 计数、几何距离） | ✅ 纯算法无日志价值，exception + warning 即可 |
| 用户传错参（空 MemCell list、错配 type） | ✅ warning（参考 numpy `errstate(all='warn')`、pandas `PerformanceWarning`） |
| API 弃用 | ✅ `DeprecationWarning`（H6 v0.x 演化阶段会有删除 / 重命名） |
| 性能耗时 | ✅ **不做**——sklearn 官方明确推 `%timeit` / `cProfile` / `line_profiler`，库不报时是行业惯例 |

### 维度 2：B（子包各挂 NullHandler）适配度

| 评估项 | 结论 |
|------|------|
| PEP 420 兼容 | ✅ **强需要**（C 方案直接被 ADR 003 排除） |
| 多 distribution 一致性（H1）| ✅ 每个 distribution 自包含，独立发版（[ADR 002](002-multi-distribution-vs-single.md)）无依赖回溯 |
| 11 处 boilerplate 成本 | ⚠️ 可接受——一次写完，新增 distribution 时 CONTRIBUTING 提示 |
| 偏离 sphinxcontrib（A 方案）的「最简」精神 | ⚠️ 可 mitigate——「最简」≠「最规范」，IO 类库行业基准是挂 |

### 维度 3：C（默认装 Filter）适配度

| 评估项 | 结论 |
|------|------|
| evermem 在生产线 DEBUG 排查 LLM 问题（H1）| ✅ **强需要**——默认安全防止 secret 进生产日志 |
| 算法同学 debug 时（H3）开 DEBUG 看 request 不用担心 leak | ✅ 强受益 |
| 副作用 mutate `record.args` | ⚠️ 可 mitigate——docstring 明文；只动 dict args，tuple args 不碰 |

### 维度 4：B（ruff 强制）适配度

| 评估项 | 结论 |
|------|------|
| 起步阶段贡献者多元、心智不齐（H3）| ✅ **强需要**——lint 是最便宜的对齐手段 |
| pre-commit 已就位（README 已配） | ✅ 零额外基建 |
| 个别合法例外（`# noqa: G004` 等）| ⚠️ 可接受——明确标注理由即可 |

## 决策

- **维度 1 选 C**：I/O 路径走 `logging`；用户行为走 `warnings.warn`；纯算法走 `raise` + 详尽 exception message；性能完全外包 profiler
- **维度 2 选 B**：每个公开子包 `__init__.py` 一行 `getLogger(__name__).addHandler(NullHandler())`
- **维度 3 选 C**：`SensitiveHeadersFilter` import 时默认 attach 到 `everalgo.llm`，无 opt-out kwarg
- **维度 4 选 B**：ruff `G` + `LOG` + `TRY` 规则集启用，pre-commit + CI 双层强制

## 实施细节

### 1. 子包 logging 配置

每个公开子包 `__init__.py` 末尾一行：

```python
# packages/everalgo-<dist>/src/everalgo/<subpkg>/__init__.py
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
```

11 个子包同样写法：`everalgo.types / .llm / .prompts / .testing / .boundary / .clustering / .rank / .parser / .user_memory / .agent_memory / .knowledge`。

`everalgo.llm` 额外两行：

```python
# packages/everalgo-core/src/everalgo/llm/__init__.py
from everalgo.llm._filters import SensitiveHeadersFilter
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())
_logger.addFilter(SensitiveHeadersFilter())
```

`everalgo` namespace 根**不挂任何东西**（PEP 420 禁止）。

### 2. 模块内调用

```python
import logging
logger = logging.getLogger(__name__)  # 解析为 everalgo.boundary._tokenize 等
```

### 3. 何时用 logging vs warnings vs exception

| 场景 | 工具 | 示例 |
|------|------|------|
| LLM I/O 路径运行时诊断 | `logger.*` | `logger.debug("provider=%s model=%s tokens=%d", p, m, n)` |
| 用户行为不当（空输入、错参） | `warnings.warn(UserWarning)` | `warnings.warn("MemCell list is empty", UserWarning, stacklevel=2)` |
| API 弃用 | `warnings.warn(DeprecationWarning)` | numpy 派 |
| 数值 / 算法错误 | `raise ValueError("详细诊断")` | numpy 派；exception message 写详尽（参 numpy `shapes (3,4) and (5,6) not aligned: 4 (dim 1) != 5 (dim 0)`）|
| 性能耗时 | **不做**——用户用 `cProfile` / `%timeit` / `line_profiler` | sklearn 官方推荐 |
| 迭代进度 | **起步阶段不做**，未来可按需引入 `verbose=int` 参数（sklearn 模式）| — |

### 4. 调用规范

| 规则 | 正例 | 反例 | 强制方式 |
|------|------|------|------|
| 用 `__name__` | `getLogger(__name__)` | `getLogger("everalgo.boundary")` 手写 | code review |
| Lazy `%`-format | `logger.debug("count=%d", n)` | `logger.debug(f"count={n}")` | ruff `G004` |
| `%`-format 不用 `.format()` | `logger.debug("v=%s", v)` | `logger.debug("v={}".format(v))` | ruff `G001` |
| 不用 `+` 拼接 | `logger.debug("v=%s", v)` | `logger.debug("v=" + str(v))` | ruff `G003` |
| 异常用 `logger.exception()` | `except X: logger.exception("...")` | `except X as e: logger.error(f"...{e}")` | ruff `TRY400` |
| 不调 root logger | `logger.warning(...)` | `logging.warning(...)` | ruff `LOG015` |

### 5. 级别语义

| Level | 用途 | 禁止 |
|-------|------|------|
| DEBUG | provider 名、model 名、token 计数、retry 次数、耗时 | **request / response body、prompt 文本、模型输出** |
| INFO | 算法主流程节点（boundary cut、cluster 决策、rank 完成）| 「成功」/「完成」类的客套日志（sklearn / numpy 都不打）|
| WARNING | fallback 触发、超时但成功；用户行为型 warning 用 `warnings.warn` 不用这个 | — |
| ERROR | 抛 `LLMError` 前的失败上下文 | — |

DEBUG 默认不 log body——body 含 prompt 嵌的 PII 和模型复述的输入，`SensitiveHeadersFilter` 不识别 body 内容。如需 body 日志，由用户**显式开**环境变量 `EVERALGO_LLM_LOG_BODY=1`，且开启后仍走 Filter 脱敏。起步阶段先不实现这个环境变量，等具体 provider 集成 PR 时再加。

### 6. Secret 过滤

`SensitiveHeadersFilter`（`packages/everalgo-core/src/everalgo/llm/_filters.py`）在 `everalgo.llm` import 时默认 attach。redact 规则：`record.args` 为 dict 时，key 匹配 `(authorization|api[-_]?key|x-api-key|bearer)` 的 value 替换为 `"<redacted>"`。tuple / `None` args 不动。

不可关闭——若未来用户提需求 opt-out，通过 `everalgo.configure(redact_headers=False)` 暴露而非 kwarg 漂移；但默认必须是过滤。

### 7. 消费者侧（evermem 等）

stdlib 标准用法，零配合：

```python
logging.basicConfig(level=logging.INFO)
logging.getLogger("everalgo").setLevel(logging.INFO)         # 只看 EverAlgo
logging.getLogger("everalgo.llm").setLevel(logging.DEBUG)    # 子包分粒度
```

## 行业实证印证

### EverAlgo 同定位（"调 LLM 的算法库 / SDK"）实证

| 项目 | NullHandler | Secret Filter | logging vs warnings |
|------|-------------|---------------|---------------------|
| requests | ✅ `src/requests/__init__.py` | ❌ | logging 派 |
| urllib3 | ✅ `src/urllib3/__init__.py` | ❌ | logging 派 |
| openai-python | ✅ `src/openai/_utils/_logs.py` | ✅ `SensitiveHeadersFilter` 默认 attach（同名同思路）`_utils/_logs.py:33-42` | logging 派 |
| anthropic-sdk-python | ✅ `src/anthropic/_utils/_logs.py` | ✅ 同 OpenAI 模式 | logging 派 |
| google-auth | ✅ `google/auth/__init__.py:34` | ❌ | logging 派 |
| google-cloud-spanner-driver | ✅ `google/cloud/spanner_driver/__init__.py:52-53` | ❌ | logging 派 |
| dspy | ✅ 自定义 `DSPyLoggingStream` `dspy/utils/logging_utils.py:60-72` | ❌ | logging 派 |
| litellm | ✅ 3 个独立 logger `litellm/_logging.py:259-266` + `SecretRedactionFilter` | ✅ | logging 派 |

共性：IO/LLM 类库**全部**挂 NullHandler；OpenAI / Anthropic / LiteLLM 在 LLM 层默认装 Secret Filter。

### 算法库 / 科学计算 实证

| 项目 | logging 用法 |
|------|--------------|
| numpy | 不用 logging — 用 `errstate(all='warn')` / `warnings.warn` |
| scipy | 不用 logging — 用 `optimize.minimize(callback=fn)` 让用户处理 |
| pandas | 不用 logging — 用 `on_bad_lines='warn'` / `PerformanceWarning` |
| sklearn | 不用 logging — 用 `verbose: int` 参数 + 条件 print() |
| instructor | 不用 logging — 完全靠 callback hook 系统 |

共性：纯算法路径用 `warnings.warn` + exception + callback + verbose 参数，不打 logging。

### 反例分析

DSPy / LiteLLM 走「库内主动 addHandler + enable/disable API」（违反 Python 官方 [logging HOWTO](https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library) 的「strongly advised that you do not add any handlers other than `NullHandler` to your library's loggers」）—— 偏向应用层定位，与 EverAlgo 算法库定位不符。EverAlgo 不学这一条。

### PEP 420 namespace 多 distribution 工程实证（维度 2 决策依据）

| 项目 | NullHandler 落点 |
|------|------------------|
| google-cloud-python | 每个子 distribution 的最顶层包 `__init__.py`（`google/auth/__init__.py:34`、`google/cloud/spanner_driver/__init__.py:52-53`）|
| azure-sdk-for-python | 单独 `_telemetry/logging_handler.py` 模块，专门处理外部依赖 logger |
| sphinxcontrib-* | 不挂 |

**全部不挂 namespace 根**（`google/`、`google/cloud/`、`azure/`、`sphinxcontrib/` 都没有 `__init__.py`），印证维度 2 的 C 方案被工业全员排除。

## QA / 边界澄清

**Q: 子包 `__init__.py` 各挂 NullHandler，与 `everalgo.llm` 上的 Filter 是否有重复挂风险？**

A: 没有。`addHandler` 只在 import 时执行一次（Python `sys.modules` 缓存保证 `__init__.py` body 不重复执行）。第二次 `import` 同一子包不会重复挂 handler。

**Q: 子包内部模块（如 `everalgo/boundary/_tokenize.py`）需要做什么 logging setup？**

A: 不需要。模块顶部 `logger = logging.getLogger(__name__)` 即可。`getLogger("everalgo.boundary._tokenize")` 自动归属到 `everalgo.boundary` → `everalgo` → root 的 logger 树，沿默认 `propagate=True` 冒泡，子包 `__init__.py` 上的 NullHandler 已经在树上。

**Q: 算法包（如 `everalgo.boundary`）调 `everalgo.llm.LLMClient` 时，Filter 是否生效？**

A: 生效。LLM SDK 内部用 `everalgo.llm.*` logger 打日志（如 `everalgo.llm.providers.openai`），record 经过 `everalgo.llm` 上的 Filter。算法包自己的 `everalgo.boundary.*` logger 不经过 LLM Filter，但算法包不打 header / secret，没有 leak 风险。

**Q: 用户开 `EVERALGO_LLM_LOG_BODY=1` 后会怎样？**

A: 起步阶段**未实现**——这是占位规范，等具体 provider 集成 PR 时再加。开了之后预期行为：body 走 dict args，仍经过 Filter（虽然 Filter 不识别 body 字段名，但 header 在 body 里时仍会 redact）。
