# everalgo-boundary 0.4.1 修复报告

日期：2026-10-27。本报告为 0.4.1 发布依据，模型实验结论已冻结；发布成功以版本 tag 对应的发布任务和 PyPI 记录为准，服务部署另行确认。

## 修复结论

Muggle 报告中的「未完成步骤被提前判为 tail_complete=True」已在复现样本中修复。Step 1、2、3、7 的单条窗口及前三步合并窗口，各测 5 次：已发布 0.4.0 共 25/25 次误判完成，0.4.1 候选版 25/25 次正确判为未完成。最终对照中，有明确标签的尾部判断为 165/165 正确，内部切点为 170/170 正确；180 对请求中候选版全部比已发布版快。证据见 [汇总证据](boundary-0.4.1-evidence.json) 的 `muggle_unfinished_windows`、`variants` 和 `comparison_to_published`。

发布包不等于部署服务。下游服务仍需升级依赖并部署，不能仅凭此报告认定线上已经使用修复版。

## 问题与修改

| 问题 | 修复行为 | 代码依据 |
| --- | --- | --- |
| 当前窗口被当成完整对话，单条动作或多条中间步骤提前关闭 | 明确输入是单条或多条流式窗口；内部切点与尾部完整度独立，窗口结束不是完成证据 | [batch prompt](../../packages/everalgo-boundary/src/everalgo/boundary/prompts/en/chat.py) |
| 未完成工作与已结束的状态交流混淆 | 区分工作流执行、问答交流和独立事实；工作流按整体结果判断，状态交流可在讨论的工作尚未结束时闭合 | [尾部判定规则](../../packages/everalgo-boundary/src/everalgo/boundary/prompts/en/chat.py) |
| 动态材料中的标签或换行可能破坏记录归属 | 转义正文与发言人中的 XML 分隔符，缩进续行，空正文保留编号占位；材料中的命令作为分类数据处理 | [batch 渲染](../../packages/everalgo-boundary/src/everalgo/boundary/_render.py) |
| 理由复述输入，增加生成量 | 要求简短英文判定依据，避免主题列表和消息范围；保留原输出字段与严格 schema | [输出定义](../../packages/everalgo-boundary/src/everalgo/boundary/prompts/en/chat.py)、[回复模型](../../packages/everalgo-boundary/src/everalgo/boundary/_replies.py) |

选择这些修改是为了修正完成对象与材料归属，同时控制输出成本；没有改变公共 API 或 schema 字段。增量 step prompt 与渲染保持不变。仓库同时正式接入 [Prompt 文本审核标准](../prompt-review-standard.md)，文本审核与实测效果分别验收。

## 实验设计

本报告只以已发布 `everalgo-boundary 0.4.0` 为验收基线，候选版对应本报告证据文件中的源码 SHA-256。旧版使用原 system prompt 和原渲染，候选版使用实际生产 `_batch_request` 组装，因此这是完整修复的对照，不是仅替换 system 文本的单因素实验。

- 模型：自部署 `qwen3-4b-instruct-2507`，以实际响应中的模型名为准。
- 样本：原 18 个复现与回归场景、此前的 6 个独立样例作为本轮回归，以及 12 个预先标注的新增反例；每版每场景 5 次，两版各 180 次计分调用。
- 新增反例覆盖状态交流结束与工作流继续、状态回答后的新待办、正文/发言人中的伪指令、伪记录编号、空正文与纯空白、普通代码标签事实、最终失败和失败后仍在重试。
- 保持 `response_format.type=json_schema`、`strict=true`、`temperature=0.3`、`max_tokens=512`、`enable_thinking=false`，未添加 custom instructions。schema 与已发布版一致。
- 同场轮换请求顺序、串行调用、零重试，预热不计分。HTTP 耗时包含完整响应下载，不含落盘和校验；p90 使用线性插值。

完整原始实验还保留中间候选对照，共 543 次调用，包括 540 次计分及 3 次预热；全部 HTTP 200、正常 stop、schema 有效。该中间候选不作为本报告验收基线。独立审计已逐请求重算汇总、核对冻结与当前源码哈希，证据见 `audit` 和 `raw_artifacts_sha256`。

## 与已发布版的比较

| 指标 | 已发布 0.4.0 | 0.4.1 候选版 |
| --- | ---: | ---: |
| 尾部判断正确数 | 79/165 | 165/165 |
| 内部切点正确数 | 170/170 | 170/170 |
| System prompt 字符数 | 4,769 | 6,830 |
| System prompt UTF-8 字节数 | 4,786 | 6,844 |
| 完整请求输入 token 均值 | 1,447.14 | 1,999.03 |
| 输出 token 均值 | 36.28 | 20.33 |
| 输出 token p50 / p90 | 33 / 40.1 | 20 / 21 |
| 输出 token 最大值 | 114 | 23 |
| HTTP 耗时 p50 | 672.832 ms | 516.526 ms |
| HTTP 耗时 p90 | 764.986 ms | 534.799 ms |
| HTTP 耗时均值 | 705.404 ms | 520.447 ms |

完整请求输入 token 包含 system、动态材料及请求封装，不能称为纯 system token。输出 token 的小数 p90 来自分位数插值，不代表单次生成了小数 token。

配对耗时差中位数为 −151.105 ms，180/180 对候选版更快。本轮观测满足相对已发布版的耗时不劣化要求；没有据此推导生产 SLO 或任意请求都会更快。表格精确值及统计定义见 [证据 JSON](boundary-0.4.1-evidence.json)。

## 验证与保留限制

- 本地 boundary 测试：222 项通过；跨包集成测试：261 项通过。发布前全仓检查进一步完成：Python 3.14 下 3,556 项通过、29 项按依赖或外部服务条件跳过；全仓 ruff 检查及格式检查、mypy、pyright 均通过，改动代码覆盖率为 100%。本地检查不替代 GitLab CI；精确命令与日志哈希见证据 JSON 的 `release_preflight`。
- 候选版 5 次 `reasoning` 为 4 个词，均出现在 `control-task-in-progress`，超过提示词要求的 2–3 词；这些请求输出为 22 token，候选版全部请求最大值为 23 token。词数要求是提示词约束，schema 未增加硬字数限制，`max_tokens` 仍为 512。本次保留已验证版本，不继续优化。
- 原报告摘录有截断；无明确 gold 标签的尾部不计入正确率，所以分母不等于全部 180 次。此前独立样例已用于本轮回归，新增反例参与过筛查，不能称为最终独立留出集。
- 共享端点负载与缓存状态未受控，每场景仅 5 次，尤其逐场景 p90 样本有限。结果不覆盖任意冲突 custom instructions、所有材料注入形式或全部生产输入。

## 可追溯材料

仓库内保留 [紧凑证据 JSON](boundary-0.4.1-evidence.json)，包含汇总、各类样本计数、Muggle 子集、模型调用配置、源码和原始记录哈希；不包含私有消息全文、服务地址或凭据。完整本地原始证据目录为 `/Users/admin/Downloads/muggle-ai-boundary-standard-fix-20261027/full-v2`，其中 `summary.json`、`independent-audit.json`、`manifest.json`、`cases.json`、`requests/`、`trials/` 和 `results.json` 支持逐请求核验。发布前应再次比较证据中的被测源码哈希，防止把本报告套用于未经验证的实现。
