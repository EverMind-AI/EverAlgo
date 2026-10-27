# Prompt 文本审核标准

本文件是项目正式的 Prompt 文本审核标准，由 [AGENTS.md](../AGENTS.md) 和 [贡献指南](../CONTRIBUTING.md) 引用，规则正文以本文件为唯一来源。

## 目的与边界

审核模型实际接收的完整 Prompt，包括指令、上下文、示例、动态输入及输出要求，排除能够从文本和请求组装结果识别的实质设计缺陷。

- 必须：适用时必须满足；存在实质缺陷时，修改后复核。
- 应当：默认采用；等效且清晰的写法可以保留。
- 按需：用于解决具体问题，不要求补齐所有技巧。
- 强制级别属于项目约定，不表示官方文档采用相同强度的措辞。
- 不设置机械评分，不以篇幅、章节数量、角色、XML 或示例数量判断质量；不要求把本准则写入运行时 Prompt。
- 不将缓存、版本管理、CI 或发布流程纳入文本合格条件。文本通过不替代目标模型的正确性、输出 token 和耗时验证。
- 保留官方建议的模型与场景条件；工程落实与项目约定分别标明，不把单厂商用法当成跨模型共识。

## 适用范围

新建或修改 Prompt、动态材料的渲染与组装、调用方附加指令或模型可见输出约束时，必须按本标准审核受影响的完整请求。审核应覆盖本次修改涉及的输入类型和配置，并明确未覆盖的范围。

本标准不要求为形式统一批量迁移未修改的历史 Prompt；涉及既有模板的修改时，连同受影响的上下文和契约一起审核。已有清晰且等效的表达可以保留。

文本审核通过不能替代目标模型的效果、输出 token 和耗时验证。模型实验的验收目标由具体任务确定，记录实际请求配置和观测结果，不以文本审核结论冒充模型效果保证，也不将本标准加入运行时输入。

## 为什么这样选择

保留原准则的条件性规则和证据要求，避免形式化改写。明确审核 API schema 和组装后消息，避免只看 system 模板；限定 XML 风格边界检查，避免误把标签引用当成不合法 XML。模板审核还需要说明动态输入与可选配置的覆盖范围。

本标准保留 C1–C10、E1–E6、F1–F7、N1–N6、S1–S6、A1–A6 共 41 个规则编号，便于审核记录引用。C9、F3、A1 明确完整请求与动态边界的审核要求；C3、E4 作为项目约定，不依赖未能定位的厂商直接表述。审核记录属于审阅产物，不新增模型运行时输出字段。

## 任务与逻辑

| 编号 | 规则与审核标准 | 适用范围 | 级别 | 依据 |
| --- | --- | --- | --- | --- |
| C1 | 任务明确。 能判断要完成什么、处理什么对象、交付什么结果；不得只有角色或背景而没有实际任务。 | 所有 Prompt | 必须 | 官方建议的工程落实；[OpenAI：明确最终目标](https://developers.openai.com/api/docs/guides/reasoning-best-practices) |
| C2 | 成功要求具体。 写明影响结果的实际要求；「专业」「深入」「高质量」不能替代具体标准。 | 所有 Prompt | 必须 | 官方建议的工程落实；[Anthropic：清楚直接地描述要求](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#be-clear-and-direct) |
| C3 | 指令相互兼容。 同一条件下的要求能够同时满足；存在例外或分支时明确适用条件，不让模型自行调和冲突。 | 多条指令或约束 | 必须 | 项目约定；逻辑兼容性检查不依赖厂商背书。原引用的 Google 页面本次未定位到该直接表述。 |
| C4 | 示例与规则一致。 示例、正文、字段定义和输出样例遵循同一套要求；示例不得隐含另一套判断标准。 | 含示例或输出样例 | 必须 | 官方明文；[OpenAI：示例应与指令一致](https://developers.openai.com/api/docs/guides/reasoning-best-practices) |
| C5 | 概念和指代明确。 影响结果的术语、分类边界、时间范围、单位及代词有明确含义；同一概念保持一致称呼。 | 含业务概念、参数或判断规则 | 必须 | [Google：Gemini 3 术语与参数建议](https://ai.google.dev/gemini-api/docs/prompting-strategies#gemini-3)；通用化为项目约定 |
| C6 | 必要信息可获得。 所需材料、定义或工具已提供，或有明确获取方式；不得依赖未提供的内部背景、材料或能力。 | 依赖上下文或工具 | 必须 | 官方建议的工程落实；[Google：上下文与任务信息](https://ai.google.dev/gemini-api/docs/prompting-strategies) |
| C7 | 事实边界明确。 说明允许使用哪些来源、是否允许推断；不得把缺失信息、猜测或虚构引用补成事实。 | 事实问答、抽取、研究与分析 | 必须 | 官方建议的工程落实；[OpenAI：事实依据与不确定性](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5-2_prompting_guide) |
| C8 | 异常情况有明确去向。 缺失、矛盾、无匹配或无法完成会影响任务时，规定澄清、返回空值、报告失败等行为；不要求任何输入都生成完整答案。 | 存在上述情况的任务 | 必须 | 官方建议的工程落实；[OpenAI：歧义处理](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5-2_prompting_guide)、[不兼容输入处理](https://developers.openai.com/api/docs/guides/structured-outputs) |
| C9 | 输出要求可执行。明确实际需要约束的内容、格式、语言、长度或粒度；机器读取结果须定义必要字段、取值含义及空值表示。将 API response_format、工具参数 schema 与自然语言要求一起核对；上层机制已提供的约束无需重复写入正文。结构符合 schema 不等于业务判定正确。 | 有交付约束或机器消费需求 | 必须 | 官方建议的工程落实；[Anthropic：输出要求](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#be-clear-and-direct)、[OpenAI：结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs) |
| C10 | 流程和依赖成立。 顺序影响结果时写清步骤；条件分支写清触发条件；步骤不得依赖尚未获得的结果。 | 多步骤任务 | 必须 | 官方建议的工程落实；[Anthropic：顺序与完整性](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#be-clear-and-direct) |

## 表达与职责

| 编号 | 规则与审核标准 | 适用范围 | 级别 | 依据 |
| --- | --- | --- | --- | --- |
| E1 | 按职责组织内容。 任务、规则、背景、示例和输入能够清楚区分；相关要求集中放置，跨区块引用明确。 | 多区块 Prompt | 必须 | 官方建议的工程落实；[Anthropic：内容分区](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#structure-prompts-with-xml-tags)、[OpenAI：分隔结构](https://developers.openai.com/api/docs/guides/reasoning-best-practices) |
| E2 | 区分指令与材料。 明确哪些内容是执行要求，哪些是待处理材料；材料中的命令不得自动成为任务指令。 | 含外部文本、检索或工具返回 | 必须 | 官方建议的工程落实；[OpenAI：不可信输入与指令边界](https://developers.openai.com/api/docs/guides/agent-builder-safety) |
| E3 | 要求直接且可判断。 优先描述期望行为，同时保留必要的禁止事项；不用反复强调或强硬语气替代具体规则。 | 所有 Prompt | 应当 | 官方建议的工程落实；[Anthropic：格式控制](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#control-the-format-of-responses)、[Google：指令与约束](https://ai.google.dev/gemini-api/docs/prompting-strategies) |
| E4 | 删去无效内容。 删除无关背景、同义反复和不改变行为的套话；必要的重复保持语义一致，并具有明确用途。 | 所有 Prompt | 应当 | 项目约定；参考 [Google 的简洁直接建议](https://ai.google.dev/gemini-api/docs/prompting-strategies#gemini-3)，不将其等同于禁止所有重复。 |
| E5 | 约束与任务匹配。 每个约束服务于目标、边界或交付；不因套模板强加无关步骤、输出或角色。 | 所有 Prompt | 应当 | 官方建议的工程落实；[OpenAI：简单直接的指令](https://developers.openai.com/api/docs/guides/reasoning-best-practices) |
| E6 | 详细程度适当。 规则已能唯一表达要求时不继续展开；存在实质歧义时补充定义、条件或示例，不依赖模型猜测。 | 所有 Prompt | 应当 | 官方建议的工程落实；[Anthropic：清晰表达](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#be-clear-and-direct)、[OpenAI：简单直接的指令](https://developers.openai.com/api/docs/guides/reasoning-best-practices) |

## 结构与格式

| 编号 | 规则与审核标准 | 适用范围 | 级别 | 依据 |
| --- | --- | --- | --- | --- |
| F1 | 复杂内容默认用 XML 分区。 根据职责划分任务、规则、材料、示例等区块，不把不同性质的内容无标识地混杂。 | 包含多类内容的 Prompt | 应当 | [Anthropic：XML 分区](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#structure-prompts-with-xml-tags)；默认格式为项目约定 |
| F2 | 同类边界使用一致形式。 不对同一层级、同一用途交替使用多套分隔方式。区块内部可以使用列表；原始代码、JSON 等保留所需格式。 | 多区块 Prompt | 应当 | [Google：Gemini 3 结构一致性](https://ai.google.dev/gemini-api/docs/prompting-strategies#gemini-3)；具体适用边界为项目约定 |
| F3 | XML 风格的区块边界完整。作为分隔符使用的起止标签匹配，动态材料不得使外围区块边界或消息记录归属产生歧义。正文中引用标签名称不要求人为补成 XML 元素，也不要求整份 Prompt 能被 XML 解析器解析。材料可能含同名分隔符时，应采用能够保持归属清晰的表示方式；这不构成提示注入防护保证。 | 使用 XML 标签 | 必须 | 项目约定，无直接官方规定；目的为保持内容边界明确 |
| F4 | 嵌套反映实际包含关系。 例如文档集合包含单篇文档；不为装饰增加无意义层级。 | 嵌套区块 | 应当 | 官方明文；[Anthropic：自然层级嵌套](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#structure-prompts-with-xml-tags) |
| F5 | 仅保留有实际用途的区块。 不为凑模板添加空角色、空背景、空示例或重复任务。 | 模板化 Prompt | 应当 | 项目约定，无直接官方规定；目的为保持精简与可读性 |
| F6 | 分开定义组装格式和输出格式。 用 XML 组织输入不意味着输出必须是 XML；输出要求另行明确。 | 对输出形式有要求 | 必须 | 项目约定；相关输出指导见 [Anthropic：格式控制](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#control-the-format-of-responses)、[OpenAI：结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs) |
| F7 | 标签不替代消息权限。 XML 标签用于表达内容边界，不提升材料的指令权限，也不构成提示注入防护保证。 | 多角色消息或外部材料 | 必须 | 官方机制的工程落实；[OpenAI：消息权限与不可信输入](https://developers.openai.com/api/docs/guides/agent-builder-safety) |

## 区块命名

| 编号 | 规则与审核标准 | 适用范围 | 级别 | 依据 |
| --- | --- | --- | --- | --- |
| N1 | 名称准确概括内容。 标签回答该区块包含什么或负责什么；避免 `<part1>`、`<misc>` 等无法表达职责的名称。 | 所有命名区块 | 必须 | [Anthropic：描述性命名](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#structure-prompts-with-xml-tags)；具体检查为工程落实 |
| N2 | 默认采用名词短语。 数据按对象命名，规则按约束对象与规则性质命名，输出定义按结果与结构命名。 | 数据、规则、输出定义区块 | 应当 | 项目约定；参考 [OpenAI 区块示例](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5-2_prompting_guide)、[Google 区块示例](https://ai.google.dev/gemini-api/docs/prompting-strategies#structured-prompting-examples) |
| N3 | 动词短语用于明确的行为模块。 使用时应准确表达动作，并与同类模块命名保持一致；不强制将所有标签改为动词。 | 独立行为模块 | 按需 | 项目约定；[Anthropic 官方示例包含动词短语标签](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#control-the-format-of-responses) |
| N4 | 多词标签统一使用小写 snake_case。 不在同一 Prompt 中混用大小写与连接形式。 | 多词 XML 标签 | 应当 | 项目约定；参考 [OpenAI 官方示例](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5-2_prompting_guide)，不宣称具有独立效果优势 |
| N5 | 引用逐字匹配。 正文引用标签或字段时使用实际名称；同一职责不随意更名，不同职责不使用容易混淆的同名标签。 | 有名称引用的 Prompt | 必须 | [Anthropic：一致命名](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#structure-prompts-with-xml-tags)；逐字匹配为项目约定 |
| N6 | 名称与正文分工清楚。 标签概括职责，正文明确动作、条件和边界；不能只靠标签名称表达完整规则。 | 指令与规则区块 | 必须 | 项目约定，无直接官方规定；目的为保证任务要求完整 |

## 按需技巧

| 编号 | 规则与审核标准 | 适用范围 | 级别 | 依据 |
| --- | --- | --- | --- | --- |
| S1 | 示例解决具体歧义。 用于说明难以仅靠规则表达的分类边界、风格或输出；保持相关、多样且与规则一致，不强制固定数量。 | 规则表达仍有歧义 | 按需 | [Anthropic：有效示例](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#use-examples-effectively)、[OpenAI：按需使用少样本](https://developers.openai.com/api/docs/guides/reasoning-best-practices) |
| S2 | 角色限定职责或表达方式。 角色与任务相关，不替代任务要求，不作为能力或正确性保证。 | 需要稳定职责、视角或语气 | 按需 | [OpenAI：角色与任务信息](https://developers.openai.com/api/docs/guides/prompting)；保证限制为项目约定 |
| S3 | 背景解释非直观要求。 约束原因影响正确理解时，补充简短用途；不为显而易见的要求重复解释。 | 特殊约束或下游用途 | 按需 | 官方明文；[Anthropic：补充背景](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#add-context-to-improve-performance) |
| S4 | 推理要求适配模型。 不统一添加逐步思考或展示完整思考过程的要求；需要解释时，明确交付可核查的步骤或依据。 | 推理与解释任务 | 应当 | 官方建议的工程落实；[OpenAI：推理提示最佳实践](https://developers.openai.com/api/docs/guides/reasoning-best-practices) |
| S5 | 自检有对象与停止条件。 写清核对哪些要求、什么情况下需要修正；不使用无边界的反复验证指令。 | 容易遗漏、复杂或高风险任务 | 按需 | [OpenAI：高风险输出自检](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5-2_prompting_guide)；停止条件为项目约定 |
| S6 | 长材料布局有针对性。 可以采用材料在前、问题在后的结构，同时保持高优先级约束所在位置清楚；不强制所有任务采用相同顺序。 | 长文档或多文档任务 | 按需 | [Anthropic：长上下文](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices#long-context-prompting)、[Google：Gemini 3 长上下文](https://ai.google.dev/gemini-api/docs/prompting-strategies#gemini-3)；须保留模型与场景限制 |

## 审核执行

| 编号 | 规则与审核标准 | 适用范围 | 级别 | 依据 |
| --- | --- | --- | --- | --- |
| A1 | 审核实际请求中的完整消息上下文及相关输出约束，包含消息角色、组装后的动态输入、可选指令与 API schema；不得将上层已明确的要求判为缺失，也不得忽略上下层之间的冲突。区分单次已渲染请求的结论与模板适用范围的结论；未覆盖的可选配置不得宣称已审核通过。 | 所有审核 | 必须 | 项目审核约定，无直接官方评分标准。 |
| A2 | 每个问题指出具体文本位置、对应规则及可能导致的理解或执行偏差；不凭个人文风偏好判定缺陷。 | 所有问题记录 | 必须 | 项目审核约定，无直接官方评分标准。 |
| A3 | 任务含糊、要求冲突、必要信息缺失、示例违背规则、内容边界不明或要求不可执行时，修改后复核。 | 存在实质缺陷 | 必须 | 项目审核约定，无直接官方评分标准。 |
| A4 | 不适用的规则不要求补齐；没有角色、示例、XML 或固定章节，不单独构成缺陷。 | 条件性要求 | 必须 | 项目审核约定，无直接官方评分标准。 |
| A5 | 信息不足时指出无法判断的事项，不自行推断已满足，也不直接判定存在缺陷。 | 审核材料不完整 | 必须 | 项目审核约定，无直接官方评分标准。 |
| A6 | 不为形式统一制造无效修改；不同写法能够同样清楚地表达任务时，允许保留。 | 等效表达 | 必须 | 项目审核约定，无直接官方评分标准。 |

## 格式与命名的适用条件

复杂 Prompt 默认使用 XML 风格标签分区；已有清晰、一致的 Markdown 结构可以保留，简单任务可以直接使用自然语言。标签默认采用描述内容或职责的名词短语；独立行为模块可用动词短语，不以词性单独否决。名称与正文应分工清晰，避免机械重命名。

推荐标签仅供选用：`<task>`、`<context>`、`<input>`、`<source_text>`、`<conversation_history>`、`<classification_rules>`、`<evidence_requirements>`、`<constraints>`、`<examples>`、`<example>`、`<output_format>`、`<response_schema>`。正文不是 schema 时，不使用 `<response_schema>`；不得为了使用这些名称添加无用区块。

## 审核记录与判定

审核记录应包含以下信息；这是审阅产物，不加入模型的运行时输出：

1. 审核对象：源文件位置、实际渲染请求及 schema；敏感材料仅引用获准访问的位置，不复制密钥或个人数据。
2. 审核范围：单次请求或模板；覆盖的输入类型、可选配置和未覆盖部分。
3. 每项问题：规则编号、具体文本/代码位置、冲突条件或最小反例、预期偏差及建议修改。
4. 证据性质：文本可确认缺陷、仅存在的风险、实际模型复现，三者分开记录。
5. 结论：通过、修改后复核或信息不足；不把尚未验证的风险写成已复现故障。

- 通过：所有适用的必须项满足，未发现影响理解或执行的实质文本问题。
- 修改后复核：存在可定位的实质缺陷，列出修改要求与规则编号。
- 信息不足：缺少必要上下文、定义或依赖，列出具体缺失项；其他已确认问题仍分别报告。

对于分类与流式窗口任务，按适用性选择单条/多条、完整/未完成、冲突证据、材料中含命令或同名分隔符等反例。这个清单只服务于审核，不要求任何 Prompt 都增加同样的示例。

## 来源与维护

本次核对日期：2026-10-27。官方页面可能更新；引用应能对应到具体建议，无法定位时标为待核实或项目约定，不用网站名替代证据。以下页面只支持其明确陈述的建议，不证明厂商内部 Prompt 统一采用本规范。

- [OpenAI：Reasoning best practices](https://developers.openai.com/api/docs/guides/reasoning-best-practices)：明确目标、分隔结构、示例一致性与推理模型适配。
- [Anthropic：Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)：清晰表达、示例、XML 风格区块与自然层级。
- [Google：Prompt design strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies)：任务、上下文及 Gemini 3 专节的结构与参数建议。
- [OpenAI：Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)：API 输出约束及其边界。
- [OpenAI：Safety in building agents](https://developers.openai.com/api/docs/guides/agent-builder-safety)：不可信输入与消息权限。
- [OpenAI：GPT-5.2 Prompting Guide](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5-2_prompting_guide)：特定模型指南，保留适用范围，不作为跨模型强制依据。

维护本标准时，应同步检查 [AGENTS.md](../AGENTS.md)、[贡献入口](../CONTRIBUTING.md) 和 [开发贡献指南](contributing.md) 的引用，避免产生多份规则正文。规则更新应写明适用范围和选择理由，不把风格偏好升级为没有依据的必须项。
