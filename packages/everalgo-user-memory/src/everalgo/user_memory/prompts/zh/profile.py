"""Chinese prompts for ProfileExtractor.

``PROFILE_INITIAL_EXTRACTION_PROMPT`` is the active prompt used by :class:`ProfileExtractor`; it
replaces the prior 2-stage ``CONVERSATION_PROFILE_PART1 + PART2`` flow with a single call returning
``{explicit_info, implicit_traits}``. ``PROFILE_INITIAL_EXTRACTION_PROMPT`` and ``PROFILE_UPDATE_PROMPT``
inject a ``{target_user}`` so extraction is scoped to a single speaker; ``PROFILE_COMPACT_PROMPT`` only
re-summarises already-stored items.
"""

# Incremental Update Prompt
PROFILE_UPDATE_PROMPT = """
**关键语言规则**：你必须使用与正在更新的已有画像**相同**的语言输出全部内容，包括每一个性格标签。即使新的对话使用了不同的语言，也**不得**切换语言——画像的语言在首次创建时即已确定。下方的标签示例仅示范格式与粒度，不代表语言。此规则强制执行。

你是用户画像更新员。根据对话记录，判断需要对用户画像做哪些操作。

**目标用户：{target_user}**
这可能是多人对话，每行都用说话人的 user_id 标注。只更新 {target_user}（user_id 等于 {target_user} 的说话人）的画像。任何其他参与者陈述的、或关于他们自己的信息都属于那个人，绝不要归到 {target_user} 名下。

【当前用户画像】（每条都有 index 编号）
{current_profile}

【对话记录】（来自同一主题的多轮对话）
{conversations}

【任务】
分析对话，输出需要执行的操作列表（可以有多条操作）。可选操作类型：
- **update**: 修改现有条目（通过 index 指定）
- **add**: 新增画像条目
- **delete**: 删除现有条目
- **none**: 无需任何操作（当对话不包含任何用户信息时使用）

【操作选择指南】
- **update**: 现有条目有信息更新、补充、修改
- **add**: 发现全新的用户信息（与现有条目无关）
- **delete**: 以下情况应该删除：
  - 用户明确否定（如"我不再吃素了"）
  - 信息已过时（如"下周要出差"但已经过了）
  - 与新信息直接矛盾

【重要规则】
1. **挖掘标签**：隐式特征必须包含【性格标签】，例如：[风险厌恶型]、[社交驱动型]、[数据考据党]。
2. 只提取 {target_user} 的信息，不要把其他参与者的信息或 AI 助手的建议当成用户特征
3. evidence 要包含时间信息 - 如"2024年10月用户提到..."
4. explicit_info 和 implicit_traits 的 index 是独立编号的
5. **去重**：在使用 "add" 前，仔细检查所有已有条目。如果类似的特征/信息已存在（即使措辞不同），请用 "update" 来补充而非重复添加。只有确实全新的信息才用 "add"。

【画像定义与分析框架】
- **explicit_info（显式信息）**：可以直接从对话中提取的用户事实。
  - *包含内容*：基本资料、健康状况、能力技能、明确偏好等。

- **implicit_traits（隐式特征）**：基于行为推断的心理画像、性格标签和决策风格。
  - *提取要求*：请结合对话上下文，从决策模式、社交偏好、生活哲学等维度进行自由分析和概括。
  - *命名规范*：
    1. 标签必须简练、可读、可复用（便于检索/对比），尽量控制在 2-6 个字。
    2. 避免把多个维度硬拼成一个长标签；如果信息包含多个维度，请拆成多条隐式特征分别表达。
    3. 标签应描述“稳定的行为/心理倾向”，不要写成一次性的事件或短期状态。
  - 请做合理推理，提取出用户的深层特征
【输出格式】
无操作时：
```json
{{"operations": [{{"action": "none"}}], "update_note": "对话不包含用户信息"}}
```

有操作时（可以组合多条 add/update/delete）：
```json
{{
  "operations": [
    {{"action": "add", "type": "explicit_info", "data": {{"category": "...", "description": "...", "evidence": "..."}}}},
    {{"action": "add", "type": "implicit_traits", "data": {{"trait": "...", "description": "...", "basis": "...", "evidence": "..."}}}},
    {{"action": "update", "type": "explicit_info", "index": 0, "data": {{"description": "..."}}}},
    {{"action": "delete", "type": "implicit_traits", "index": 1, "reason": "..."}}
  ],
  "update_note": "新增2条显式信息和1条隐式特征，更新1条，删除1条"
}}
```

**关键语言规则**：你必须使用与正在更新的已有画像**相同**的语言输出全部内容，包括每一个性格标签。即使新的对话使用了不同的语言，也**不得**切换语言——画像的语言在首次创建时即已确定。上方的标签示例仅示范格式与粒度，不代表语言。此规则强制执行。
"""

# Compacting Prompt
PROFILE_COMPACT_PROMPT = """
**关键语言规则**：你必须使用与正在精简的画像**相同**的语言输出全部内容，包括每一个性格标签。精简操作绝不改变画像的语言。下方的标签示例仅示范格式与粒度，不代表语言。此规则强制执行。

当前用户画像有 {total_items} 条记录（explicit_info + implicit_traits 合计），超过了上限 {max_items} 条。

请精简画像至 **合计 {max_items} 条**（explicit_info + implicit_traits 两类加起来，不是每类 {max_items} 条）。

精简原则：
1. **合并同类项**：将同一维度的多条记录（如多次体重记录）合并为一条"当前状态+趋势"的描述。
2. **提炼标签**：隐式特征应归纳为性格标签（如[风险厌恶型]），删除重复或浅层的描述。
3. 删除不重要、已过时或短期状态。
4. 保留每条条目的字段完整（尤其是 evidence）。

当前画像：
{profile_text}

**重要**：输出的 explicit_info + implicit_traits 合计必须 ≤ {max_items} 条。
```json
{{
  "explicit_info": [
    {{"category": "...", "description": "...", "evidence": "..."}}
  ],
  "implicit_traits": [
    {{"trait": "...", "description": "...", "basis": "...", "evidence": "..."}}
  ],
  "compact_note": "说明删除/合并了哪些内容"
}}
```

**关键语言规则**：你必须使用与正在精简的画像**相同**的语言输出全部内容，包括每一个性格标签。精简操作绝不改变画像的语言。上方的标签示例仅示范格式与粒度，不代表语言。此规则强制执行。
"""

# Initial Extraction Prompt
PROFILE_INITIAL_EXTRACTION_PROMPT = """
**关键语言规则**：你必须使用与输入的对话内容**相同**的语言输出。所有输出**必须**与该语言保持一致。此规则强制执行。本次调用决定了画像的语言：后续的更新与精简调用都会沿用你在此处选择的语言，因此每一个性格标签也必须使用该语言书写——绝不能与画像其余部分使用不同的语言。

判定语言时**仅依据对话参与者本人撰写的内容**，不依据引用或粘贴的材料。判断何为粘贴材料时适用以下检验：连续两句及以上、读起来像是别处写好的成品文字——语气偏解释性或文献性、不对话中任何人说话、自身不提出请求、回答或决定——即为粘贴材料，无论其使用何种语言，也无论是否被引号或代码块包裹。判定时必须排除这部分内容，**即使它在篇幅上占据对话主体**。对话参与者的句子中夹杂的外语词汇不改变判定结果，判定依据是该句子结构所使用的语言。专有名词与技术术语在输出中保留原文形式，不因输出语言而被翻译。

你是一个"用户画像分析师"。请阅读下面的对话，构建用户画像。

**目标用户：{target_user}**
这可能是多人对话，每行都用说话人的 user_id 标注。只为 {target_user}（user_id 等于 {target_user} 的说话人）构建画像。任何其他参与者陈述的、或关于他们自己的信息都属于那个人，绝不要归到 {target_user} 名下。

【第一部分：显式信息 (explicit_info)】
用户的客观事实和当前状态，如身高体重、喜好、疾病等。

【第二部分：隐式特征 (implicit_traits)】
基于行为推断的心理画像、性格标签和决策风格。
*提取要求*：从决策、社交、生活观念等维度进行深度挖掘。
*命名规范*：Trait 字段必须简练精准，推荐“[形容词] [名词]”格式，严禁过度堆砌形容词。

【提取原则】
1. 只提取 {target_user} 本人的信息，不要把其他参与者的信息或助手的建议当成用户特征
2. 隐式特征必须有多个证据支撑：同一条隐式特征的 evidence 必须来自多个信号；证据可来自【当前对话】与/或【已有画像 current_profile 的 evidence】（更新时可用），不能仅凭单条新对话臆断
3. 每条信息用一句自然语言描述，通俗易懂

【输出格式】
请直接输出 JSON，格式如下：
```json
{{
  "explicit_info": [
    {{
      "category": "分类名",
      "description": "一句话描述",
      "evidence": "一句话证据（来自对话内容）"
    }}
  ],
  "implicit_traits": [
    {{
      "trait": "特征名称",
      "description": "一句话描述这个特征",
      "basis": "从哪些行为/对话推断出来的",
      "evidence": "一句话证据（来自对话内容）"
    }}
  ]
}}
```

**关键语言规则**：你必须使用与输入的对话内容**相同**的语言输出。所有输出**必须**与该语言保持一致，包括每一个性格标签。此规则强制执行。

判定语言时**仅依据对话参与者本人撰写的内容**，不依据引用或粘贴的材料。判断何为粘贴材料时适用以下检验：连续两句及以上、读起来像是别处写好的成品文字——语气偏解释性或文献性、不对话中任何人说话、自身不提出请求、回答或决定——即为粘贴材料，无论其使用何种语言，也无论是否被引号或代码块包裹。判定时必须排除这部分内容，**即使它在篇幅上占据对话主体**。对话参与者的句子中夹杂的外语词汇不改变判定结果，判定依据是该句子结构所使用的语言。专有名词与技术术语在输出中保留原文形式，不因输出语言而被翻译。

【对话原文】
{conversation_text}"""
