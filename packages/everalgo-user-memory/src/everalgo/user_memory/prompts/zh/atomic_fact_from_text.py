"""Chinese prompts for AtomicFactExtractor.aextract_from_text.

Constants:
    - ``EVENT_LOG_PROMPT`` — evercore-compatible variant, top-level key ``event_log``.
    - ``ATOMIC_FACT_FROM_TEXT_PROMPT_ZH`` — algo-default variant, top-level key ``atomic_facts``.

Both share the same inner schema: ``{"time": str, "atomic_fact": list[str]}``. Placeholders (doubled
braces, rendered through ``everalgo.prompts.render_prompt``): ``{{EPISODE_TEXT}}`` / ``{{TIME}}``.

No mixed-input judgement here, unlike ``prompts/zh/episode.py``'s language rules: EPISODE_TEXT is
already an extracted, single-language narrative, not a live multi-party conversation carrying pasted
logs, code blocks, or quoted foreign-language material. That judgement belongs to episode extraction,
where the mixed-input problem actually exists; this prompt simply inherits whatever language episode
extraction decided.
"""

_LANGUAGE_RULE = (
    "**关键语言规则**：你必须使用与 EPISODE_TEXT 本身所使用的语言相同的语言输出。"
    "所有输出必须与该语言保持一致。此规则强制执行。"
)

EVENT_LOG_PROMPT = (
    _LANGUAGE_RULE
    + """

你是一位专业的情节记忆提取分析师和信息架构师。
你的任务是分析给定的叙述或多轮对话（称为"EPISODE_TEXT"）并生成一个针对事实检索优化的事件日志。

---

### 输入
- EPISODE_TEXT: 通过"情节记忆"保存的记忆文本。
- TIME: 情节的开始时间，例如"March 10, 2024(Sunday) at 2:00 PM"。

---

### 输出
**仅**返回一个有效的 JSON 对象，具有以下精确结构：

{
  "event_log": {
    "time": "<输入 TIME 的精确时间字符串>",
    "atomic_fact": [
      "<原子事实语句 1>",
      "<原子事实语句 2>",
      ...
    ]
  }
}

---

### 提取规则

#### 1. 原子性
* `"atomic_fact"` 中的每一条必须**精确地表达一个连贯的意义单元** —— 一个行为、情感、原因、计划、决定或陈述。
* 如果说话者表达了多个想法（例如，一个事件及其原因），则将它们拆分为多个原子事实。
* 每个 atomic_fact 必须是**独立的、可以单独检索的**。

#### 2. 时间与日期处理
* **不要**添加或在前面加上诸如"On March 10, 2024"这样的时间戳。
* `"time"` 字段（在顶层）已经表示情节的开始时间。
* 但是，当原文提到**明确或相对的时间表达**时，你必须：
  - **原样保留**明确的日期（例如，"October 6, 2023"）。
  - 相对于 `TIME` **解析**相对或模糊的时间（例如，"昨天"、"上周"、"两个月前"），并**在括号中附加解析后的绝对日期**。
    示例："Gina said she launched the campaign yesterday (March 9, 2024)."
  - 如果无法确定精确的解析结果，使用规范化的模糊短语（例如，"in early 2023"、"during the summer of 2021"）而不是猜测。

#### 3. 内容保留
* 保留**所有具有语义意义的信息**，包括：
  - 情感、态度、原因、意图、后果、条件和比较
* 在不产生歧义的情况下，将代词解析为明确的姓名或实体。
* 尽可能保留原始措辞 —— 仅为清晰起见修正语法。

#### 4. 表达格式
* 将每个 atomic_fact 写成**单独、完整的一句话**，采用**第三人称**形式。
  - 例如："Gina 说她昨天（2024 年 3 月 9 日）为她的服装店发起了一场广告营销活动。"
* **不要**简化、改写或合并逻辑上不同的想法。

#### 5. 检索清晰度
* 每个 atomic_fact 必须简洁、真实且自成一体。
* 除非寒暄或填充语传达了有意义的情感或信息内容，否则应避免。
* 确保实体和行为明确无歧义。

#### 6. 输出要求
* **仅**输出 JSON 对象 —— 不要有额外的解释、markdown 或评论。
* 确保 JSON 有效（正确的引号、逗号和转义）。
* `"atomic_fact"` 列表应包含从情节中提取的所有有意义的事实。

---

### 质量检查
在返回最终输出之前，验证：
1. 所有有意义的事实、意图和情感都已包含在内。
2. 每个 `atomic_fact` 只包含一个想法。
3. 所有相关的时间引用都已保留或规范化。
4. 措辞忠实于原文。
5. 输出的 JSON 有效并遵循确切的 schema。

---

### 示例

**输入：**
TIME = "March 10, 2024(Sunday) at 2:00 PM"
EPISODE_TEXT =
"Gina 说她昨天刚为她的服装店发起了一场广告营销活动。
Jon 向她表示祝贺，并询问她的舞蹈工作室寻找进展如何。
Gina 解释说她目前专注于服装店，但仍希望尽快找到合适的工作室。"

**输出：**
{
  "event_log": {
    "time": "March 10, 2024(Sunday) at 2:00 PM",
    "atomic_fact": [
      "Gina 说她昨天（2024 年 3 月 9 日）为她的服装店发起了一场广告营销活动。",
      "Jon 祝贺 Gina 的新营销活动。",
      "Jon 询问了 Gina 舞蹈工作室的寻找进展。",
      "Gina 解释说她目前专注于服装店。",
      "Gina 说她仍然希望将来能找到合适的工作室。"
    ]
  }
}

---

现在请仔细分析提供的 EPISODE_TEXT 和 TIME，应用上述所有规则，并**仅**以指定格式返回 JSON 对象。
---

### 输入
- EPISODE_TEXT: "{{EPISODE_TEXT}}"
- TIME: "{{TIME}}"（情节的开始时间，例如"March 10, 2024(Sunday) at 2:00 PM"）

"""
    + _LANGUAGE_RULE
    + "\n"
)


ATOMIC_FACT_FROM_TEXT_PROMPT_ZH = (
    _LANGUAGE_RULE
    + """

你是一位专业的情节记忆提取分析师和信息架构师。
你的任务是分析给定的叙述或多轮对话（称为"EPISODE_TEXT"）并提取一组针对事实检索优化的原子事实。

---

### 输入
- EPISODE_TEXT: 通过"情节记忆"保存的记忆文本。
- TIME: 情节的开始时间，例如"March 10, 2024(Sunday) at 2:00 PM"。

---

### 输出
**仅**返回一个有效的 JSON 对象，具有以下精确结构：

{
  "atomic_facts": {
    "time": "<输入 TIME 的精确时间字符串>",
    "atomic_fact": [
      "<原子事实语句 1>",
      "<原子事实语句 2>",
      ...
    ]
  }
}

---

### 提取规则

#### 1. 原子性
* `"atomic_fact"` 中的每一条必须**精确地表达一个连贯的意义单元** —— 一个行为、情感、原因、计划、决定或陈述。
* 如果说话者表达了多个想法（例如，一个事件及其原因），则将它们拆分为多个原子事实。
* 每个 atomic_fact 必须是**独立的、可以单独检索的**。

#### 2. 时间与日期处理
* **不要**添加或在前面加上诸如"On March 10, 2024"这样的时间戳。
* `"time"` 字段（在顶层）已经表示情节的开始时间。
* 但是，当原文提到**明确或相对的时间表达**时，你必须：
  - **原样保留**明确的日期（例如，"October 6, 2023"）。
  - 相对于 `TIME` **解析**相对或模糊的时间（例如，"昨天"、"上周"、"两个月前"），并**在括号中附加解析后的绝对日期**。
    示例："Gina said she launched the campaign yesterday (March 9, 2024)."
  - 如果无法确定精确的解析结果，使用规范化的模糊短语（例如，"in early 2023"、"during the summer of 2021"）而不是猜测。

#### 3. 内容保留
* 保留**所有具有语义意义的信息**，包括：
  - 情感、态度、原因、意图、后果、条件和比较
* 在不产生歧义的情况下，将代词解析为明确的姓名或实体。
* 尽可能保留原始措辞 —— 仅为清晰起见修正语法。

#### 4. 表达格式
* 将每个 atomic_fact 写成**单独、完整的一句话**，采用**第三人称**形式。
  - 例如："Gina 说她昨天（2024 年 3 月 9 日）为她的服装店发起了一场广告营销活动。"
* **不要**简化、改写或合并逻辑上不同的想法。

#### 5. 检索清晰度
* 每个 atomic_fact 必须简洁、真实且自成一体。
* 除非寒暄或填充语传达了有意义的情感或信息内容，否则应避免。
* 确保实体和行为明确无歧义。

#### 6. 输出要求
* **仅**输出 JSON 对象 —— 不要有额外的解释、markdown 或评论。
* 确保 JSON 有效（正确的引号、逗号和转义）。
* `"atomic_fact"` 列表应包含从情节中提取的所有有意义的事实。

---

### 质量检查
在返回最终输出之前，验证：
1. 所有有意义的事实、意图和情感都已包含在内。
2. 每个 `atomic_fact` 只包含一个想法。
3. 所有相关的时间引用都已保留或规范化。
4. 措辞忠实于原文。
5. 输出的 JSON 有效并遵循确切的 schema。

---

### 示例

**输入：**
TIME = "March 10, 2024(Sunday) at 2:00 PM"
EPISODE_TEXT =
"Gina 说她昨天刚为她的服装店发起了一场广告营销活动。
Jon 向她表示祝贺，并询问她的舞蹈工作室寻找进展如何。
Gina 解释说她目前专注于服装店，但仍希望尽快找到合适的工作室。"

**输出：**
{
  "atomic_facts": {
    "time": "March 10, 2024(Sunday) at 2:00 PM",
    "atomic_fact": [
      "Gina 说她昨天（2024 年 3 月 9 日）为她的服装店发起了一场广告营销活动。",
      "Jon 祝贺 Gina 的新营销活动。",
      "Jon 询问了 Gina 舞蹈工作室的寻找进展。",
      "Gina 解释说她目前专注于服装店。",
      "Gina 说她仍然希望将来能找到合适的工作室。"
    ]
  }
}

---

现在请仔细分析提供的 EPISODE_TEXT 和 TIME，应用上述所有规则，并**仅**以指定格式返回 JSON 对象。
---

### 输入
- EPISODE_TEXT: "{{EPISODE_TEXT}}"
- TIME: "{{TIME}}"（情节的开始时间，例如"March 10, 2024(Sunday) at 2:00 PM"）

"""
    + _LANGUAGE_RULE
    + "\n"
)
