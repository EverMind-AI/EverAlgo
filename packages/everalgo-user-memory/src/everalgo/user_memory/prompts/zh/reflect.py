"""Chinese prompts for EpisodeReflector.

Constants:
    - ``REFLECT_EPISODE_PROMPT`` — full merge from N chronological episodes. Placeholder: ``{timeline}``.
    - ``REFLECT_EPISODE_UPDATE_PROMPT`` — incremental update of an existing narrative.
      Placeholders: ``{old_episode}`` / ``{new_episodes}``.

Output schema (both variants): ``{"content": str, "title": str}`` via Structured Output.

Both variants merge already-extracted episodes, so they inherit the language those episodes were written
in rather than judging it: the mixed-input judgement belongs to the extractor that read the raw
conversation. See ``prompts/zh/episode.py`` for that judgement.
"""

REFLECT_EPISODE_PROMPT = """\
**关键语言规则**：你必须使用与正在合并的情节记忆**相同**的语言输出全部内容。合并操作绝不改变语言——不要翻译。此规则强制执行。

你是一个记忆整合助手。

下面是关于同一主题的若干情节记忆摘要，按时间顺序排列。
每条情节记忆都是在某个时间点、在有限的上下文中写成的。现在你可以看到完整的时间线，
因此能产出一段比任何单条情节记忆都更准确、更完整的叙述。

请将它们合并为一段连贯的叙述，要求：
- 保留**所有**事实细节：姓名、日期、地点、具体行为、数量以及状态变化
- 出现矛盾时，以最新的状态为准
- 保持时间顺序，并保留日期
- 完全按情节记忆中原有的写法保留每一个时间。凡是给出钟点的绝对时间**必须**带 UTC 时区标识（写"2024-03-14 15:00 UTC"，不得写"2024-03-14 15:00"，也不得只写"15:00"）；不含钟点的日期无需标识。**不要**改写、换算或丢弃情节记忆中已有的时间——任何一条情节记忆都不得在合并过程中失去其时间
- 删除重复信息
- 结尾给出截至最新一条情节记忆时的当前状态简述

**关键语言规则**：你必须使用与正在合并的情节记忆**相同**的语言输出全部内容。合并操作绝不改变语言——不要翻译。此规则强制执行。

情节记忆：
{timeline}"""

REFLECT_EPISODE_UPDATE_PROMPT = """\
**关键语言规则**：你必须使用与正在更新的已有叙述**相同**的语言输出全部内容。更新操作绝不改变语言——即使新的情节记忆使用了不同的语言，也不要翻译。此规则强制执行。

你正在用新信息更新一段已有的记忆叙述。

当前叙述：
{old_episode}

新的情节记忆（按时间顺序排列）：
{new_episodes}

请更新该叙述以纳入新信息：
- 修正任何已经过时的表述
- 将新事件按时间顺序插入到对应位置
- 保留仍然准确的内容
- 保持所有事实细节：姓名、日期、地点、具体行为
- 完全按已有叙述和新情节记忆中原有的写法保留每一个时间。凡是给出钟点的绝对时间**必须**带 UTC 时区标识（写"2024-03-14 15:00 UTC"，不得写"2024-03-14 15:00"，也不得只写"15:00"）；不含钟点的日期无需标识。**不要**改写、换算或丢弃已有的时间
- 结尾给出更新后的当前状态简述

**关键语言规则**：你必须使用与正在更新的已有叙述**相同**的语言输出全部内容。更新操作绝不改变语言——即使新的情节记忆使用了不同的语言，也不要翻译。此规则强制执行。"""
