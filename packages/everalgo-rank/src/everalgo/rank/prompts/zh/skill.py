"""Chinese prompt for ``rank.skill.arank`` LLM rerank."""

SKILL_RERANK_PROMPT_ZH = """你是相关性裁判。给定一个用户查询和一组检索到的 agent skill 候选，给每个 skill 打分，衡量它对完成查询任务的帮助程度。

如果一个 skill **非常有帮助**（分数接近 1.0），通常满足：
- 提供了可直接应用于该查询的可执行步骤、知识或方法
- 覆盖了查询所属的问题类型（即使关键词没精确匹配）
- 遵循该 skill 能显著提高 agent 处理该查询的能力

如果一个 skill **没有帮助**（分数接近 0.0），通常是因为：
- 只是表面相关（共享关键词但解决的是另一类问题）
- 太宽泛或太局部，对当前查询没有实际价值
- 查询不落在该 skill 的"何时使用"场景里

用户查询：
{query}

候选列表（JSON 数组；每项含 `id`、`score`，以及元数据如 `name` / `description` / `maturity_score` / `confidence`）：
{candidates_json}

最多返回 {top_k} 项，按 score 降序排列。分数接近 0 的候选直接丢弃。

只输出以下 JSON，不要包含任何解释文字：
{{"ranked": [{{"id": "<候选 id>", "score": <0..1 浮点数>}}]}}
"""
