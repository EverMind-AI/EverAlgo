"""Chinese prompt for ``rank.episodic.arank`` LLM rerank."""

EPISODIC_RERANK_PROMPT_ZH = """你是相关性裁判。给定一个用户查询和一组检索到的 episodic 记忆候选（其中可能混入了由 episode 展开出的 atomic_fact 项），给每个候选打分，衡量它**直接回答或佐证**该查询的程度。

如果一个候选**强相关**（分数接近 1.0），通常满足：
- 它捕获了查询所问的事件、时间段、参与者或决策
- 它的 episode 文本（或 atomic_fact 文本）显式提到了查询里的实体或动作
- 阅读这条记忆，agent 不需要再查别处就能回答或响应查询

如果一个候选**不相关**（分数接近 0.0），通常是因为：
- 仅在关键词上与查询有交集，但描述的是另一个事件
- 时间窗口或参与者与查询所问的对不上
- 太泛泛，没法支撑任何针对该查询的具体回答

用户查询：
{query}

候选列表（JSON 数组；每项含 `id`、`score`、`item_type` 为 `episode` 或 `atomic_fact`，以及元数据如 `episode` / `subject` / `summary` / `parent_episode_id`）：
{candidates_json}

最多返回 {top_k} 项，按 score 降序排列。分数接近 0 的候选直接丢弃。

只输出以下 JSON，不要包含任何解释文字：
{{"ranked": [{{"id": "<候选 id>", "score": <0..1 浮点数>}}]}}
"""
