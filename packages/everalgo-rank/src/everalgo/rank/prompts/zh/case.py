"""Chinese prompt for ``rank.case.arank`` LLM rerank."""

CASE_RERANK_PROMPT_ZH = """你是相关性裁判。给定一个用户查询和一组检索到的 agent 执行案例（case）候选，给每个 case 打分，衡量它的过往经验对当前查询的参考价值。

如果一个 case **非常有用**（分数接近 1.0），通常满足：
- 它的 `task_intent` 与查询想要完成的任务对齐
- 它的 `approach`（或解题轨迹）适用于查询面对的问题
- 它的 `quality_score` 表明该 case 最终是好结果（经验越被验证，分数越高）
- 阅读该 case 能切实指导 agent 如何处理这次查询

如果一个 case **没用**（分数接近 0.0），通常是因为：
- 仅在关键词上与查询有交集，但实际解决的是另一类问题
- 它的 `quality_score` 偏低（案例失败或无定论），且没有可借鉴的失败模式
- 它所解决的任务与查询所问的不相关

用户查询：
{query}

候选列表（JSON 数组；每项含 `id`、`score`，以及元数据如 `task_intent` / `approach` / `quality_score`）：
{candidates_json}

最多返回 {top_k} 项，按 score 降序排列。分数接近 0 的候选直接丢弃。

只输出以下 JSON，不要包含任何解释文字：
{{"ranked": [{{"id": "<候选 id>", "score": <0..1 浮点数>}}]}}
"""
