"""Chinese prompt for ``rank.skill.arank`` LLM rerank.

Anchors on the enterprise hybrid-rerank ``instruction`` passed to the cross-encoder
in ``search_mem_service._search_agent_skills`` — methodology + domain applicability,
same-domain preference, directly-relevant steps. The downstream
``AGENT_SKILL_RELEVANCE_VERIFY_PROMPT`` is a separate post-rerank verification stage
and is intentionally NOT inlined here.
"""

SKILL_RERANK_PROMPT_ZH = """判断每个 agent skill 的方法（methodology）和领域（domain）是否适用于用户查询，优先选择同领域且步骤直接相关的 skill。按 0.0（完全不适用）到 1.0（方法和领域都强适配）给每个 skill 打分。

用户查询：
{query}

候选列表（JSON 数组；每项含 `id`、`score`，以及元数据如 `name` / `description` / `content` / `confidence`）：
{candidates_json}

最多返回 {top_k} 项，按 score 降序排列。分数接近 0 的候选直接丢弃。

只输出以下 JSON，不要包含任何解释文字：
{{"ranked": [{{"id": "<候选 id>", "score": <0..1 浮点数>}}]}}
"""
