"""Chinese prompt for ``rank.skill.averify`` — post-rerank LLM relevance verification.

Chinese counterpart of ``SKILL_VERIFY_PROMPT_EN``. Drives the post-rerank hard-threshold
filter stage; distinct from the rerank prompt.
"""

SKILL_VERIFY_PROMPT_ZH = """你是相关性裁判。给定一个用户查询和一组检索到的 agent skill 候选，评估每个 skill 对完成该查询的帮助程度。

评估每个 skill 时同时考虑：
- skill 的步骤或方法（steps / approach）是否适用于查询的问题类型
- skill 的目标领域（由 description / 触发场景 / 关键词体现）是否与查询的主题重叠 —— 同领域的 skill 应该打更高分

按 0.0 到 1.0 给每个 skill 打分：
- **0.0**: 完全不相关 —— 方法不适用、领域也不重叠
- **0.1-0.3**: 弱相关 —— 方法勉强能套，但领域不一样
- **0.4-0.6**: 中等有帮助 —— 方法可用，领域部分重叠
- **0.7-0.8**: 有帮助 —— 方法适用且领域对齐良好
- **0.9-1.0**: 高度相关 —— 方法和领域都强匹配

用户查询：
{query}

候选 skill 列表：
{skills_json}

对每个 skill，输出包含 index 和 relevance score 的 JSON 对象。
只返回合法 JSON：
{{"results": [{{"index": 0, "score": 0.85, "reason": "简短理由"}}, {{"index": 1, "score": 0.15, "reason": "简短理由"}}]}}
"""
