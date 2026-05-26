"""Chinese prompts for ``rank.fusion.aagentic_rank`` — sufficiency check + multi-query generation."""

AGENTIC_SUFFICIENCY_CHECK_PROMPT_ZH = """你是记忆检索评估专家。请判断当前检索到的记忆是否足以回答用户的查询。

用户查询：
{query}

检索到的记忆：
{retrieved_docs}

请判断这些记忆是否足以回答用户查询。

输出格式（JSON）：
{{
    "is_sufficient": true/false,
    "reasoning": "你的判断理由",
    "missing_information": ["缺失信息 1", "缺失信息 2"]
}}

要求：
1. 如果记忆中包含回答查询所需的关键信息，判定为充分（true）
2. 如果缺少关键信息，判定为不充分（false），并列出缺失的信息
3. reasoning 应当简明清晰
4. missing_information 仅在不充分时填写，否则为空数组
"""


AGENTIC_MULTI_QUERY_PROMPT_ZH = """你是查询优化专家。用户的原始查询未能检索到足够信息，请生成多个互补的改进查询。

原始查询：
{original_query}

当前检索到的记忆：
{retrieved_docs}

缺失的信息：
{missing_info}

请生成 2-3 个互补的查询，以帮助找到缺失的信息。这些查询应当：
1. 聚焦不同的缺失信息点
2. 使用不同的表达方式
3. 避免与原始查询完全相同
4. 保持简洁清晰

输出格式（JSON）：
{{
    "queries": [
        "改进查询 1",
        "改进查询 2",
        "改进查询 3"
    ],
    "reasoning": "查询生成策略的说明"
}}

要求：
1. queries 数组包含 2-3 条查询
2. 每条查询长度在 5-200 字符之间
3. reasoning 解释生成策略
"""
