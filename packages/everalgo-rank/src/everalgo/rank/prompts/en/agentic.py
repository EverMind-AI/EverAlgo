"""English prompts for ``rank.fusion.aagentic_rank`` — sufficiency check + multi-query generation.

Re-exposed as module-level constants so callers can override per call
(``sufficiency_prompt=`` / ``multi_query_prompt=``) or monkey-patch globally (AGENTS.md §5).
"""

AGENTIC_SUFFICIENCY_CHECK_PROMPT_EN = """You are a memory retrieval evaluation expert. Please assess whether the currently retrieved memories are sufficient to answer the user's query.

User query:
{query}

Retrieved memories:
{retrieved_docs}

Please determine whether these memories are sufficient to answer the user's query.

Output format (JSON):
{{
    "is_sufficient": true/false,
    "reasoning": "Your reasoning for the judgment",
    "missing_information": ["Missing information 1", "Missing information 2"]
}}

Requirements:
1. If the memories contain key information needed to answer the query, judge as sufficient (true)
2. If key information is missing, judge as insufficient (false), and list the missing information
3. reasoning should be concise and clear
4. missing_information should only be filled when insufficient, otherwise empty array
"""


AGENTIC_MULTI_QUERY_PROMPT_EN = """You are a query optimization expert. The user's original query failed to retrieve sufficient information; please generate multiple complementary improved queries.

Original query:
{original_query}

Currently retrieved memories:
{retrieved_docs}

Missing information:
{missing_info}

Please generate 2-3 complementary queries to help find the missing information. These queries should:
1. Focus on different missing information points
2. Use different expressions
3. Avoid being identical to the original query
4. Remain concise and clear

Output format (JSON):
{{
    "queries": [
        "Improved query 1",
        "Improved query 2",
        "Improved query 3"
    ],
    "reasoning": "Explanation of query generation strategy"
}}

Requirements:
1. queries array contains 2-3 queries
2. Each query length between 5-200 characters
3. reasoning explains the generation strategy
"""
