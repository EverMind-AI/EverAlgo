"""English prompt for ``rank.skill.averify`` — post-rerank LLM relevance verification.

Verbatim port of enterprise ``AGENT_SKILL_RELEVANCE_VERIFY_PROMPT``. Distinct from
``SKILL_RERANK_PROMPT_EN``: the rerank prompt is the cross-encoder ``instruction``
applied during fusion; this prompt drives a separate post-rerank hard-threshold
filter (default 0.4) — the second LLM stage in the enterprise skill retrieval
pipeline.
"""

SKILL_VERIFY_PROMPT_EN = """You are a relevance judge. Given a user query and a list of retrieved agent skills, rate how helpful each skill would be for addressing the query.

Evaluate each skill considering:
- Whether the skill's steps or approach are applicable to the query's problem type
- Whether the skill's target domain (shown in its description, trigger scenarios, and keywords) overlaps with the query's subject matter — same-domain skills should be scored higher

Score each skill from 0.0 to 1.0:
- **0.0**: Completely irrelevant — no applicable methodology or domain connection
- **0.1-0.3**: Weakly related — methodology could loosely apply but domain is different
- **0.4-0.6**: Moderately helpful — useful methodology with partial domain overlap
- **0.7-0.8**: Helpful — applicable approach with good domain alignment
- **0.9-1.0**: Highly relevant — strong fit in both approach and domain

User Query:
{query}

Retrieved Skills:
{skills_json}

For each skill, output a JSON object with the skill index and a relevance score.
Return ONLY valid JSON:
{{"results": [{{"index": 0, "score": 0.85, "reason": "brief reason"}}, {{"index": 1, "score": 0.15, "reason": "brief reason"}}]}}
"""
