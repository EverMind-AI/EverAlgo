"""English prompt for ``rank.skill.arank`` LLM rerank.

Anchors on the enterprise hybrid-rerank ``instruction`` passed to the cross-encoder in
``search_mem_service._search_agent_skills`` — methodology + domain applicability, same-domain
preference, directly-relevant steps. The detailed 0.0-1.0 scoring bands previously carried by the
separate post-rerank verify stage (``AGENT_SKILL_RELEVANCE_VERIFY_PROMPT``) are folded in here so a
single LLM pass both reorders and quality-grades candidates — the standalone verify stage is gone.
"""

SKILL_RERANK_PROMPT_EN = """Determine whether each agent skill's methodology and domain are applicable to the user query, preferring same-domain skills with directly relevant steps. Score each skill from 0.0 (not applicable at all) to 1.0 (methodology and domain both strongly apply).

Evaluate each skill considering:
- Whether the skill's steps or approach are applicable to the query's problem type
- Whether the skill's target domain (shown in its description, trigger scenarios, and keywords) overlaps with the query's subject matter — same-domain skills should be scored higher

Score bands:
- **0.0**: Completely irrelevant — no applicable methodology or domain connection
- **0.1-0.3**: Weakly related — methodology could loosely apply but domain is different
- **0.4-0.6**: Moderately helpful — useful methodology with partial domain overlap
- **0.7-0.8**: Helpful — applicable approach with good domain alignment
- **0.9-1.0**: Highly relevant — strong fit in both approach and domain

User query:
{query}

Candidates (JSON array; each item has `id`, `score`, plus metadata like `name`, `description`, `content`, `confidence`):
{candidates_json}

Return at most {top_k} items, sorted by score descending. Drop candidates whose score would be near 0.

Output strictly the following JSON, no prose:
{{"ranked": [{{"id": "<candidate_id>", "score": <float in 0..1>}}]}}
"""
