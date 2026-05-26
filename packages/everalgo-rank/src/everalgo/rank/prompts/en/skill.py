"""English prompt for ``rank.skill.arank`` LLM rerank.

Anchors on the enterprise hybrid-rerank ``instruction`` passed to the cross-encoder
in ``search_mem_service._search_agent_skills`` — methodology + domain applicability,
same-domain preference, directly-relevant steps. The downstream
``AGENT_SKILL_RELEVANCE_VERIFY_PROMPT`` is a separate post-rerank verification stage
and is intentionally NOT inlined here.
"""

SKILL_RERANK_PROMPT_EN = """Determine whether each agent skill's methodology and domain are applicable to the user query, preferring same-domain skills with directly relevant steps. Score each skill from 0.0 (not applicable at all) to 1.0 (methodology and domain both strongly apply).

User query:
{query}

Candidates (JSON array; each item has `id`, `score`, plus metadata like `name`, `description`, `content`, `confidence`):
{candidates_json}

Return at most {top_k} items, sorted by score descending. Drop candidates whose score would be near 0.

Output strictly the following JSON, no prose:
{{"ranked": [{{"id": "<candidate_id>", "score": <float in 0..1>}}]}}
"""
