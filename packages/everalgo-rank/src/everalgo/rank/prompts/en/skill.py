"""English prompt for ``rank.skill.arank`` LLM rerank."""

SKILL_RERANK_PROMPT_EN = """You are a relevance judge. Given a user query and a list of retrieved agent skills, score each skill on how helpful it is for addressing the query.

A skill is highly helpful (score close to 1.0) if:
- It provides actionable steps, knowledge, or methodology that directly applies to the query
- It covers a problem class that the query falls into (even if not an exact keyword match)
- Following this skill would meaningfully improve the agent's ability to handle the query

A skill is NOT helpful (score close to 0.0) if:
- It is only superficially related (shares keywords but solves a different problem)
- It is too generic or too specific to be useful for this particular query
- The query does not fall within the skill's "When to use" scenarios

User query:
{query}

Candidates (JSON array; each item has `id`, `score`, plus metadata like `name`, `description`, `maturity_score`, `confidence`):
{candidates_json}

Return at most {top_k} items, sorted by score descending. Drop candidates whose score would be near 0.

Output strictly the following JSON, no prose:
{{"ranked": [{{"id": "<candidate_id>", "score": <float in 0..1>}}]}}
"""
