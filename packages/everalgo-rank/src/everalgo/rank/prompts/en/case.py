"""English prompt for ``rank.case.arank`` LLM rerank.

Derived in the same style as ``SKILL_RERANK_PROMPT_EN`` and ``EPISODIC_RERANK_PROMPT_EN``
(single-instruction framing). Case has no cross-encoder rerank stage in the enterprise
hybrid path (it uses ``vector_anchored`` fusion instead), so this prompt is a parallel
extrapolation rather than a port of an existing instruction.
"""

CASE_RERANK_PROMPT_EN = """Determine whether each agent execution case's task and approach are useful for addressing the user query, preferring cases whose verified successful approach (high `quality_score`) directly applies to the query's problem. Score each case from 0.0 (no transferable experience) to 1.0 (same task pattern with a verified successful approach).

User query:
{query}

Candidates (JSON array; each item has `id`, `score`, plus metadata like `task_intent`, `approach`, `quality_score`, `key_insight`):
{candidates_json}

Return at most {top_k} items, sorted by score descending. Drop candidates whose score would be near 0.

Output strictly the following JSON, no prose:
{{"ranked": [{{"id": "<candidate_id>", "score": <float in 0..1>}}]}}
"""
