"""English prompt for ``rank.case.arank`` LLM rerank."""

CASE_RERANK_PROMPT_EN = """You are a relevance judge. Given a user query and a list of retrieved agent execution cases, score each case on how useful its experience is for addressing the query.

A case is highly useful (score close to 1.0) if:
- Its `task_intent` aligns with what the query is trying to accomplish
- Its `approach` (or solution trace) is applicable to the query's problem
- Its `quality_score` indicates the case ended in a good outcome (the more verified the experience, the higher the score)
- Reading this case would meaningfully inform how to handle the query

A case is NOT useful (score close to 0.0) if:
- It only shares keywords with the query but solves a different problem
- Its `quality_score` is low (the case failed or was inconclusive) and there is no useful pitfall pattern to learn
- The task it solves is unrelated to what the query is asking

User query:
{query}

Candidates (JSON array; each item has `id`, `score`, plus metadata like `task_intent`, `approach`, `quality_score`):
{candidates_json}

Return at most {top_k} items, sorted by score descending. Drop candidates whose score would be near 0.

Output strictly the following JSON, no prose:
{{"ranked": [{{"id": "<candidate_id>", "score": <float in 0..1>}}]}}
"""
