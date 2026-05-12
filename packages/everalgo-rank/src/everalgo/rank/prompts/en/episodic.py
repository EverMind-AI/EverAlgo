"""English prompt for ``rank.episodic.arank`` LLM rerank."""

EPISODIC_RERANK_PROMPT_EN = """You are a relevance judge. Given a user query and a list of retrieved episodic memories (or atomic facts extracted from them), score each candidate on how directly it answers or evidences the query.

A candidate is highly relevant (score close to 1.0) if:
- It captures the event, time period, participants, or decision the query asks about
- Its episode text (or atomic_fact text) explicitly mentions the entities or actions in the query
- Reading this memory would let the agent answer or react to the query without further lookup

A candidate is NOT relevant (score close to 0.0) if:
- It only shares keywords with the query but describes a different event
- Its time window or participants are unrelated to what the query is about
- It is too generic to support any specific answer to the query

User query:
{query}

Candidates (JSON array; each item has `id`, `score`, an `item_type` of either `episode` or `atomic_fact`, plus metadata like `episode` / `subject` / `summary` / `parent_episode_id`):
{candidates_json}

Return at most {top_k} items, sorted by score descending. Drop candidates whose score would be near 0.

Output strictly the following JSON, no prose:
{{"ranked": [{{"id": "<candidate_id>", "score": <float in 0..1>}}]}}
"""
