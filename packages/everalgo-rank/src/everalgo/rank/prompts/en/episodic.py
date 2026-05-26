"""English prompt for ``rank.episodic.arank`` LLM rerank.

Anchors on the enterprise default cross-encoder ``instruction`` used by
``rerank_deepinfra``/``rerank_vllm`` when no per-call instruction is supplied —
``_search_episodic_memory`` does not pass an ``instruction``, so episodic hybrid
rerank inherits this generic relevance check.
"""

EPISODIC_RERANK_PROMPT_EN = """Given the user query and a list of retrieved episodic memories (or atomic facts extracted from them), determine whether each candidate contains information relevant to answering the query. Score each candidate from 0.0 (no relevant information) to 1.0 (directly answers the query).

User query:
{query}

Candidates (JSON array; each item has `id`, `score`, an `item_type` of either `episode` or `atomic_fact`, plus metadata like `episode` / `subject` / `summary` / `parent_episode_id`):
{candidates_json}

Return at most {top_k} items, sorted by score descending. Drop candidates whose score would be near 0.

Output strictly the following JSON, no prose:
{{"ranked": [{{"id": "<candidate_id>", "score": <float in 0..1>}}]}}
"""
