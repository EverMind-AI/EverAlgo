"""English prompt for ``aclassify_category`` document classification.

Single LLM call that picks one category from a caller-supplied taxonomy
(``list[CategorySpec]``). The taxonomy block is rendered as a stable prefix
so providers that support prompt caching can hit the cache across documents
in one ingest run.

The model is asked to output a single JSON object ``{"category_id": "<id>"}``;
caller validates the id against the closed taxonomy and falls back to ``""``
on parse failure or out-of-set responses.
"""

CATEGORY_CLASSIFY_PROMPT_EN = """\
You are a document classifier. Pick exactly ONE category id from the taxonomy below.

# Taxonomy
{taxonomy}

# Document
Title: {title}
Summary: {doc_summary}

# Task
Return a JSON object with a single key ``category_id`` whose value is the id of the best
matching category from the taxonomy above. If none of the categories fit, return the empty
string. Do NOT invent ids that are not in the taxonomy.

Output strictly this JSON shape and nothing else:
{{"category_id": "<one of the ids listed in the taxonomy, or empty string>"}}
"""
