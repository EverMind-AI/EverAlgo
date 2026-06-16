"""English prompt for merging duplicate topics across batch boundaries.

When a long document is segmented in multiple batches, the same conceptual
topic can show up as separate entries in different batches. This prompt asks
the LLM to identify *true* duplicates (same specific subject, not merely
related themes) and produce a merged topic name + summary.
"""

TOPIC_MERGE_PROMPT_EN = """
You are an expert document analyzer. A long document was analyzed in multiple sections independently, producing separate topic lists. Some topics may be **true duplicates** — different sections describing the exact same specific event or subject — and should be merged.

Document Title: {title}

All topics (each with an index number and optional parent_topic for context):
```
{all_topics_json}
```

Instructions:
1. Scan topics by name, summary, and parent_topic context to find **true duplicates only**: topics that describe the exact same specific event, scene, or subject matter with different wording (e.g. "Elizabeth's Death" and "Elizabeth's Murder" both describing the same murder scene, or two copyright notices at the start and end of a document). Topics with the same parent_topic are more likely to be duplicates if their summaries are semantically similar.
2. Do NOT merge topics just because they share a broad theme, involve the same characters, or are temporally adjacent. Different events are different topics.
3. For each group of duplicates, provide:
   - "indices": the index numbers of the topics to merge
   - "topic": the best topic name (max 20 characters, use the SAME LANGUAGE as the input)
   - "summary": a coherent 1-2 sentence summary covering all merged topics (use the SAME LANGUAGE as the input)
4. If no duplicates are found, return an empty merges list.

Return a JSON object:
{{
    "merges": [
        {{
            "indices": [0, 33],
            "topic": "Merged topic name",
            "summary": "Merged summary"
        }}
    ]
}}

Return only the JSON object, no additional text or formatting.
"""
