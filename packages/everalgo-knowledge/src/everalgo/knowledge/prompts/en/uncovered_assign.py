"""English prompt for assigning uncovered blocks back to existing topics.

After topic extraction + post-processing, some paragraph IDs may remain
unassigned. This prompt asks the LLM to attach each orphan block to the
most appropriate existing topic by ``topic_index``.

Note: the inner JSON example uses ``{{{{`` / ``}}}}`` deliberately so the
top-level ``.format()`` call in the caller (substituting ``{title}`` /
``{topics_text}`` / ``{uncovered_text}``) emits valid ``{{`` / ``}}`` to
the LLM.
"""

UNCOVERED_ASSIGN_PROMPT_EN = """
You are a document topic segmentation expert. The following blocks were not assigned to any topic during extraction. Assign each block to the most appropriate existing topic.

Document Title: {title}

Existing topics:
{topics_text}

Uncovered blocks:
{uncovered_text}

Instructions:
- Assign each block to the topic it most naturally belongs to, based on content proximity and semantic relevance.
- Use the topic index (0-based) from the list above.
- If a block truly does not fit any topic, you may assign it to the nearest preceding topic.

Return a JSON object:
{{{{
    "assignments": [
        {{{{
            "block_id": 10,
            "topic_index": 0,
            "reason": "Brief explanation"
        }}}}
    ]
}}}}

Return only the JSON object, no additional text or formatting.
"""
