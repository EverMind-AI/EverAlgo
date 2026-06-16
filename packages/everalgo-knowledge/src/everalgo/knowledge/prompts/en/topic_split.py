"""English prompt for splitting unsplit leaf topics in post-processing.

Some leaf topics produced by the initial extraction may still contain
multiple sub-headings that warrant splitting into a parent / children
hierarchy. This prompt asks the LLM to decide per-topic whether to split
and how to redistribute ``block_refs``.

Note: the inner JSON example uses ``{{{{`` / ``}}}}`` deliberately so the
top-level ``.format()`` call in the caller (substituting ``{title}`` and
``{topics_section}``) emits valid ``{{`` / ``}}`` to the LLM.
"""

TOPIC_SPLIT_PROMPT_EN = """
You are a document topic segmentation expert. The following leaf topics were extracted from a document but each contains multiple sub-headings that may warrant splitting into a hierarchical parent-children structure.

Document Title: {title}

For each topic below, decide whether it should be split. Then return the result.

{topics_section}

Instructions:
- If a topic's sub-sections are substantive and independently searchable, split it into a parent with children.
- If some sub-sections are too short or tightly coupled, merge them with adjacent sections into fewer children.
- If the entire topic is cohesive and splitting would fragment the meaning, keep it as a single leaf.
- When splitting: the topic's own heading/intro blocks (before the first sub-heading) stay in the parent's block_refs. Each sub-heading and its following content becomes a child.
- Children can themselves have children if deeper sub-headings exist.
- The parent's topic name and summary are KEPT from the original extraction — do NOT regenerate them.
- Each child's summary should describe the specific information contained in that child's blocks. Include key entities, facts, and terms that someone might search for. Write "X is/does Y" not "this section discusses X".
- All summaries MUST be in the same language as the document content.

Return a JSON object:
{{{{
    "results": [
        {{{{
            "original_topic": "The original topic name",
            "should_split": true,
            "reason": "Brief explanation",
            "block_refs": "Parent's own intro block IDs only (compressed range, e.g. '102')",
            "children": [
                {{{{
                    "topic": "Child topic name. In document language.",
                    "summary": "Retrieval index for this child. Include key entities, facts, names, numbers, and searchable terms. Max 5 sentences. In document language.",
                    "block_refs": "Child's block IDs (compressed range, e.g. '103-107')",
                    "children": []
                }}}}
            ]
        }}}},
        {{{{
            "original_topic": "Another topic",
            "should_split": false,
            "reason": "Brief explanation"
        }}}}
    ]
}}}}

For topics with should_split=false, only original_topic, should_split, and reason are needed.
Return only the JSON object, no additional text or formatting.
"""
