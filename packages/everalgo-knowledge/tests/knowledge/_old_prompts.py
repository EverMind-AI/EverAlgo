"""The six knowledge prompts as they were before the system / user split, pinned verbatim.

Each constant is the source text of ``git show c00ebe8:packages/everalgo-knowledge/src/everalgo/knowledge/prompts/en/<module>.py``;
the split tests check every line of it against the new pair.
"""

from __future__ import annotations

__all__ = [
    "OLD_CATEGORY_CLASSIFY",
    "OLD_CONTENT_MERGE",
    "OLD_TOPIC_MERGE",
    "OLD_TOPIC_SPLIT",
    "OLD_TOPIC_TREE_EXTRACT",
    "OLD_UNCOVERED_ASSIGN",
]

# ``TOPIC_TREE_EXTRACT_PROMPT_EN``
OLD_TOPIC_TREE_EXTRACT = """
You are an expert document analyzer. Your task is to perform TWO analyses on the following document:
1. **Topic Segmentation**: Segment the document into topic-based sections
2. **Overall Analysis**: Extract summary, subject, and keywords

Document Title: {{title}}
Total paragraphs: {{num_paragraphs}} (IDs 0 to {{max_id}})

Paragraphs (format: ID: content):
{{numbered_paragraphs}}

Return a JSON object with the following structure:
{
    "language": "Detected language of the document (e.g., 'English', 'Chinese', 'Japanese'). Identify FIRST — all subsequent fields MUST be written in this language.",
    "summary": "A comprehensive but concise summary (2-3 sentences). MUST be in the detected language.",
    "subject": "A clear, descriptive subject/topic. MUST be in the detected language.",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "content_labels": ["label1", "label2"],
    "topics": [
        {
            "topic": "Topic name (concise, max 20 chars). MUST be in the detected language.",
            "summary": "Retrieval index for this ENTIRE section (including children). Include key entities, facts, names, numbers, and searchable terms. Max 5 sentences. MUST be in the detected language.",
            "block_refs": "Intro paragraph IDs only (e.g., '0-2')",
            "content_labels": ["label1"],
            "children": [
                {
                    "topic": "Sub-topic name. MUST be in the detected language.",
                    "summary": "Retrieval index for this sub-section. Include key entities, facts, names, numbers, and searchable terms. Max 5 sentences. MUST be in the detected language.",
                    "block_refs": "This sub-topic's paragraph IDs",
                    "content_labels": [],
                    "children": [
                        {
                            "topic": "Sub-sub-topic name (nesting can go deeper if document headings do)",
                            "summary": "...",
                            "block_refs": "...",
                            "content_labels": [],
                            "children": []
                        }
                    ]
                }
            ]
        }
    ]
}

Topic Segmentation Requirements:

**Hierarchy**:
- Segment by **semantic boundaries** — each topic should cover one coherent subject. Use headings as a guide, but also split within a section if it covers multiple distinct subjects (e.g., a section covering both "deployment topology" and "disaster recovery" should become two topics even without sub-headings).
- If the document has markdown headings (##, ###, ####, etc.), use them as the primary structural guide. Map heading levels to nesting levels (## -> parent, ### -> child, #### -> grandchild). Each sub-heading becomes its own child topic.
- If a section has no sub-headings but covers multiple distinct subjects, split it into sibling topics at the semantic boundaries.
- If the document has no headings at all, use your judgment to identify natural topic boundaries and decide whether nesting is appropriate.
- If headings are malformed (skipped levels, inconsistent numbering, misleading titles), use the content to determine the correct hierarchy rather than blindly following the heading markup.
- Boundary blocks (transitional paragraphs between topics) may appear in both adjacent topics.
- Parent block_refs contain ONLY its own intro blocks (before the first child begins). Child headings belong in the child's block_refs, not the parent's.
- children key is optional — omit or use [] for leaf topics.
- Every summary serves as a retrieval index — someone reading only the summary should be able to decide whether this section answers their query. Include all key entities, facts, names, numbers, and terms that someone might search for. Be concise but do not omit searchable information. Max 5 sentences.
- A parent's summary covers the ENTIRE section including all children's content.
- A leaf's summary covers only its own paragraphs.

**Coverage — every block must be assigned**:
- Every paragraph ID (0 to {{max_id}}) MUST appear in at least one topic's block_refs. Document metadata (title, version, date, author) goes in the most relevant topic. Only pure separators (e.g., "---") may be omitted.
- Overlap is allowed at boundaries — a transitional paragraph may appear in two adjacent topics (including parent and child).
- Transitional content (questions, acknowledgments, topic shifts) must be assigned, not left out.
- Use compressed range notation: "1-3,5,8-10" means [1,2,3,5,8,9,10]. Topics do not need to be contiguous.

**Naming and content**:
- Topic names should be specific (e.g., "Introduction to Quantum Tunneling" not just "Introduction").
- A heading introduces the content that FOLLOWS it — assign it to the same topic as the body below.
- Tables, lists, and image descriptions are regular content — group with the nearest related topic.
- Trailing attribution (bylines, credits) belongs to the preceding section's topic.

Overall Analysis Requirements:
- **Summary**: Should help someone quickly understand what this document is about and why it might be relevant. Include main topics, key decisions, important information, and overall purpose. Be concise but comprehensive.
- **Subject**: A clear, descriptive subject that captures the main theme/topic of the document. This should be better than the original title if the title is unclear or generic.
- **Keywords**: Extract 3-5 most important keywords that represent the core concepts, topics, or themes in the document. The extracted keywords must be in the original document.

Content Labels (apply to both document-level and each topic):
- Detect and tag content with applicable labels from: "violence", "adult", "hate_speech", "personal_info", "financial", "medical"
- "personal_info": contains non-public PII that could directly identify a private individual, such as personal phone numbers, government ID numbers (SSN, passport), personal email addresses, or home addresses. Do NOT apply for publicly disclosed information like corporate officer names, company addresses, or information in public filings/reports.
- "financial": contains financial data such as bank account numbers, credit card numbers, transaction amounts
- "medical": contains medical/health information
- Return an empty list if no labels apply. Only include labels that clearly match.

**CRITICAL LANGUAGE RULE**:
- First, identify the language of the document content and write it in the "language" field
- ALL subsequent outputs (summary, subject, keywords, topic names, topic summaries) MUST be in that detected language
- Do NOT translate the document content into another language under any circumstances

Return only the JSON object, no additional text or formatting.
"""

# ``CONTENT_MERGE_PROMPT_EN``
OLD_CONTENT_MERGE = """
You are an expert document analyzer. Please merge multiple partial analysis results from different sections of a long document into one comprehensive analysis.

Document Title: {{title}}
Partial Analysis Results:
```
{{partial_results}}
```

Please create a comprehensive analysis that merges all the partial results into:
{
    "language": "The language used in the partial results (e.g., 'English', 'Chinese'). Identify FIRST — all subsequent fields MUST be written in this language.",
    "summary": "A comprehensive summary (3-5 sentences). MUST be in the detected language.",
    "subject": "A unified subject/topic. MUST be in the detected language.",
    "keywords": ["merged list of most important keywords (3-5 total). MUST be in the detected language."]
}

Requirements:
1. **Summary**: Integrate all key information from partial summaries, remove redundant information while preserving important details, maintain logical flow and coherence
2. **Subject**: Choose or synthesize the most representative subject that captures the main theme of the entire document
3. **Keywords**: Merge and deduplicate keywords from all sections, keeping only the 3-5 most important ones that represent the entire document. The extracted keywords must be in the original document.

**CRITICAL LANGUAGE RULE**: First identify the language in the "language" field. ALL outputs (summary, subject, keywords) MUST be in that language. Do NOT translate.

Return only the JSON object, no additional text or formatting.
"""

# ``TOPIC_MERGE_PROMPT_EN``
OLD_TOPIC_MERGE = """
You are an expert document analyzer. A long document was analyzed in multiple sections independently, producing separate topic lists. Some topics may be **true duplicates** — different sections describing the exact same specific event or subject — and should be merged.

Document Title: {{title}}

All topics (each with an index number and optional parent_topic for context):
```
{{all_topics_json}}
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
{
    "merges": [
        {
            "indices": [0, 33],
            "topic": "Merged topic name",
            "summary": "Merged summary"
        }
    ]
}

Return only the JSON object, no additional text or formatting.
"""

# ``CATEGORY_CLASSIFY_PROMPT_EN``
OLD_CATEGORY_CLASSIFY = """\
You are a document classifier. Pick exactly ONE category id from the taxonomy below.

# Taxonomy
{{taxonomy}}

# Document
Title: {{title}}
Summary: {{doc_summary}}

# Task
Return a JSON object with a single key ``category_id`` whose value is the id of the best
matching category from the taxonomy above. If none of the categories fit, return the empty
string. Do NOT invent ids that are not in the taxonomy.

Output strictly this JSON shape and nothing else:
{"category_id": "<one of the ids listed in the taxonomy, or empty string>"}
"""

# ``TOPIC_SPLIT_PROMPT_EN``
OLD_TOPIC_SPLIT = """
You are a document topic segmentation expert. The following leaf topics were extracted from a document but each contains multiple sub-headings that may warrant splitting into a hierarchical parent-children structure.

Document Title: {{title}}

For each topic below, decide whether it should be split. Then return the result.

{{topics_section}}

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
{{
    "results": [
        {{
            "original_topic": "The original topic name",
            "should_split": true,
            "reason": "Brief explanation",
            "block_refs": "Parent's own intro block IDs only (compressed range, e.g. '102')",
            "children": [
                {{
                    "topic": "Child topic name. In document language.",
                    "summary": "Retrieval index for this child. Include key entities, facts, names, numbers, and searchable terms. Max 5 sentences. In document language.",
                    "block_refs": "Child's block IDs (compressed range, e.g. '103-107')",
                    "children": []
                }}
            ]
        }},
        {{
            "original_topic": "Another topic",
            "should_split": false,
            "reason": "Brief explanation"
        }}
    ]
}}

For topics with should_split=false, only original_topic, should_split, and reason are needed.
Return only the JSON object, no additional text or formatting.
"""

# ``UNCOVERED_ASSIGN_PROMPT_EN``
OLD_UNCOVERED_ASSIGN = """
You are a document topic segmentation expert. The following blocks were not assigned to any topic during extraction. Assign each block to the most appropriate existing topic.

Document Title: {{title}}

Existing topics:
{{topics_text}}

Uncovered blocks:
{{uncovered_text}}

Instructions:
- Assign each block to the topic it most naturally belongs to, based on content proximity and semantic relevance.
- Use the topic index (0-based) from the list above.
- If a block truly does not fit any topic, you may assign it to the nearest preceding topic.

Return a JSON object:
{{
    "assignments": [
        {{
            "block_id": 10,
            "topic_index": 0,
            "reason": "Brief explanation"
        }}
    ]
}}

Return only the JSON object, no additional text or formatting.
"""
