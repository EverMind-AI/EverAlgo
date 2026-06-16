"""English prompt for ``KnowledgeExtractor.aextract`` topic-tree segmentation.

Performs the combined extraction LLM call:

* topic segmentation with hierarchy and ``block_refs`` per topic
* document-level summary / subject / keywords / content_labels

A single LLM call output populates both the topic tree and the document root
node (``KnowledgeMemory[0]``).
"""

TOPIC_TREE_EXTRACT_PROMPT_EN = """
You are an expert document analyzer. Your task is to perform TWO analyses on the following document:
1. **Topic Segmentation**: Segment the document into topic-based sections
2. **Overall Analysis**: Extract summary, subject, and keywords

Document Title: {title}
Total paragraphs: {num_paragraphs} (IDs 0 to {max_id})

Paragraphs (format: ID: content):
{numbered_paragraphs}

Return a JSON object with the following structure:
{{
    "language": "Detected language of the document (e.g., 'English', 'Chinese', 'Japanese'). Identify FIRST — all subsequent fields MUST be written in this language.",
    "summary": "A comprehensive but concise summary (2-3 sentences). MUST be in the detected language.",
    "subject": "A clear, descriptive subject/topic. MUST be in the detected language.",
    "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
    "content_labels": ["label1", "label2"],
    "topics": [
        {{
            "topic": "Topic name (concise, max 20 chars). MUST be in the detected language.",
            "summary": "Retrieval index for this ENTIRE section (including children). Include key entities, facts, names, numbers, and searchable terms. Max 5 sentences. MUST be in the detected language.",
            "block_refs": "Intro paragraph IDs only (e.g., '0-2')",
            "content_labels": ["label1"],
            "children": [
                {{
                    "topic": "Sub-topic name. MUST be in the detected language.",
                    "summary": "Retrieval index for this sub-section. Include key entities, facts, names, numbers, and searchable terms. Max 5 sentences. MUST be in the detected language.",
                    "block_refs": "This sub-topic's paragraph IDs",
                    "content_labels": [],
                    "children": [
                        {{
                            "topic": "Sub-sub-topic name (nesting can go deeper if document headings do)",
                            "summary": "...",
                            "block_refs": "...",
                            "content_labels": [],
                            "children": []
                        }}
                    ]
                }}
            ]
        }}
    ]
}}

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
- Every paragraph ID (0 to {max_id}) MUST appear in at least one topic's block_refs. Document metadata (title, version, date, author) goes in the most relevant topic. Only pure separators (e.g., "---") may be omitted.
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
