"""English prompt for merging document-level summary / subject / keywords across batches.

When a document is too long for a single LLM call, ``KnowledgeExtractor`` runs
the topic-tree prompt over multiple batches and produces partial document-level
fields. This prompt consolidates them into a single coherent set.
"""

CONTENT_MERGE_PROMPT_EN = """
You are an expert document analyzer. Please merge multiple partial analysis results from different sections of a long document into one comprehensive analysis.

Document Title: {title}
Partial Analysis Results:
```
{partial_results}
```

Please create a comprehensive analysis that merges all the partial results into:
{{
    "language": "The language used in the partial results (e.g., 'English', 'Chinese'). Identify FIRST — all subsequent fields MUST be written in this language.",
    "summary": "A comprehensive summary (3-5 sentences). MUST be in the detected language.",
    "subject": "A unified subject/topic. MUST be in the detected language.",
    "keywords": ["merged list of most important keywords (3-5 total). MUST be in the detected language."]
}}

Requirements:
1. **Summary**: Integrate all key information from partial summaries, remove redundant information while preserving important details, maintain logical flow and coherence
2. **Subject**: Choose or synthesize the most representative subject that captures the main theme of the entire document
3. **Keywords**: Merge and deduplicate keywords from all sections, keeping only the 3-5 most important ones that represent the entire document. The extracted keywords must be in the original document.

**CRITICAL LANGUAGE RULE**: First identify the language in the "language" field. ALL outputs (summary, subject, keywords) MUST be in that language. Do NOT translate.

Return only the JSON object, no additional text or formatting.
"""
