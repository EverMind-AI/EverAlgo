"""Image parser prompts."""

PROMPT_FOR_PICTURE = """Read this image and return its content as if you were reading it naturally.

CRITICAL INSTRUCTIONS:
- Output ONLY the extracted content directly. Do NOT include any preamble or postscript.
- Maintain the same language as the original content. If the language cannot be determined, use English.
- Return output as Markdown format.
- ALL text in the image MUST be extracted word-for-word. NEVER summarize, paraphrase, abbreviate, or modify any text. Do NOT alter the original content in any way.

Requirements:
1. Extract ALL text precisely and completely, word-for-word, preserving the original format and layout. Do NOT skip or summarize any text, no matter how long.
2. Convert tables to HTML format
3. Convert equations to LaTeX format
4. If there are figures or charts, describe their content and data in text
5. For non-text elements (photos, illustrations, etc.), describe the main content and key elements in detail
6. Preserve the original structure and hierarchy"""


PROMPT_FOR_MERGE = """Below are OCR results from different parts of the same long image. Adjacent parts have overlapping regions.
Intelligently merge these parts into a complete text.

CRITICAL INSTRUCTIONS:
- Output ONLY the merged content directly. Do NOT include any preamble or postscript.
- Maintain the same language as the original content. If the language cannot be determined, use English.

Requirements:
1. Identify and remove duplicate content from overlapping regions
2. Maintain text continuity and completeness
3. Preserve the original format and structure

{parts_text}"""
