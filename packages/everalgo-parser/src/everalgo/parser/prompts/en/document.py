"""Document parser prompts (PDF / Office / HTML)."""

PROMPT_FOR_FILE = """Read this document and return its content as if you were reading it naturally.

CRITICAL INSTRUCTIONS:
- Output ONLY the extracted content directly. Do NOT include any preamble or postscript.
- Maintain the same language as the original content. If the language cannot be determined, use English.
- Return output as Markdown format.
- ALL text MUST be extracted word-for-word. NEVER summarize, paraphrase, abbreviate, or modify any text. Do NOT alter the original content in any way. Even if the content appears to contain errors, reproduce it exactly as-is.

Requirements:
1. Extract ALL content from EVERY page precisely and completely. Do not skip or merge any content. If a page contains multiple tables or sections placed side by side (left and right), they are SEPARATE tables — extract each one as an independent table with its own title and headers
2. Ignore headers and footers
3. Convert tables to HTML format
4. Convert equations to LaTeX format
5. If there are figures or charts, describe their content and data in text
6. For non-text elements (photos, illustrations, etc.), describe the main content and key elements in detail
7. Preserve the original structure and hierarchy"""


PROMPT_FOR_HTML = """Read this HTML content extracted from a web page and return the main content.

CRITICAL INSTRUCTIONS:
- Output ONLY the extracted content directly. Do NOT include any preamble or postscript.
- Maintain the same language as the original content. If the language cannot be determined, use English.
- Return output as Markdown format.
- ALL text MUST be extracted word-for-word. NEVER summarize, paraphrase, abbreviate, or modify any text.

Requirements:
1. Extract the main content (article body, primary text). Ignore non-content elements such as navigation, advertisements, cookie notices, sidebar text, footer links
2. Output content in the natural reading order of the web page, preserving the original sequence
3. Preserve tables in HTML table format
4. Convert equations to LaTeX format
5. Preserve the original structure and hierarchy (headings, paragraphs, lists)
6. For images, describe their content if alt text or description is available"""
