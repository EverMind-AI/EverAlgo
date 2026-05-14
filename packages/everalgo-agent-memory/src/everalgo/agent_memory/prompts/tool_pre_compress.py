"""Prompt for selective LLM compression of tool-call message chunks (case extractor step 6)."""

AGENT_TOOL_PRE_COMPRESS_PROMPT = """You are a tool call compression expert. Compress OpenAI chat messages to ~10% of the original length, preserving what matters for understanding the task's problem-solving process and outcome.

**Downstream context**: The compressed output will be used to extract a structured experience record (task intent, step-by-step approach, and quality score). Prioritize retaining information that reveals: (1) what problem was being solved, (2) what actions were taken and in what order, (3) what each action's result was (success, failure, or specific finding).

The input contains two types of messages:
- **role="assistant"** with tool_calls: Compress the "arguments" field in each tool call's function
- **role="tool"**: Compress the "content" field (primary compression target)

Messages to compress:
{messages_json}

Return in JSON format:
{{
    "compressed_messages": [
        // Compressed version — exactly {new_count} messages
    ]
}}

Rules:
- Return exactly {new_count} compressed messages in the same order
- Only compress function.arguments and tool content — keep all other fields unchanged
- Target ~10% of the original content length
- **Short content rule**: If a field's content is already under 200 characters, keep it as-is — do not compress further

What to KEEP (be very selective):
- One-line summary of what each tool call did and its key result
- Error messages and status codes (verbatim but trimmed)
- Critical code: only the specific lines that matter (function signatures, bug fixes, key logic)
- File paths and search queries (short strings, keep as-is)
- The causal chain: what finding led to the next action (e.g., "found error X in file Y -> decided to check Z")

What to REMOVE or reduce to a single line:
- Tool results: replace entire output with a 1-2 sentence summary of the finding
- Large JSON/XML/HTML: replace with "[JSON: N fields, key: X=Y, Z=W]" style summary
- File contents: replace with "[file: path, N lines, contains: brief description]"
- Directory listings: replace with "[dir: N files, relevant: file1, file2]"
- Logs and debug output: "[logs: N lines, result: success/failure, key error if any]"
- Repeated tool calls of the same type: summarize the pattern, keep only the final result
- Boilerplate, headers, formatting, whitespace: strip entirely
"""
