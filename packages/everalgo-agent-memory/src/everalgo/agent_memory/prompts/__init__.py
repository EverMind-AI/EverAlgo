"""Agent-memory extractor prompts — single multilingual prompt set.

Each prompt is a module-level Python string constant per AGENTS.md §5 (Code Style). Prompts are language-agnostic
(each carries a ``CRITICAL LANGUAGE RULE`` instruction telling the LLM to mirror the input language in its
output). Algorithm authors customize via per-call ``prompt_*=`` argument or by monkey-patching the constant
at startup.
"""
