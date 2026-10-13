"""Knowledge extractor prompts (English).

Each prompt is a pair of module-level string constants per design.md §1.4 (no external .md / .yaml / .toml prompt
stores): ``*_SYSTEM_PROMPT`` holds the static rules and ``*_USER_PROMPT`` the data the operator renders. Algorithm
authors replace the rules per call via the operator's ``prompt=`` argument; the data message is always the operator's.
"""
