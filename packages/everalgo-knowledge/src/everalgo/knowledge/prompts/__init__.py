"""Knowledge extractor prompts (en + zh).

Each prompt is a module-level Python string constant per design.md §1.4 (no external .md / .yaml /
.toml prompt stores). Algorithm authors customize via per-call ``prompt=`` argument or by
monkey-patching the constant at startup.
"""
