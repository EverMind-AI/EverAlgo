"""Prompt strings for the multimodal parser, organised by language.

Per ``everalgo`` convention (AGENTS.md §5), prompts live as module-level string
constants under ``prompts/{en,zh}/<operator>.py``. Callers select a language
explicitly when wiring a parser; the parser operators import from
``prompts.en`` by default and accept caller-side overrides via the
monkey-patch pattern documented in AGENTS.md.
"""

from everalgo.parser.prompts import en, zh

__all__ = ["en", "zh"]
