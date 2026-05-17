"""LLM-layer error types — minimal single-base set."""


class LLMError(Exception):
    """Raised on any LLM call failure. Inspect ``__cause__`` (PEP 3134) for the original SDK exception."""
