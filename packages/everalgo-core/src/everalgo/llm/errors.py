"""LLM-layer error types — minimal single-base set."""


class LLMError(Exception):
    """Raised on any LLM call failure.

    The provider implementation maps native SDK exceptions (e.g. ``openai.RateLimitError`` /
    ``openai.APIConnectionError``) to ``LLMError`` using ``raise LLMError(...) from sdk_err``. Callers wanting
    fine-grained handling can either:

    - Inspect ``e.__cause__`` (PEP 3134) to reach the SDK-native exception, or
    - ``except`` the SDK type directly (the SDK exception is still in the cause chain).

    Future minor bumps may introduce subclasses (``LLMRateLimitError``, ``LLMTimeoutError``, etc.) without
    breaking ``except LLMError`` callers.
    """


class LLMNotConfiguredError(RuntimeError):
    """Raised when no LLM is configured at any of the 3 injection layers.

    Inherits ``RuntimeError`` (NOT ``LLMError``) intentionally — this is a developer misuse error (forgot to
    inject), not a runtime SDK call failure. Mirrors pydantic-ai ``UserError(RuntimeError)``
    (``pydantic_ai_slim/pydantic_ai/exceptions.py:144``); see spec §5 (Exception Family Boundaries) for the
    decision rationale.
    """
