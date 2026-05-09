"""Factory for building an LLM client from configuration."""

from evercore.llm.config import LLMConfig
from evercore.llm.protocols import LLMClient


def build_client(config: LLMConfig) -> LLMClient:
    """Build an OpenAI-compatible LLM client from ``config``.

    Implementation note: ``OpenAICompatClient`` is imported lazily inside the
    function body so that ``evercore.llm.factory`` itself does not pull the
    ``openai`` SDK at import time. This keeps ``import evercore.llm`` cheap
    for callers that only need the Protocol / Config / Error types and never
    call ``build_client``. Maintainers — please do **not** "optimise" this
    into a top-level import; the laziness is load-bearing.
    """
    from evercore.llm.providers.openai_compat import OpenAICompatClient

    return OpenAICompatClient(config)
