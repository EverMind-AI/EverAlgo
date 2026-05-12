"""Factory for building an LLM client from configuration."""

from everalgo.llm.config import LLMConfig
from everalgo.llm.protocols import LLMClient


def build_client(config: LLMConfig) -> LLMClient:
    """Build an OpenAI-compatible LLM client from ``config``.

    Implementation note: ``OpenAICompatClient`` is imported lazily inside the function body so that
    ``everalgo.llm.factory`` itself does not pull the ``openai`` SDK at import time. This keeps
    ``import everalgo.llm`` cheap for callers that only need the Protocol / Config / Error types and never call
    ``build_client``. Maintainers — please do **not** "optimise" this into a top-level import; the laziness is
    load-bearing.
    """
    from everalgo.llm.providers.openai_compat import OpenAICompatClient

    return OpenAICompatClient(config)
