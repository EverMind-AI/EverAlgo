"""Factory for building an LLM client from configuration."""

from everalgo.llm.config import LLMConfig
from everalgo.llm.protocols import LLMClient


def build_client(config: LLMConfig) -> LLMClient:
    """Build an OpenAI-compatible LLM client.

    ``OpenAICompatClient`` is imported lazily so that ``import everalgo.llm`` remains cheap for callers
    that only need the Protocol / Config / Error types. Do not hoist this to a top-level import.
    """
    from everalgo.llm.providers.openai_compat import OpenAICompatClient

    return OpenAICompatClient(config)
