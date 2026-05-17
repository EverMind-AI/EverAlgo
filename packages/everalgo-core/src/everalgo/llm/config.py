"""LLM client configuration."""

from typing import Any

from pydantic import BaseModel, Field, SecretStr


class LLMConfig(BaseModel):
    """OpenAI-compatible LLM client configuration.

    ``api_key`` uses ``SecretStr`` so repr / serialization never leaks the raw key; call
    ``config.api_key.get_secret_value()`` explicitly when passing it to an SDK.
    """

    model: str
    api_key: SecretStr
    base_url: str
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float = 60.0
    extra: dict[str, Any] = Field(default_factory=dict)
