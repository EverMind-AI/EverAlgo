"""LLM client configuration."""

from typing import Any

from pydantic import BaseModel, Field, SecretStr


class LLMConfig(BaseModel):
    """OpenAI-compatible LLM client configuration.

    The set of fields mirrors the openai-python SDK's ``AsyncOpenAI`` constructor (``api_key`` / ``base_url`` /
    ``timeout``) plus per-call sampling defaults (``temperature`` / ``max_tokens``) and an ``extra`` bucket for
    provider-specific knobs.

    ``api_key`` is wrapped in ``pydantic.SecretStr`` so that ``repr(config)``, ``config.model_dump()``,
    ``config.model_dump_json()`` and ``config.model_json_schema()`` all mask its value. Provider code must
    explicitly call ``config.api_key.get_secret_value()`` to obtain the raw string before passing it to the SDK
    — that explicit call is itself a safety checkpoint reminding the reader they are touching a credential.
    """

    model: str
    api_key: SecretStr
    base_url: str
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float = 60.0
    extra: dict[str, Any] = Field(default_factory=dict)
