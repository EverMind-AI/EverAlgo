"""Tests for everalgo.llm.config.LLMConfig."""

import pytest
from pydantic import SecretStr, ValidationError

from everalgo.llm.config import LLMConfig


def test_llm_config_minimum_required_fields() -> None:
    cfg = LLMConfig(
        model="gpt-4o-mini",
        api_key="sk-real-secret",  # type: ignore[arg-type]
        base_url="https://api.openai.com/v1",
    )
    assert cfg.model == "gpt-4o-mini"
    assert cfg.base_url == "https://api.openai.com/v1"


def test_llm_config_default_field_values() -> None:
    cfg = LLMConfig(
        model="m",
        api_key="k",  # type: ignore[arg-type]
        base_url="u",
    )
    assert cfg.temperature == 0.0
    assert cfg.max_tokens is None
    assert cfg.timeout == 60.0
    assert cfg.extra == {}


def test_llm_config_api_key_is_secret_str() -> None:
    cfg = LLMConfig(model="m", api_key="sk-secret", base_url="u")  # type: ignore[arg-type]
    assert isinstance(cfg.api_key, SecretStr)


def test_llm_config_repr_masks_api_key() -> None:
    cfg = LLMConfig(model="m", api_key="sk-real-secret", base_url="u")  # type: ignore[arg-type]
    assert "sk-real-secret" not in repr(cfg)


def test_llm_config_model_dump_masks_api_key() -> None:
    cfg = LLMConfig(model="m", api_key="sk-real-secret", base_url="u")  # type: ignore[arg-type]
    dumped = cfg.model_dump()
    assert dumped["api_key"] != "sk-real-secret"


def test_llm_config_model_dump_json_masks_api_key() -> None:
    cfg = LLMConfig(model="m", api_key="sk-real-secret", base_url="u")  # type: ignore[arg-type]
    assert "sk-real-secret" not in cfg.model_dump_json()


def test_llm_config_get_secret_value_returns_raw() -> None:
    cfg = LLMConfig(model="m", api_key="sk-real-secret", base_url="u")  # type: ignore[arg-type]
    assert cfg.api_key.get_secret_value() == "sk-real-secret"


def test_llm_config_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        LLMConfig(model="m", api_key="k")  # type: ignore[call-arg, arg-type]


def test_llm_config_extra_dict_field_default_is_empty() -> None:
    cfg = LLMConfig(model="m", api_key="k", base_url="u")  # type: ignore[arg-type]
    cfg2 = LLMConfig(model="m", api_key="k", base_url="u")  # type: ignore[arg-type]
    cfg.extra["seed"] = 42
    assert cfg2.extra == {}, "extra default_factory must produce a fresh dict per instance"
