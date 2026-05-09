"""Tests for evercore.llm.factory.build_client."""

import importlib
import sys

from evercore.llm.config import LLMConfig
from evercore.llm.factory import build_client
from evercore.llm.protocols import LLMClient


def _config() -> LLMConfig:
    return LLMConfig.model_validate({"model": "m", "api_key": "k", "base_url": "https://api.openai.com/v1"})


def test_build_client_returns_llm_client_instance() -> None:
    client = build_client(_config())
    assert isinstance(client, LLMClient)


def test_build_client_returns_openai_compat_client() -> None:
    """Without a provider field on LLMConfig the only target is openai_compat."""
    from evercore.llm.providers.openai_compat import OpenAICompatClient

    client = build_client(_config())
    assert isinstance(client, OpenAICompatClient)


def test_factory_module_does_not_import_provider_eagerly() -> None:
    """``import evercore.llm.factory`` must not pull openai_compat into sys.modules.

    The lazy import inside ``build_client`` is load-bearing for cold-start
    cost; this test is a regression guard against a maintainer "fixing" it
    by hoisting the import to the top of the module.
    """
    # Force a clean reload of evercore.llm.factory.
    sys.modules.pop("evercore.llm.factory", None)
    sys.modules.pop("evercore.llm.providers.openai_compat", None)

    importlib.import_module("evercore.llm.factory")

    assert "evercore.llm.factory" in sys.modules
    assert "evercore.llm.providers.openai_compat" not in sys.modules, (
        "evercore.llm.factory must not import openai_compat at import time"
    )
