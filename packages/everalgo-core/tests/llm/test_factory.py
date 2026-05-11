"""Tests for everalgo.llm.factory.build_client."""

import importlib
import sys

from everalgo.llm.config import LLMConfig
from everalgo.llm.factory import build_client
from everalgo.llm.protocols import LLMClient


def _config() -> LLMConfig:
    return LLMConfig.model_validate({"model": "m", "api_key": "k", "base_url": "https://api.openai.com/v1"})


def test_build_client_returns_llm_client_instance() -> None:
    client = build_client(_config())
    assert isinstance(client, LLMClient)


def test_build_client_returns_openai_compat_client() -> None:
    """Without a provider field on LLMConfig the only target is openai_compat."""
    from everalgo.llm.providers.openai_compat import OpenAICompatClient

    client = build_client(_config())
    assert isinstance(client, OpenAICompatClient)


def test_factory_module_does_not_import_provider_eagerly() -> None:
    """``import everalgo.llm.factory`` must not pull openai_compat into sys.modules.

    The lazy import inside ``build_client`` is load-bearing for cold-start
    cost; this test is a regression guard against a maintainer "fixing" it
    by hoisting the import to the top of the module.
    """
    # Force a clean reload of everalgo.llm.factory.
    sys.modules.pop("everalgo.llm.factory", None)
    sys.modules.pop("everalgo.llm.providers.openai_compat", None)

    importlib.import_module("everalgo.llm.factory")

    assert "everalgo.llm.factory" in sys.modules
    assert "everalgo.llm.providers.openai_compat" not in sys.modules, (
        "everalgo.llm.factory must not import openai_compat at import time"
    )
