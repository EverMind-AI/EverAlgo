"""Pytest fixtures for live-LLM functional tests.

All tests in this directory should be marked ``@pytest.mark.integration`` and
take the ``real_llm`` fixture. Run them explicitly with::

    pytest -m integration packages/everalgo-knowledge/tests/functional/

The fixture reads the LLM endpoint from three env vars:

* ``LLM_API_KEY``
* ``LLM_BASE_URL``  (any OpenAI-compatible endpoint, e.g. OpenRouter)
* ``LLM_MODEL``     (e.g. ``anthropic/claude-sonnet-4-6``)

Missing any of the three triggers ``pytest.skip`` so the test does not falsely
fail when credentials are unavailable.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from pydantic import SecretStr

from everalgo.llm import build_client
from everalgo.llm.config import LLMConfig

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient


_REQUIRED_ENV_VARS = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")


@pytest.fixture(scope="session")
def real_llm() -> LLMClient:
    """OpenAI-compatible client built from environment variables.

    Skips with a clear message when any of ``LLM_API_KEY`` / ``LLM_BASE_URL`` /
    ``LLM_MODEL`` is unset.
    """
    missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        pytest.skip(f"integration test needs env vars: {', '.join(missing)}")
    return build_client(
        LLMConfig(
            api_key=SecretStr(os.environ["LLM_API_KEY"]),
            base_url=os.environ["LLM_BASE_URL"],
            model=os.environ["LLM_MODEL"],
        ),
    )
