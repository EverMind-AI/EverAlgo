"""Tests for everalgo.llm.providers.openai_compat.OpenAICompatClient."""

import json
from typing import cast

import httpx
import pytest
import respx

from everalgo.llm.config import LLMConfig
from everalgo.llm.errors import LLMError
from everalgo.llm.providers.openai_compat import OpenAICompatClient
from everalgo.llm.types import ChatMessage, ChatResponse


def _build_config(**overrides: object) -> LLMConfig:
    base: dict[str, object] = {
        "model": "gpt-4o-mini",
        "api_key": "sk-test",
        "base_url": "https://api.openai.com/v1",
    }
    base.update(overrides)
    return LLMConfig.model_validate(base)


@pytest.fixture
def chat_completion_payload() -> dict[str, object]:
    return {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi there"},
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


async def test_openai_compat_client_chat_returns_structured_response(
    chat_completion_payload: dict[str, object],
) -> None:
    cfg = _build_config()
    client = OpenAICompatClient(cfg)

    with respx.mock(base_url=cfg.base_url, assert_all_called=True) as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=chat_completion_payload))
        resp = await client.chat([ChatMessage(role="user", content="hello")])

    assert isinstance(resp, ChatResponse)
    assert resp.content == "hi there"
    assert resp.model == "gpt-4o-mini"
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 7
    assert resp.usage.completion_tokens == 3
    assert resp.finish_reason == "stop"
    assert route.called


async def test_openai_compat_client_uses_config_defaults(
    chat_completion_payload: dict[str, object],
) -> None:
    cfg = _build_config(temperature=0.5, max_tokens=128)
    client = OpenAICompatClient(cfg)

    with respx.mock(base_url=cfg.base_url) as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=chat_completion_payload))
        await client.chat([ChatMessage(role="user", content="hi")])

    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["model"] == "gpt-4o-mini"
    assert sent_body["temperature"] == 0.5
    assert sent_body["max_tokens"] == 128


async def test_openai_compat_client_per_call_overrides_take_precedence(
    chat_completion_payload: dict[str, object],
) -> None:
    cfg = _build_config(temperature=0.5, max_tokens=128)
    client = OpenAICompatClient(cfg)

    with respx.mock(base_url=cfg.base_url) as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=chat_completion_payload))
        await client.chat(
            [ChatMessage(role="user", content="hi")],
            model="gpt-4o",
            temperature=0.0,
            max_tokens=64,
        )

    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["model"] == "gpt-4o"
    assert sent_body["temperature"] == 0.0
    assert sent_body["max_tokens"] == 64


async def test_openai_compat_client_wraps_sdk_error_as_llm_error() -> None:
    cfg = _build_config()
    client = OpenAICompatClient(cfg)

    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/chat/completions").mock(
            return_value=httpx.Response(429, json={"error": {"message": "rate limit"}})
        )
        with pytest.raises(LLMError) as caught:
            await client.chat([ChatMessage(role="user", content="hi")])

    assert caught.value.__cause__ is not None
    # __cause__ must be a subclass of openai.OpenAIError
    import openai

    assert isinstance(caught.value.__cause__, openai.OpenAIError)


async def test_openai_compat_client_normalises_unknown_finish_reason(
    chat_completion_payload: dict[str, object],
) -> None:
    """If the provider emits a finish_reason outside the 3-value Literal, normalise to None."""
    payload = dict(chat_completion_payload)
    choices = list(cast("list[dict[str, object]]", payload["choices"]))
    choices[0] = {**choices[0], "finish_reason": "tool_calls"}
    payload["choices"] = choices

    cfg = _build_config()
    client = OpenAICompatClient(cfg)

    with respx.mock(base_url=cfg.base_url) as router:
        router.post("/chat/completions").mock(return_value=httpx.Response(200, json=payload))
        resp = await client.chat([ChatMessage(role="user", content="hi")])

    assert resp.finish_reason is None
