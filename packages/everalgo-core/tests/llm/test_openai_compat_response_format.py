"""Tests for ``OpenAICompatClient``: how ``response_format_mode`` requests the reply format, and parameter precedence.

The transport is mocked with respx, so every assertion reads the JSON body the client really sent.
"""

from __future__ import annotations

import json
from typing import Any, cast

import httpx
import pytest
import respx
from pydantic import BaseModel

from everalgo.llm import StrictOutputModel, to_openai_response_format
from everalgo.llm.openai_compat import OpenAICompatClient
from everalgo.llm.types import LLMMessage

_BASE_URL = "https://api.openai.com/v1"


def _completion(content: str = '{"value": "x"}') -> dict[str, object]:
    return {
        "id": "chatcmpl-ok",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "cfg-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    }


class _Answer(StrictOutputModel):
    value: str


class _WithDict(BaseModel):
    counts: dict[str, int]


def _client(**overrides: Any) -> OpenAICompatClient:
    return OpenAICompatClient(**{"model": "cfg-model", "api_key": "sk-test", "base_url": _BASE_URL, **overrides})


async def _sent_body(
    client: OpenAICompatClient,
    *,
    response_format: type[BaseModel] = _Answer,
    reply: str = '{"value": "x"}',
    **chat_kwargs: Any,
) -> dict[str, Any]:
    """Run one ``achat`` against a mocked upstream and return the JSON body that was sent."""
    with respx.mock(base_url=_BASE_URL) as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=_completion(reply)))
        await client.achat([LLMMessage(role="user", content="hi")], response_format=response_format, **chat_kwargs)
    request = cast("httpx.Request", route.calls[0].request)  # pyright: ignore[reportUnknownMemberType]
    return cast("dict[str, Any]", json.loads(request.content))


async def test_omitting_response_format_is_a_type_error() -> None:
    """The parameter has no default, so an omitted model fails at the call — not silently on the wire."""
    client = _client()

    with pytest.raises(TypeError, match="response_format"):
        await client.achat([LLMMessage(role="user", content="hi")])  # type: ignore[call-arg]  # response_format omitted on purpose


# ---------------------------------------------------------------------------
# The client sends what to_openai_response_format builds for its response_format_mode
# ---------------------------------------------------------------------------


async def test_default_response_format_mode_sends_the_strict_schema_of_the_model() -> None:
    sent = await _sent_body(_client())

    assert sent["response_format"] == to_openai_response_format(_Answer, "json_schema")


async def test_json_object_response_format_mode_sends_json_mode() -> None:
    sent = await _sent_body(_client(response_format_mode="json_object"))

    assert sent["response_format"] == {"type": "json_object"}


async def test_prompt_only_response_format_mode_sends_no_response_format() -> None:
    sent = await _sent_body(_client(response_format_mode="prompt_only"))

    assert "response_format" not in sent


async def test_json_schema_response_format_mode_refuses_a_model_outside_the_strict_subset_before_sending() -> None:
    client = _client()

    with respx.mock(base_url=_BASE_URL, assert_all_called=False) as router:
        route = router.post("/chat/completions").mock(return_value=httpx.Response(200, json=_completion()))
        with pytest.raises(ValueError, match=r"_WithDict\.counts"):
            await client.achat([LLMMessage(role="user", content="hi")], response_format=_WithDict)

    assert not route.called


@pytest.mark.parametrize("response_format_mode", ["json_object", "prompt_only"])
async def test_other_response_format_modes_never_build_a_schema(response_format_mode: str) -> None:
    """Only the strict-schema route needs the model inside the strict subset."""
    sent = await _sent_body(
        _client(response_format_mode=response_format_mode),
        response_format=_WithDict,
        reply='{"counts": {"a": 1}}',
    )

    assert "json_schema" not in json.dumps(sent.get("response_format"))


# ---------------------------------------------------------------------------
# Argument precedence — an explicit argument outranks the client's defaults
# ---------------------------------------------------------------------------


async def test_explicit_arguments_outrank_the_client_defaults() -> None:
    sent = await _sent_body(_client(temperature=0.5, max_tokens=128), temperature=0.0, max_tokens=64)

    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 64


async def test_none_arguments_fall_back_to_the_client_defaults() -> None:
    sent = await _sent_body(_client(temperature=0.35, max_tokens=256))

    assert sent["model"] == "cfg-model"
    assert sent["temperature"] == 0.35
    assert sent["max_tokens"] == 256


async def test_parameters_unset_on_both_sides_are_omitted() -> None:
    sent = await _sent_body(_client())

    assert "temperature" not in sent
    assert "max_tokens" not in sent


async def test_extra_reaches_the_request() -> None:
    sent = await _sent_body(_client(), seed=7, top_p=0.9)

    assert sent["seed"] == 7
    assert sent["top_p"] == 0.9


async def test_extra_remains_the_only_way_to_steer_the_model() -> None:
    """``model`` is not a formal parameter, but a host that genuinely routes per call has somewhere to put it."""
    sent = await _sent_body(_client(model="cfg-model"), model="call-model")

    assert sent["model"] == "call-model"
