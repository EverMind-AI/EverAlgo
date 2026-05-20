"""Tests for OpenAICompatClient.chat — BaseModel response_format parse branch."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from everalgo.llm.config import LLMConfig
from everalgo.llm.errors import LLMError
from everalgo.llm.providers.openai_compat import OpenAICompatClient
from everalgo.llm.types import ChatMessage, ChatResponse

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _Answer(BaseModel):
    """Minimal Pydantic schema used as response_format in tests."""

    value: str


def _build_config(**overrides: object) -> LLMConfig:
    base: dict[str, object] = {
        "model": "gpt-4o-mini",
        "api_key": "sk-test",
        "base_url": "https://api.openai.com/v1",
    }
    base.update(overrides)
    return LLMConfig.model_validate(base)


def _make_parse_completion(
    *,
    parsed: BaseModel | None,
    content: str = "raw content",
    refusal: str | None = None,
    model: str = "gpt-4o-mini",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a MagicMock that mimics openai ParsedChatCompletion structure."""
    message = MagicMock()
    message.parsed = parsed
    message.content = content
    message.refusal = refusal

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    completion = MagicMock()
    completion.choices = [choice]
    completion.model = model
    completion.usage = usage
    return completion


def _make_create_completion(
    *,
    content: str = "plain response",
    model: str = "gpt-4o-mini",
    finish_reason: str = "stop",
    prompt_tokens: int = 7,
    completion_tokens: int = 3,
) -> MagicMock:
    """Build a MagicMock that mimics openai ChatCompletion structure."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    completion = MagicMock()
    completion.choices = [choice]
    completion.model = model
    completion.usage = usage
    return completion


# ---------------------------------------------------------------------------
# Case 1: BaseModel response_format → calls .parse(), returns ChatResponse with parsed
# ---------------------------------------------------------------------------


async def test_basemodel_response_format_calls_beta_parse() -> None:
    """BaseModel response_format must route to client.beta.chat.completions.parse."""
    cfg = _build_config()
    client = OpenAICompatClient(cfg)

    parsed_instance = _Answer(value="hello")
    mock_completion = _make_parse_completion(parsed=parsed_instance, content='{"value": "hello"}')

    mock_parse = AsyncMock(return_value=mock_completion)
    mock_create = AsyncMock()

    with (
        patch.object(client._client.beta.chat.completions, "parse", mock_parse),
        patch.object(client._client.chat.completions, "create", mock_create),
    ):
        resp = await client.chat(
            [ChatMessage(role="user", content="hi")],
            response_format=_Answer,
        )

    mock_parse.assert_awaited_once()
    mock_create.assert_not_awaited()

    assert isinstance(resp, ChatResponse)
    assert resp.parsed is parsed_instance
    assert isinstance(resp.parsed, _Answer)
    assert resp.parsed.value == "hello"
    assert resp.content == '{"value": "hello"}'
    assert resp.model == "gpt-4o-mini"
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 5
    assert resp.finish_reason == "stop"


async def test_basemodel_response_format_passes_schema_to_parse() -> None:
    """The schema class itself must be forwarded as response_format kwarg to .parse()."""
    cfg = _build_config()
    client = OpenAICompatClient(cfg)

    parsed_instance = _Answer(value="x")
    mock_completion = _make_parse_completion(parsed=parsed_instance)
    mock_parse = AsyncMock(return_value=mock_completion)

    with patch.object(client._client.beta.chat.completions, "parse", mock_parse):
        await client.chat(
            [ChatMessage(role="user", content="hi")],
            response_format=_Answer,
        )

    call_kwargs = cast("dict[str, Any]", mock_parse.call_args.kwargs)
    assert call_kwargs.get("response_format") is _Answer


# ---------------------------------------------------------------------------
# Case 3: None response_format → calls .create() without response_format key
# ---------------------------------------------------------------------------


async def test_none_response_format_calls_create_without_key() -> None:
    """No response_format must call chat.completions.create without the key entirely."""
    cfg = _build_config()
    client = OpenAICompatClient(cfg)

    mock_completion = _make_create_completion()
    mock_create = AsyncMock(return_value=mock_completion)
    mock_parse = AsyncMock()

    with (
        patch.object(client._client.chat.completions, "create", mock_create),
        patch.object(client._client.beta.chat.completions, "parse", mock_parse),
    ):
        resp = await client.chat([ChatMessage(role="user", content="hi")])

    mock_create.assert_awaited_once()
    mock_parse.assert_not_awaited()

    call_kwargs = cast("dict[str, Any]", mock_create.call_args.kwargs)
    assert "response_format" not in call_kwargs

    assert isinstance(resp, ChatResponse)
    assert resp.parsed is None


# ---------------------------------------------------------------------------
# Case 4: refusal → raises LLMError
# ---------------------------------------------------------------------------


async def test_parse_refusal_raises_llm_error() -> None:
    """A non-None refusal field must raise LLMError mentioning the refusal text."""
    cfg = _build_config()
    client = OpenAICompatClient(cfg)

    mock_completion = _make_parse_completion(parsed=None, refusal="I cannot comply.")
    mock_parse = AsyncMock(return_value=mock_completion)

    with (
        patch.object(client._client.beta.chat.completions, "parse", mock_parse),
        pytest.raises(LLMError, match="LLM refused"),
    ):
        await client.chat(
            [ChatMessage(role="user", content="hi")],
            response_format=_Answer,
        )


# ---------------------------------------------------------------------------
# Case 5: parsed=None (no refusal) → raises LLMError
# ---------------------------------------------------------------------------


async def test_parse_none_parsed_raises_llm_error() -> None:
    """parsed=None with no refusal must raise LLMError about missing structured output."""
    cfg = _build_config()
    client = OpenAICompatClient(cfg)

    mock_completion = _make_parse_completion(parsed=None, refusal=None)
    mock_parse = AsyncMock(return_value=mock_completion)

    with (
        patch.object(client._client.beta.chat.completions, "parse", mock_parse),
        pytest.raises(LLMError, match="parsed=None"),
    ):
        await client.chat(
            [ChatMessage(role="user", content="hi")],
            response_format=_Answer,
        )
