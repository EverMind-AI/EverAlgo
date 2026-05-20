"""Tests for AtomicFactExtractor.aextract_from_text and its sync bridge extract_from_text."""

from __future__ import annotations

import pytest

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.prompts.en.atomic_fact_from_text import ATOMIC_FACT_FROM_TEXT_PROMPT_EN

# Unix epoch ms for 2024-03-10 Sunday 14:00:00 UTC  →  "March 10, 2024 (Sunday) at 2:00 PM UTC"
_MARCH_10_2024_14H_UTC_MS: int = 1710079200000
_EXPECTED_TIME_STR: str = "March 10, 2024 (Sunday) at 2:00 PM UTC"

_NESTED_HAPPY = (
    '{"atomic_facts": {"time": "March 10, 2024 (Sunday) at 2:00 PM UTC", "atomic_fact": ["fact 1", "fact 2"]}}'
)
_NESTED_EMPTY = '{"atomic_facts": {"time": "March 10, 2024 (Sunday) at 2:00 PM UTC", "atomic_fact": []}}'


# ---------------------------------------------------------------------------
# Happy-path and edge cases
# ---------------------------------------------------------------------------


async def test_returns_atomic_fact_list_on_happy_path() -> None:
    """FakeLLM returns two facts inside nested schema; result equals that list."""
    fake = FakeLLMClient(responses=[_NESTED_HAPPY])
    result = await AtomicFactExtractor(llm=fake).aextract_from_text("some text", timestamp=_MARCH_10_2024_14H_UTC_MS)
    assert result == ["fact 1", "fact 2"]


async def test_empty_atomic_fact_list_returns_empty() -> None:
    """FakeLLM returns empty nested list; result is empty list."""
    fake = FakeLLMClient(responses=[_NESTED_EMPTY])
    result = await AtomicFactExtractor(llm=fake).aextract_from_text(
        "greeting only", timestamp=_MARCH_10_2024_14H_UTC_MS
    )
    assert result == []


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


async def test_missing_atomic_facts_key_raises_value_error() -> None:
    """FakeLLM returns a dict without 'atomic_facts' key → ValueError."""
    fake = FakeLLMClient(responses=['{"foo": "bar"}'])
    with pytest.raises(ValueError, match="atomic_facts"):
        await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)


async def test_missing_inner_atomic_fact_key_raises_value_error() -> None:
    """FakeLLM returns 'atomic_facts' dict without inner 'atomic_fact' key → ValueError."""
    fake = FakeLLMClient(responses=['{"atomic_facts": {"time": "March 10, 2024 (Sunday) at 2:00 PM UTC"}}'])
    with pytest.raises(ValueError, match="atomic_fact"):
        await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)


async def test_atomic_facts_not_a_dict_raises_type_error() -> None:
    """FakeLLM returns 'atomic_facts' as a string → TypeError."""
    fake = FakeLLMClient(responses=['{"atomic_facts": "not a dict"}'])
    with pytest.raises(TypeError):
        await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)


async def test_atomic_fact_not_a_list_raises_type_error() -> None:
    """FakeLLM returns 'atomic_fact' as a string inside atomic_facts → TypeError."""
    fake = FakeLLMClient(
        responses=['{"atomic_facts": {"time": "March 10, 2024 (Sunday) at 2:00 PM UTC", "atomic_fact": "not a list"}}']
    )
    with pytest.raises(TypeError):
        await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)


async def test_invalid_json_raises_value_error() -> None:
    """FakeLLM returns non-JSON → ValueError."""
    fake = FakeLLMClient(responses=["not json at all"])
    with pytest.raises(ValueError):
        await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


async def test_timestamp_formatted_in_prompt() -> None:
    """_MARCH_10_2024_14H_UTC_MS renders to _EXPECTED_TIME_STR in the sent prompt."""
    captured: list[str] = []

    def handler(messages: list[LLMChatMessage], **_: object) -> ChatResponse:
        captured.append(str(messages[0].content))
        return ChatResponse(content=_NESTED_EMPTY, model="fake")

    fake = FakeLLMClient(handler=handler)
    await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)
    assert _EXPECTED_TIME_STR in captured[0], (
        f"Expected time string {_EXPECTED_TIME_STR!r} not found in rendered prompt"
    )


async def test_text_substituted_in_prompt() -> None:
    """Unique marker in text argument appears verbatim in the rendered prompt."""
    captured: list[str] = []

    def handler(messages: list[LLMChatMessage], **_: object) -> ChatResponse:
        captured.append(str(messages[0].content))
        return ChatResponse(content=_NESTED_EMPTY, model="fake")

    fake = FakeLLMClient(handler=handler)
    await AtomicFactExtractor(llm=fake).aextract_from_text("UNIQUE_TEXT_MARKER_42", timestamp=_MARCH_10_2024_14H_UTC_MS)
    assert "UNIQUE_TEXT_MARKER_42" in captured[0]


async def test_prompt_override_used() -> None:
    """Custom prompt template is used instead of the default."""
    captured: list[str] = []

    def handler(messages: list[LLMChatMessage], **_: object) -> ChatResponse:
        captured.append(str(messages[0].content))
        return ChatResponse(content=_NESTED_EMPTY, model="fake")

    fake = FakeLLMClient(handler=handler)
    await AtomicFactExtractor(llm=fake).aextract_from_text(
        "hello", timestamp=_MARCH_10_2024_14H_UTC_MS, prompt="custom {{TEXT}} {{TIME}}"
    )
    assert captured[0] == f"custom hello {_EXPECTED_TIME_STR}"


async def test_default_prompt_is_atomic_fact_from_text_prompt_en() -> None:
    """prompt=None must use ATOMIC_FACT_FROM_TEXT_PROMPT_EN with placeholders filled."""
    captured: list[str] = []

    def handler(messages: list[LLMChatMessage], **_: object) -> ChatResponse:
        captured.append(str(messages[0].content))
        return ChatResponse(content=_NESTED_EMPTY, model="fake")

    fake = FakeLLMClient(handler=handler)
    marker = "MARKER_DEFAULT_PROMPT_CHECK"
    await AtomicFactExtractor(llm=fake).aextract_from_text(marker, timestamp=_MARCH_10_2024_14H_UTC_MS, prompt=None)
    expected_body = ATOMIC_FACT_FROM_TEXT_PROMPT_EN.replace("{{TEXT}}", marker).replace("{{TIME}}", _EXPECTED_TIME_STR)
    assert captured[0] == expected_body
    # Sanity: no un-substituted placeholders remain
    assert "{{TEXT}}" not in captured[0]
    assert "{{TIME}}" not in captured[0]


# ---------------------------------------------------------------------------
# Sync bridge
# ---------------------------------------------------------------------------


def test_sync_bridge_extract_from_text_works() -> None:
    """extract_from_text (sync bridge) called outside event loop returns valid list."""
    fake = FakeLLMClient(
        responses=['{"atomic_facts": {"time": "March 10, 2024 (Sunday) at 2:00 PM UTC", "atomic_fact": ["sync fact"]}}']
    )
    result = AtomicFactExtractor(llm=fake).extract_from_text("some text", timestamp=_MARCH_10_2024_14H_UTC_MS)
    assert result == ["sync fact"]
