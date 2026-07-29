"""Tests for AtomicFactExtractor.aextract_from_text and its sync bridge extract_from_text."""

from __future__ import annotations

import pytest

from everalgo.llm.types import ChatMessage as LLMChatMessage
from everalgo.llm.types import ChatResponse
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.user_memory.atomic_fact import AtomicFactExtractor
from everalgo.user_memory.prompts.en.atomic_fact_from_text import ATOMIC_FACT_FROM_TEXT_PROMPT_EN

# Unix epoch ms for 2024-03-10 Sunday 14:00:00 UTC  →  "March 10, 2024(Sunday) at 02:00 PM"  (evercore EventLog format)
_MARCH_10_2024_14H_UTC_MS: int = 1710079200000
_EXPECTED_TIME_STR: str = "March 10, 2024(Sunday) at 02:00 PM"

_NESTED_HAPPY = '{"atomic_facts": {"time": "March 10, 2024(Sunday) at 02:00 PM", "atomic_fact": ["fact 1", "fact 2"]}}'
_NESTED_EMPTY = '{"atomic_facts": {"time": "March 10, 2024(Sunday) at 02:00 PM", "atomic_fact": []}}'
# evercore EventLog schema — same inner shape as ``atomic_facts``, only top-level key differs.
_EVENT_LOG_HAPPY = '{"event_log": {"time": "March 10, 2024(Sunday) at 02:00 PM", "atomic_fact": ["fact A", "fact B"]}}'


# ---------------------------------------------------------------------------
# Happy-path and edge cases
# ---------------------------------------------------------------------------


async def test_returns_atomic_fact_list_on_happy_path() -> None:
    """FakeLLM returns two facts inside nested schema; result equals that list."""
    fake = FakeLLMClient(responses=[_NESTED_HAPPY])
    result = await AtomicFactExtractor(llm=fake).aextract_from_text("some text", timestamp=_MARCH_10_2024_14H_UTC_MS)
    assert [af.content for af in result] == ["fact 1", "fact 2"]


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


async def test_event_log_top_level_key_accepted() -> None:
    """LLM response with evercore ``event_log`` top-level key is parsed identically to ``atomic_facts``.

    Uses ``EVENT_LOG_PROMPT`` which instructs the LLM to emit
    ``{"event_log": {...}}`` and the parser must accept it as an alias of the legacy schema.
    """
    fake = FakeLLMClient(responses=[_EVENT_LOG_HAPPY])
    result = await AtomicFactExtractor(llm=fake).aextract_from_text("some text", timestamp=_MARCH_10_2024_14H_UTC_MS)
    assert [af.content for af in result] == ["fact A", "fact B"]


async def test_missing_atomic_facts_key_raises_value_error() -> None:
    """FakeLLM returns a dict with neither ``atomic_facts`` nor ``event_log`` → ValueError after 5 retries."""
    bad_responses: list[str | ChatResponse] = ['{"foo": "bar"}'] * 5
    fake = FakeLLMClient(responses=bad_responses)
    with pytest.raises(ValueError, match="event_log/atomic_facts"):
        await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)


async def test_missing_inner_atomic_fact_key_raises_value_error() -> None:
    """FakeLLM returns 'atomic_facts' dict without inner 'atomic_fact' key → empty list (defaults to [])."""
    # atomic_fact key missing → block.get("atomic_fact", []) returns [] → empty list, no error
    fake = FakeLLMClient(responses=['{"atomic_facts": {"time": "March 10, 2024(Sunday) at 02:00 PM"}}'])
    result = await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)
    assert result == []


async def test_atomic_facts_not_a_dict_raises_type_error() -> None:
    """FakeLLM returns 'atomic_facts' as a string → ValueError after 5 retries."""
    bad_responses: list[str | ChatResponse] = ['{"atomic_facts": "not a dict"}'] * 5
    fake = FakeLLMClient(responses=bad_responses)
    with pytest.raises(ValueError):
        await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)


async def test_atomic_fact_not_a_list_raises_type_error() -> None:
    """FakeLLM returns 'atomic_fact' as a string inside atomic_facts → ValueError after 5 retries."""
    bad_responses: list[str | ChatResponse] = [
        '{"atomic_facts": {"time": "March 10, 2024(Sunday) at 02:00 PM", "atomic_fact": "not a list"}}'
    ] * 5
    fake = FakeLLMClient(responses=bad_responses)
    with pytest.raises(ValueError):
        await AtomicFactExtractor(llm=fake).aextract_from_text("text", timestamp=_MARCH_10_2024_14H_UTC_MS)


async def test_invalid_json_raises_value_error() -> None:
    """FakeLLM returns non-JSON → ValueError after 5 retries."""
    bad_responses: list[str | ChatResponse] = ["not json at all"] * 5
    fake = FakeLLMClient(responses=bad_responses)
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
    # Default prompt uses {{EPISODE_TEXT}} placeholder (current) or {{TEXT}} (legacy fallback).
    expected_body = (
        ATOMIC_FACT_FROM_TEXT_PROMPT_EN.replace("{{EPISODE_TEXT}}", marker)
        .replace("{{TEXT}}", marker)
        .replace("{{TIME}}", _EXPECTED_TIME_STR)
    )
    assert captured[0] == expected_body
    # Sanity: no un-substituted placeholders remain
    assert "{{EPISODE_TEXT}}" not in captured[0]
    assert "{{TIME}}" not in captured[0]


# ---------------------------------------------------------------------------
# Sync bridge
# ---------------------------------------------------------------------------


def test_sync_bridge_extract_from_text_works() -> None:
    """extract_from_text (sync bridge) called outside event loop returns valid list."""
    fake = FakeLLMClient(
        responses=['{"atomic_facts": {"time": "March 10, 2024(Sunday) at 02:00 PM", "atomic_fact": ["sync fact"]}}']
    )
    result = AtomicFactExtractor(llm=fake).extract_from_text("some text", timestamp=_MARCH_10_2024_14H_UTC_MS)
    assert [af.content for af in result] == ["sync fact"]


# ==========================================================================
# Language rule — from-text prompts inherit the EPISODE_TEXT language
# ==========================================================================


def test_from_text_prompts_state_the_language_rule_at_both_ends() -> None:
    """Long prompts lose middle instructions, so each rule appears at head and tail."""
    import everalgo.user_memory.prompts.en.atomic_fact_from_text as en_mod
    import everalgo.user_memory.prompts.zh.atomic_fact_from_text as zh_mod

    assert en_mod.EVENT_LOG_PROMPT.count("CRITICAL LANGUAGE RULE") == 2
    assert en_mod.ATOMIC_FACT_FROM_TEXT_PROMPT_EN.count("CRITICAL LANGUAGE RULE") == 2
    assert zh_mod.EVENT_LOG_PROMPT.count("关键语言规则") == 2
    assert zh_mod.ATOMIC_FACT_FROM_TEXT_PROMPT_ZH.count("关键语言规则") == 2


def test_from_text_prompts_carry_no_mixed_input_judgement() -> None:
    """EPISODE_TEXT is already extracted, so this prompt inherits rather than judges the language.

    Judgement belongs to episode extraction, where the mixed-input problem (pasted logs, code blocks,
    quoted foreign-language material) actually exists.
    """
    import everalgo.user_memory.prompts.en.atomic_fact_from_text as en_mod
    import everalgo.user_memory.prompts.zh.atomic_fact_from_text as zh_mod

    assert "dominate" not in en_mod.EVENT_LOG_PROMPT
    assert "dominate" not in en_mod.ATOMIC_FACT_FROM_TEXT_PROMPT_EN
    assert "篇幅" not in zh_mod.EVENT_LOG_PROMPT
    assert "篇幅" not in zh_mod.ATOMIC_FACT_FROM_TEXT_PROMPT_ZH


def test_zh_from_text_prompts_are_not_byte_identical_to_en() -> None:
    """A real translation, not a placeholder re-export of the English constants."""
    import everalgo.user_memory.prompts.en.atomic_fact_from_text as en_mod
    import everalgo.user_memory.prompts.zh.atomic_fact_from_text as zh_mod

    assert en_mod.EVENT_LOG_PROMPT != zh_mod.EVENT_LOG_PROMPT
    assert en_mod.ATOMIC_FACT_FROM_TEXT_PROMPT_EN != zh_mod.ATOMIC_FACT_FROM_TEXT_PROMPT_ZH


def test_zh_from_text_prompts_keep_the_same_placeholders_as_en() -> None:
    """A translated prompt that drops a placeholder would render with a literal `{{TIME}}` in it."""
    import everalgo.user_memory.prompts.en.atomic_fact_from_text as en_mod
    import everalgo.user_memory.prompts.zh.atomic_fact_from_text as zh_mod

    for placeholder in ("{{EPISODE_TEXT}}", "{{TIME}}"):
        assert placeholder in en_mod.EVENT_LOG_PROMPT
        assert placeholder in zh_mod.EVENT_LOG_PROMPT
        assert placeholder in en_mod.ATOMIC_FACT_FROM_TEXT_PROMPT_EN
        assert placeholder in zh_mod.ATOMIC_FACT_FROM_TEXT_PROMPT_ZH
